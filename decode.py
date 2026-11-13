"""Decode aggregate power using a synthetic commissioning day."""
import argparse
import math
import pathlib

import numpy as np
import yaml

from synthetic_site import generate_day
from utils import load_config, read_aggregate, write_csv

HERE = pathlib.Path(__file__).resolve().parent


def check_size(cards, steps):
    """Reject oversized backpointers and transition arrays before allocation."""
    size = math.prod(cards)
    if max(steps * len(cards) * size, size * max(cards)) > 25_000_000:
        raise ValueError("decoder state space is too large; use fewer channels, "
                         "fewer levels, a coarser grid or a shorter input")


def product_states(cards):
    """Product indices, with the first chain varying fastest."""
    index = np.arange(int(np.prod(cards)))
    return np.column_stack([(index // int(np.prod(cards[:k]))) % c
                            for k, c in enumerate(cards)])


def viterbi(y, transitions, levels, sigma, initials=None):
    """Exact factorial MAP using one max-plus contraction per chain.

    Implementation described in Mostafavi and Cox, Section 3.3:
    https://arxiv.org/abs/2610.15247
    """
    cards = [len(m) for m in levels]
    check_size(cards, len(y))
    size, chains = math.prod(cards), len(cards)
    states = product_states(cards)
    means = sum(m[states[:, k]] for k, m in enumerate(levels))
    initial = (-sum(np.log(c) for c in cards) if initials is None else
               sum(p[states[:, k]] for k, p in enumerate(initials)))
    delta = initial - 0.5 * ((y[0] - means) / sigma) ** 2
    shape = tuple(reversed(cards))
    dtype = np.uint8 if max(cards) <= 256 else np.int32
    back = np.zeros((len(y), chains, size), dtype=dtype)
    for t in range(1, len(y)):
        tensor = delta.reshape(shape)
        for k, count in enumerate(cards):
            axis = chains - 1 - k
            moved = np.moveaxis(tensor, axis, 0)
            scores = moved[None, ...] + transitions[k].T.reshape(
                (count, count) + (1,) * (moved.ndim - 1))
            arg = np.argmax(scores, axis=1)
            best = np.take_along_axis(scores, arg[:, None, ...], axis=1)[:, 0]
            tensor = np.moveaxis(best, 0, axis)
            back[t, k] = np.moveaxis(arg, 0, axis).reshape(-1)
        delta = tensor.reshape(-1) - 0.5 * ((y[t] - means) / sigma) ** 2
    place = np.cumprod([1] + cards[:-1])
    path = np.zeros(len(y), dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(len(y) - 1, 0, -1):
        z = states[path[t]].copy()
        for k in range(chains - 1, -1, -1):
            z[k] = back[t, k, int(z @ place)]
        path[t - 1] = int(z @ place)
    return states[path]


def calibrate(cfg):
    """Estimate decoder parameters from one generated labeled day."""
    day = generate_day(cfg["seed"], cfg)
    levels, transitions = [], []
    residual = day["power"].copy()
    for k, channel in enumerate(cfg["channels"]):
        path = day["states"][:, k]
        means = []
        for state in range(len(channel["levels_w"])):
            observed = path == state
            if not observed.any():
                raise ValueError(f"calibration never observes {channel['name']} state {state}; "
                                 "use a longer day or another seed")
            means.append(float(day["power"][observed, k].mean()))
        means = np.array(means)
        residual[:, k] -= means[path]
        counts = np.ones((len(means), len(means)))  # additive smoothing
        np.add.at(counts, (path[:-1], path[1:]), 1)
        transitions.append(np.log(counts / counts.sum(axis=1, keepdims=True)))
        levels.append(means)
    step = cfg["decoder"]["grid_step_w"]
    low, high = float(day["background"].min()), float(day["background"].max())
    bins = max(2, math.ceil((high - low) / step) + 1)
    check_size([bins], 1)
    grid = np.linspace(low, high, bins)
    return {"levels": levels, "transitions": transitions, "grid": grid,
            "offset": float(day["background"].mean()),
            "sigma": max(float(residual.sum(axis=1).std()), 1.0)}


def decode(aggregate, cfg, model, restricted=False, background="dynamic"):
    """Decode aggregate power only; the input's truth columns are never read."""
    y = np.asarray(aggregate, dtype=float)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all():
        raise ValueError("aggregate must be a nonempty finite vector")
    if background not in ("dynamic", "constant"):
        raise ValueError("background must be dynamic or constant")
    cards = [len(c.get("support_order") or m) if restricted else len(m)
             for c, m in zip(cfg["channels"], model["levels"])]
    check_size(cards + [len(model["grid"]) if background == "dynamic" else 1], len(y))
    levels = list(model["levels"])
    transitions = list(model["transitions"])
    initials = [np.full(len(m), -np.log(len(m))) for m in levels]
    maps = [np.arange(len(m)) for m in levels]
    for k, channel in enumerate(cfg["channels"]):
        order = channel.get("support_order") if restricted else None
        if order:
            maps[k] = np.asarray(order)
            levels[k] = levels[k][maps[k]]
            initials[k] = initials[k][maps[k]]
            # Keep original costs; forward jumps allow empty phases.
            trans = transitions[k][np.ix_(maps[k], maps[k])].copy()
            trans[np.tril_indices(len(order), -1)] = -np.inf
            transitions[k] = trans  # do not renormalize after removing edges
    if background == "dynamic":
        grid = model["grid"]
        log_walk = -0.5 * ((grid[:, None] - grid[None, :]) /
                           cfg["decoder"]["step_scale_w"]) ** 2
        log_walk -= np.log(np.exp(log_walk).sum(axis=1, keepdims=True))
    else:
        grid, log_walk = np.array([model["offset"]]), np.zeros((1, 1))
    decoded = viterbi(y, transitions + [log_walk], levels + [grid], model["sigma"],
                      initials + [np.full(len(grid), -np.log(len(grid)))])
    states = np.column_stack([mapping[decoded[:, k]] for k, mapping in enumerate(maps)])
    power = np.column_stack([m[states[:, k]] for k, m in enumerate(model["levels"])])
    return {"states": states, "power": power, "background": grid[decoded[:, -1]],
            "aggregate": power.sum(axis=1) + grid[decoded[:, -1]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=HERE / "config.yaml")
    parser.add_argument("--input", type=pathlib.Path, required=True, help="CSV containing aggregate_w")
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "decoded.csv")
    parser.add_argument("--restricted", action="store_true", help="apply configured support_order rules")
    parser.add_argument("--background", choices=["dynamic", "constant"], default="dynamic")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        y = read_aggregate(args.input)
        day = decode(y, cfg, calibrate(cfg), args.restricted, args.background)
        write_csv(args.out, cfg, day, y)
        print(f"Wrote decoded estimates to {args.out}")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
