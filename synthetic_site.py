"""Generate synthetic aggregate power and channel truth."""
import argparse
import pathlib

import numpy as np
import yaml

from utils import load_config, write_csv

HERE = pathlib.Path(__file__).resolve().parent


def transition_matrix(dwell):
    """Uniform destinations on leaving a state, with its specified mean dwell."""
    q = 1.0 / np.asarray(dwell, dtype=float)
    count = len(q)
    matrix = np.repeat((q / (count - 1))[:, None], count, axis=1)
    np.fill_diagonal(matrix, 1.0 - q)
    return matrix


def generate_day(seed, cfg, day=0):
    """Generate load states, noisy channel power, background and aggregate."""
    size = cfg["minutes_per_day"]
    streams = np.random.SeedSequence([seed, day]).spawn(len(cfg["channels"]) + 2)
    paths = []
    for index, channel in enumerate(cfg["channels"]):
        rng = np.random.default_rng(streams[index])
        path = np.zeros(size, dtype=int)
        if channel["type"] == "markov":
            trans = transition_matrix(channel["dwell_minutes"])
            path[0] = rng.integers(len(trans))
            for t in range(1, size):
                path[t] = rng.choice(len(trans), p=trans[path[t - 1]])
        else:
            jitter = channel.get("jitter_minutes", 0)
            switches = np.asarray(channel["switch_minutes"]) + rng.integers(
                -jitter, jitter + 1, len(channel["switch_minutes"]))
            edges = [0] + switches.tolist() + [size]
            for start, stop, state in zip(edges, edges[1:], channel["states"]):
                path[start:stop] = state
        paths.append(path)
    states = np.array(paths).T
    levels = np.column_stack([np.asarray(c["levels_w"])[states[:, k]]
                              for k, c in enumerate(cfg["channels"])])
    rng = np.random.default_rng(streams[len(paths)])
    power = levels + rng.normal(0, cfg["noise_w"], levels.shape)
    # Background alternates between plateaus and ramps.
    bg_cfg = cfg["background"]
    rng = np.random.default_rng(streams[len(paths) + 1])
    background = np.empty(size)
    level = rng.uniform(*bg_cfg["range_w"])
    t = 0
    while t < size:
        stop = min(size, t + rng.integers(bg_cfg["hold_minutes"][0],
                                         bg_cfg["hold_minutes"][1] + 1))
        target = rng.uniform(*bg_cfg["range_w"])
        background[t:stop] = (np.linspace(level, target, stop - t)
                              if rng.random() < 0.5 else target)
        level, t = target, stop
    return {"states": states, "power": power, "background": background,
            "aggregate": power.sum(axis=1) + background}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=HERE / "config.yaml")
    parser.add_argument("--out", type=pathlib.Path, default=HERE / "days")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        write_csv(args.out / "commissioning.csv", cfg, generate_day(cfg["seed"], cfg))
        for index in range(1, cfg["days"] + 1):
            day = generate_day(cfg["seed"], cfg, day=index)
            write_csv(args.out / f"day_{index:03d}.csv", cfg, day)
        print(f"Wrote one commissioning day and {cfg['days']} synthetic days to {args.out}")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
