"""Configuration validation and CSV input/output."""
import csv
import pathlib

import numpy as np
import yaml


def load_config(path):
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    """Check channel, timing and decoder settings."""
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML mapping")
    for key in ("seed", "days", "minutes_per_day"):
        if type(cfg.get(key)) is not int or cfg[key] < (1 if key != "seed" else 0):
            raise ValueError(f"{key} must be a positive integer (seed may be zero)")
    channels = cfg.get("channels", [])
    if not channels:
        raise ValueError("channels must contain at least one load")
    names = [c["name"] for c in channels]
    if any(not isinstance(n, str) or not n.isidentifier() for n in names) or len(set(names)) != len(names):
        raise ValueError("channel names must be unique identifiers")
    reserved = {"minute", "aggregate_w", "background_w", "reconstruction_w", "residual_w"}
    fields = [f"{n}_{suffix}" for n in names for suffix in ("w", "state")]
    if reserved.intersection(fields):
        raise ValueError("channel name conflicts with an output column")
    noise = cfg.get("noise_w", 0)
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("noise_w must be finite and positive")
    for c in channels:
        levels = np.asarray(c["levels_w"], dtype=float)
        if levels.ndim != 1 or levels.size < 2 or not np.isfinite(levels).all() or np.any(np.diff(levels) <= 0):
            raise ValueError(f"{c['name']}: levels_w must be finite and increasing")
        if c["type"] == "markov":
            dwell = np.asarray(c["dwell_minutes"], dtype=float)
            if dwell.shape != levels.shape or not np.isfinite(dwell).all() or np.any(dwell < 1):
                raise ValueError(f"{c['name']}: one dwell_minutes >= 1 per level is required")
        elif c["type"] == "schedule":
            switches, states = c["switch_minutes"], c["states"]
            if (not switches or any(type(t) is not int for t in switches)
                    or any(a >= b for a, b in zip([0] + switches, switches + [cfg["minutes_per_day"]]))):
                raise ValueError(f"{c['name']}: switch_minutes must increase inside the day")
            if len(states) != len(switches) + 1 or any(type(s) is not int or not 0 <= s < len(levels) for s in states):
                raise ValueError(f"{c['name']}: one valid state per schedule block is required")
            jitter = c.get("jitter_minutes", 0)
            if type(jitter) is not int or jitter < 0:
                raise ValueError(f"{c['name']}: jitter_minutes must be a nonnegative integer")
            gaps = np.diff([0] + switches + [cfg["minutes_per_day"]])
            if jitter * 2 >= min(gaps):
                raise ValueError(f"{c['name']}: jitter is too large for the schedule blocks")
        else:
            raise ValueError(f"{c['name']}: type must be markov or schedule")
        order = c.get("support_order")
        if order is not None and (not order or any(type(s) is not int or not 0 <= s < len(levels) for s in order)):
            raise ValueError(f"{c['name']}: support_order must contain valid states")
    bg = cfg["background"]
    lo, hi = bg["range_w"]
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("background.range_w must be finite and increasing")
    hold = bg["hold_minutes"]
    if len(hold) != 2 or any(type(v) is not int for v in hold) or not 1 <= hold[0] <= hold[1]:
        raise ValueError("background.hold_minutes must be a positive inclusive range")
    for key in ("grid_step_w", "step_scale_w"):
        if not np.isfinite(cfg["decoder"][key]) or cfg["decoder"][key] <= 0:
            raise ValueError(f"decoder.{key} must be finite and positive")


def write_csv(path, cfg, day, observations=None):
    """Save generated truth or decoded estimates in the same channel order."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [c["name"] for c in cfg["channels"]]
    header = ["minute", "aggregate_w", "background_w"]
    header += [f"{name}_w" for name in names] + [f"{name}_state" for name in names]
    if observations is not None:
        header += ["reconstruction_w", "residual_w"]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for t, total in enumerate(day["aggregate"]):
            measured = total if observations is None else observations[t]
            row = [t, round(float(measured), 3), round(float(day["background"][t]), 3)]
            row += [round(float(v), 3) for v in day["power"][t]]
            row += [int(v) for v in day["states"][t]]
            if observations is not None:
                row += [round(float(total), 3), round(float(measured - total), 3)]
            writer.writerow(row)


def read_aggregate(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if "aggregate_w" not in (reader.fieldnames or []):
            raise ValueError("input CSV must contain aggregate_w")
        return np.array([float(row["aggregate_w"]) for row in reader])


