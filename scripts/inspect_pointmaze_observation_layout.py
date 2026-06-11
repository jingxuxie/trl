#!/usr/bin/env python
"""Inspect PointMaze observation layout and dataset transition statistics."""

from pathlib import Path
import sys

import numpy as np
from absl import app, flags

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets
from utils.pointmaze_graph import dataset_xy, median_step_xy, parse_xy_dims


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "pointmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_string("dataset_dir", None, "Optional OGBench dataset directory.")
flags.DEFINE_string("xy_dims", "0,1", "Comma-separated observation dims to use as xy.")
flags.DEFINE_integer("num_rows", 10, "Number of rows to print.")
flags.DEFINE_integer("range_sample_size", 200000, "Rows used for dim range summaries.")


def dataset_path_from_dir(dataset_dir):
    if dataset_dir is None:
        return None
    candidates = [
        str(path)
        for path in sorted(Path(dataset_dir).glob("*.npz"))
        if "-val.npz" not in path.name
    ]
    return candidates[0] if candidates else None


def unwrap_chain(env):
    chain = []
    cur = env
    for _ in range(16):
        chain.append(cur)
        if not hasattr(cur, "env"):
            break
        cur = cur.env
    return chain


def summarize_dataset(name, dataset, xy_dims):
    observations = np.asarray(dataset["observations"])
    actions = np.asarray(dataset["actions"])
    terminals = np.asarray(dataset["terminals"])
    valids = np.asarray(dataset["valids"]) if "valids" in dataset else None
    sample_size = min(len(observations), int(FLAGS.range_sample_size))
    sample_idxs = np.linspace(0, len(observations) - 1, sample_size).astype(np.int64)
    sample = observations[sample_idxs]
    ranges = np.stack([sample.min(axis=0), sample.max(axis=0)], axis=-1)

    print(f"\n{name} dataset")
    print(f"  observations: {observations.shape}")
    print(f"  actions:      {actions.shape}")
    print(f"  terminals:    {int(terminals.sum())}")
    if valids is not None:
        print(f"  valid transitions: {int(valids.sum())}")
    print("  observation dim ranges from sampled rows:")
    for dim, (lo, hi) in enumerate(ranges):
        print(f"    dim {dim}: [{lo:.4f}, {hi:.4f}]")
    print(f"  first {FLAGS.num_rows} observations:")
    print(observations[: FLAGS.num_rows])
    print(f"  first {FLAGS.num_rows} actions:")
    print(actions[: FLAGS.num_rows])
    xy = dataset_xy(dataset, xy_dims)
    valid_idxs = np.nonzero(valids > 0)[0] if valids is not None else np.arange(len(xy) - 1)
    valid_idxs = valid_idxs[valid_idxs < len(xy) - 1]
    deltas = xy[valid_idxs + 1] - xy[valid_idxs]
    norms = np.linalg.norm(deltas, axis=-1)
    norms = norms[norms > 1e-8]
    print(f"  xy dims: {xy_dims}")
    print(f"  xy range: min={xy.min(axis=0)}, max={xy.max(axis=0)}")
    print(f"  median nonzero one-step xy displacement: {np.median(norms):.6f}")
    print(f"  first {FLAGS.num_rows} valid xy deltas:")
    print(deltas[: FLAGS.num_rows])


def print_env_layout(env):
    print("\nEnvironment unwrap chain")
    for depth, wrapped in enumerate(unwrap_chain(env)):
        attrs = [
            name
            for name in dir(wrapped)
            if "maze" in name.lower()
            or "map" in name.lower()
            or "wall" in name.lower()
            or "layout" in name.lower()
        ]
        print(f"  {depth}: {type(wrapped)} attrs={attrs[:12]}")
        if hasattr(wrapped, "maze_map"):
            print("  maze_map:")
            print(getattr(wrapped, "maze_map"))
        for attr in ("_maze_type", "_maze_unit", "_maze_height"):
            if hasattr(wrapped, attr):
                print(f"  {attr}: {getattr(wrapped, attr)}")


def main(_):
    xy_dims = parse_xy_dims(FLAGS.xy_dims)
    dataset_path = dataset_path_from_dir(FLAGS.dataset_dir)
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, dataset_path=dataset_path
    )
    print(f"env_name: {FLAGS.env_name}")
    print_env_layout(env)
    summarize_dataset("train", train_dataset, xy_dims)
    summarize_dataset("validation", val_dataset, xy_dims)
    median_step = median_step_xy((train_dataset, val_dataset), xy_dims)
    print(f"\ncombined median nonzero one-step xy displacement: {median_step:.6f}")
    print("Conclusion: for PointMaze xy-only observations, dims (0, 1) are xy.")


if __name__ == "__main__":
    app.run(main)
