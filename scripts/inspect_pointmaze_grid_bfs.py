#!/usr/bin/env python
"""Inspect PointMaze layout/grid BFS distances and budget coverage."""

from pathlib import Path
import sys

import numpy as np
from absl import app, flags

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets
from utils.pointmaze_graph import dataset_xy, median_step_xy, parse_xy_dims
from utils.pointmaze_grid import (
    free_cell_distance_matrix,
    grid_distance_statistics,
    unwrap_maze_env,
    xy_to_ij,
)


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "pointmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_string("dataset_dir", None, "Optional OGBench dataset directory.")
flags.DEFINE_string("xy_dims", "0,1", "Comma-separated observation dims to use as xy.")
flags.DEFINE_string("budgets", "32,64,96,128,160,256,512", "Budgets to inspect.")


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def dataset_path_from_dir(dataset_dir):
    if dataset_dir is None:
        return None
    candidates = [
        str(path)
        for path in sorted(Path(dataset_dir).glob("*.npz"))
        if "-val.npz" not in path.name
    ]
    return candidates[0] if candidates else None


def cell_coverage(dataset, xy_dims, maze_env):
    xy = dataset_xy(dataset, xy_dims)
    ij = xy_to_ij(
        xy,
        maze_unit=maze_env._maze_unit,
        offset_x=maze_env._offset_x,
        offset_y=maze_env._offset_y,
    )
    in_bounds = (
        (ij[:, 0] >= 0)
        & (ij[:, 0] < maze_env.maze_map.shape[0])
        & (ij[:, 1] >= 0)
        & (ij[:, 1] < maze_env.maze_map.shape[1])
    )
    free = np.zeros(len(ij), dtype=bool)
    free[in_bounds] = maze_env.maze_map[ij[in_bounds, 0], ij[in_bounds, 1]] == 0
    cells = np.unique(ij[free], axis=0)
    return dict(
        total=int(len(ij)),
        in_bounds=int(in_bounds.sum()),
        free=int(free.sum()),
        wall_or_oob=int((~free).sum()),
        occupied_free_cells=int(len(cells)),
    )


def main(_):
    xy_dims = parse_xy_dims(FLAGS.xy_dims)
    budgets = parse_int_list(FLAGS.budgets)
    dataset_path = dataset_path_from_dir(FLAGS.dataset_dir)
    env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, dataset_path=dataset_path
    )
    maze_env = unwrap_maze_env(env)
    median_step = median_step_xy((train_dataset, val_dataset), xy_dims)
    steps_per_cell = float(maze_env._maze_unit) / median_step
    free_cells, cell_distances = free_cell_distance_matrix(maze_env.maze_map)
    stats = grid_distance_statistics(cell_distances, steps_per_cell)
    finite = cell_distances >= 0
    finite_steps = cell_distances[finite].astype(np.float32) * steps_per_cell

    print("PointMaze grid BFS diagnostic")
    print(f"  env_name: {FLAGS.env_name}")
    print(f"  maze_type: {maze_env._maze_type}")
    print(f"  maze_unit: {maze_env._maze_unit}")
    print(f"  xy_dims: {xy_dims}")
    print(f"  median_step_xy: {median_step:.6f}")
    print(f"  steps_per_cell: {steps_per_cell:.4f}")
    print(f"  free_cells: {len(free_cells)}")
    print(f"  max grid distance: {stats['max_cells']} cells / {stats['max_steps']:.2f} steps")
    print(
        "  mean finite grid distance: "
        f"{stats['mean_cells']:.2f} cells / {stats['mean_steps']:.2f} steps"
    )
    print("  maze_map:")
    print(maze_env.maze_map)

    print("\nDataset cell coverage")
    for name, dataset in (("train", train_dataset), ("validation", val_dataset)):
        coverage = cell_coverage(dataset, xy_dims, maze_env)
        print(
            f"  {name}: free={coverage['free']}/{coverage['total']} "
            f"wall_or_oob={coverage['wall_or_oob']} "
            f"occupied_free_cells={coverage['occupied_free_cells']}"
        )

    print("\nBudget class coverage over free-cell pairs")
    print("H | positive cell pairs | negative cell pairs")
    print("--|---------------------|--------------------")
    for budget in budgets:
        pos = int((finite_steps <= budget).sum())
        neg = int((finite_steps > budget).sum())
        print(f"{budget:4d} | {pos:19d} | {neg:18d}")


if __name__ == "__main__":
    app.run(main)
