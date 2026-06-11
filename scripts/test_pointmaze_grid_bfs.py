#!/usr/bin/env python
"""Synthetic checks for PointMaze grid BFS utilities."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.pointmaze_grid import (
    free_cell_distance_matrix,
    grid_distance_statistics,
    xy_pair_grid_distances,
)


def main():
    maze_map = np.asarray(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    free_cells, cell_distances = free_cell_distance_matrix(maze_map)
    stats = grid_distance_statistics(cell_distances, steps_per_cell=10.0)
    assert stats["max_cells"] == 6
    assert stats["max_steps"] == 60.0

    starts = np.asarray([[-4.0 + 1 * 4.0, -4.0 + 1 * 4.0]], dtype=np.float32)
    goals = np.asarray([[-4.0 + 3 * 4.0, -4.0 + 1 * 4.0]], dtype=np.float32)
    distances = xy_pair_grid_distances(
        starts,
        goals,
        maze_map,
        free_cells,
        cell_distances,
        steps_per_cell=1.0,
    )
    assert distances.shape == (1,)
    assert distances[0] == 6.0

    print("PointMaze grid BFS checks passed.")


if __name__ == "__main__":
    main()
