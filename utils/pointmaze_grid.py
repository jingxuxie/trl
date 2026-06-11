"""Layout/grid BFS utilities for PointMaze reachability diagnostics."""

from collections import deque

import numpy as np


def unwrap_maze_env(env):
    """Return the innermost env object exposing maze_map."""
    cur = env
    for _ in range(32):
        if hasattr(cur, "maze_map"):
            return cur
        if not hasattr(cur, "env"):
            break
        cur = cur.env
    raise ValueError("Could not find an unwrapped env with maze_map.")


def xy_to_ij(xy, maze_unit=4.0, offset_x=4.0, offset_y=4.0):
    """Vectorized version of OGBench MazeEnv.xy_to_ij."""
    xy = np.asarray(xy, dtype=np.float32)
    i = ((xy[..., 1] + offset_y + 0.5 * maze_unit) / maze_unit).astype(np.int32)
    j = ((xy[..., 0] + offset_x + 0.5 * maze_unit) / maze_unit).astype(np.int32)
    return np.stack([i, j], axis=-1)


def ij_to_xy(ij, maze_unit=4.0, offset_x=4.0, offset_y=4.0):
    """Vectorized version of OGBench MazeEnv.ij_to_xy."""
    ij = np.asarray(ij, dtype=np.float32)
    x = ij[..., 1] * maze_unit - offset_x
    y = ij[..., 0] * maze_unit - offset_y
    return np.stack([x, y], axis=-1)


def free_cell_distance_matrix(maze_map):
    """Return all-pairs BFS distances between free cells."""
    maze_map = np.asarray(maze_map)
    free_cells = np.argwhere(maze_map == 0).astype(np.int32)
    cell_to_idx = {tuple(cell): idx for idx, cell in enumerate(free_cells)}
    distances = np.full((len(free_cells), len(free_cells)), -1, dtype=np.int32)

    for src_idx, src_cell in enumerate(free_cells):
        src = tuple(int(x) for x in src_cell)
        distances[src_idx, src_idx] = 0
        queue = deque([src])
        while queue:
            i, j = queue.popleft()
            cur_idx = cell_to_idx[(i, j)]
            next_distance = distances[src_idx, cur_idx] + 1
            for di, dj in ((-1, 0), (0, -1), (1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                neighbor = (ni, nj)
                if (
                    0 <= ni < maze_map.shape[0]
                    and 0 <= nj < maze_map.shape[1]
                    and maze_map[ni, nj] == 0
                    and neighbor in cell_to_idx
                ):
                    neighbor_idx = cell_to_idx[neighbor]
                    if distances[src_idx, neighbor_idx] < 0:
                        distances[src_idx, neighbor_idx] = next_distance
                        queue.append(neighbor)
    return free_cells, distances


def xy_pair_grid_distances(
    start_xy,
    goal_xy,
    maze_map,
    free_cells,
    cell_distances,
    steps_per_cell,
    maze_unit=4.0,
    offset_x=4.0,
    offset_y=4.0,
):
    """Return calibrated grid distances for xy pairs; invalid pairs are inf."""
    start_ij = xy_to_ij(start_xy, maze_unit, offset_x, offset_y)
    goal_ij = xy_to_ij(goal_xy, maze_unit, offset_x, offset_y)
    cell_to_idx = {tuple(cell): idx for idx, cell in enumerate(np.asarray(free_cells))}
    result = np.full(start_ij.shape[:-1], np.inf, dtype=np.float32)
    flat_result = result.reshape(-1)
    flat_start = start_ij.reshape(-1, 2)
    flat_goal = goal_ij.reshape(-1, 2)
    maze_map = np.asarray(maze_map)

    for idx, (src, dst) in enumerate(zip(flat_start, flat_goal)):
        src = tuple(int(x) for x in src)
        dst = tuple(int(x) for x in dst)
        if (
            src not in cell_to_idx
            or dst not in cell_to_idx
            or maze_map[src] != 0
            or maze_map[dst] != 0
        ):
            continue
        cell_distance = cell_distances[cell_to_idx[src], cell_to_idx[dst]]
        if cell_distance >= 0:
            flat_result[idx] = float(cell_distance) * float(steps_per_cell)
    return result


def grid_distance_statistics(cell_distances, steps_per_cell):
    """Summarize finite grid distances."""
    finite = cell_distances >= 0
    max_cells = int(cell_distances[finite].max()) if finite.any() else 0
    mean_cells = float(cell_distances[finite].mean()) if finite.any() else np.nan
    return dict(
        max_cells=max_cells,
        max_steps=float(max_cells * steps_per_cell),
        mean_cells=mean_cells,
        mean_steps=float(mean_cells * steps_per_cell)
        if np.isfinite(mean_cells)
        else np.nan,
        finite_pair_count=int(finite.sum()),
    )
