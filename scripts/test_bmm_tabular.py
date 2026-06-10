#!/usr/bin/env python
"""Tabular sanity checks for the BMM max-min reachability backup."""

import os
from collections import deque
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def shortest_paths(num_states, edges):
    neighbors = [[] for _ in range(num_states)]
    for src, dst in edges:
        neighbors[src].append(dst)

    dist = np.full((num_states, num_states), np.inf, dtype=np.float32)
    for start in range(num_states):
        dist[start, start] = 0
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for next_state in neighbors[state]:
                if np.isinf(dist[start, next_state]):
                    dist[start, next_state] = dist[start, state] + 1
                    queue.append(next_state)
    return dist


def compose_reachability(left, right):
    return np.max(np.minimum(left[:, :, None], right[None, :, :]), axis=1)


def max_min_reachability(dist, budgets):
    reach = {1: (dist <= 1).astype(np.float32)}
    for horizon in budgets[1:]:
        left_horizon = horizon // 2
        right_horizon = horizon - left_horizon
        reach[horizon] = compose_reachability(
            reach[left_horizon], reach[right_horizon]
        )
    return reach


def chain_edges(num_states):
    return [(idx, idx + 1) for idx in range(num_states - 1)]


def grid_edges(width, height):
    edges = []
    for y in range(height):
        for x in range(width):
            state = y * width + x
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    edges.append((state, ny * width + nx))
    return edges


def assert_exact_backup(name, num_states, edges, budgets):
    dist = shortest_paths(num_states, edges)
    reach = max_min_reachability(dist, budgets)
    print(f"\n{name}")
    print("H | positives | exact_match")
    print("--|-----------|------------")
    for horizon in budgets:
        exact = (dist <= horizon).astype(np.float32)
        matches = np.array_equal(reach[horizon], exact)
        print(f"{horizon:2d} | {int(exact.sum()):9d} | {matches}")
        assert matches, f"{name}: max-min backup mismatch at H={horizon}"


def residual_error_table(max_horizon=1024, epsilon=0.02):
    horizons = [1]
    while horizons[-1] < max_horizon:
        horizons.append(horizons[-1] * 2)

    balanced = {1: epsilon}
    for horizon in horizons[1:]:
        balanced[horizon] = epsilon + max(
            balanced[horizon // 2], balanced[horizon - horizon // 2]
        )

    unbalanced = {1: epsilon}
    for horizon in range(2, max_horizon + 1):
        unbalanced[horizon] = epsilon + max(unbalanced[horizon - 1], unbalanced[1])

    additive = {horizon: epsilon * horizon for horizon in horizons}

    print("\nInjected per-level residual error")
    print("H    | balanced max-min | unbalanced max-min | additive-distance")
    print("-----|------------------|--------------------|------------------")
    for horizon in horizons:
        print(
            f"{horizon:4d} | {balanced[horizon]:16.4f} | "
            f"{unbalanced[horizon]:18.4f} | {additive[horizon]:16.4f}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_path = Path(__file__).with_name("bmm_tabular_errors.png")
    plt.figure(figsize=(6, 4))
    plt.plot(horizons, [balanced[h] for h in horizons], marker="o", label="balanced")
    plt.plot(
        horizons,
        [unbalanced[h] for h in horizons],
        marker="o",
        label="unbalanced",
    )
    plt.plot(horizons, [additive[h] for h in horizons], marker="o", label="additive")
    plt.xscale("log", base=2)
    plt.xlabel("Budget H")
    plt.ylabel("Sup-norm error bound")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nSaved optional plot to {plot_path}")


def main():
    budgets = (1, 2, 4, 8)
    assert_exact_backup("Directed chain", 12, chain_edges(12), budgets)
    assert_exact_backup("Undirected 4x4 grid", 16, grid_edges(4, 4), budgets)
    residual_error_table()


if __name__ == "__main__":
    main()
