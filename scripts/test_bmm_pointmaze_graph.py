#!/usr/bin/env python
"""Synthetic checks for PointMaze dataset-position graph utilities."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bmm_reachability_utils import binary_auc
from utils.datasets import Dataset
from utils.pointmaze_graph import (
    adjacency_lists,
    build_dataset_position_graph,
    graph_step_distances,
    sample_graph_budget_pairs,
    shortest_hop_distances,
)


def make_line_dataset(length=9):
    observations = np.stack(
        [
            np.arange(length, dtype=np.float32),
            np.zeros(length, dtype=np.float32),
        ],
        axis=-1,
    )
    actions = np.ones((length, 2), dtype=np.float32)
    terminals = np.zeros(length, dtype=np.float32)
    terminals[-1] = 1.0
    valids = np.ones(length, dtype=np.float32)
    valids[-1] = 0.0
    return Dataset.create(
        observations=observations,
        actions=actions,
        terminals=terminals,
        valids=valids,
    )


def main():
    rng = np.random.default_rng(0)
    train_dataset = make_line_dataset()
    val_dataset = make_line_dataset()
    graph = build_dataset_position_graph(
        train_dataset,
        val_dataset,
        xy_dims=(0, 1),
        bin_size=1.0,
    )
    assert len(graph["bin_centers"]) == 9
    assert len(graph["edge_src"]) == 8
    assert np.isclose(graph["metadata"]["env_steps_per_graph_edge"], 1.0)

    adjacency = adjacency_lists(
        len(graph["bin_centers"]), graph["edge_src"], graph["edge_dst"]
    )
    hops = shortest_hop_distances(adjacency, graph["train_state_to_bin"][0])
    distances = graph_step_distances(hops, graph)
    assert distances[graph["train_state_to_bin"][0]] == 0
    assert distances[graph["train_state_to_bin"][4]] == 4

    pair_batch = sample_graph_budget_pairs(
        train_dataset,
        graph["train_state_to_bin"],
        graph,
        budget=3,
        num_pairs=64,
        rng=rng,
        adjacency=adjacency,
    )
    assert pair_batch is not None
    assert pair_batch["labels"].sum() > 0
    assert (pair_batch["labels"] == 0).sum() > 0
    assert np.all(pair_batch["graph_distances"][pair_batch["labels"] == 1] <= 3)
    assert np.all(pair_batch["graph_distances"][pair_batch["labels"] == 0] > 3)
    assert binary_auc(-pair_batch["graph_distances"], pair_batch["labels"]) == 1.0

    print("PointMaze graph utility checks passed.")


if __name__ == "__main__":
    main()
