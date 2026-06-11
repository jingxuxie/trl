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
    graph_step_distance_matrix,
    graph_step_distances,
    sample_graph_budget_pairs,
    shortest_hop_distances,
)
from scripts import train_bmm_geodesic_q as qdiag


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
    distance_matrix = graph_step_distance_matrix(adjacency, graph)
    assert distances[graph["train_state_to_bin"][0]] == 0
    assert distances[graph["train_state_to_bin"][4]] == 4
    np.testing.assert_allclose(distance_matrix[graph["train_state_to_bin"][0]], distances)

    pair_batch = sample_graph_budget_pairs(
        train_dataset,
        graph["train_state_to_bin"],
        graph,
        budget=3,
        num_pairs=64,
        rng=rng,
        adjacency=adjacency,
        distance_matrix=distance_matrix,
    )
    assert pair_batch is not None
    assert pair_batch["labels"].sum() > 0
    assert (pair_batch["labels"] == 0).sum() > 0
    assert np.all(pair_batch["graph_distances"][pair_batch["labels"] == 1] <= 3)
    assert np.all(pair_batch["graph_distances"][pair_batch["labels"] == 0] > 3)
    assert binary_auc(-pair_batch["graph_distances"], pair_batch["labels"]) == 1.0

    if not qdiag.FLAGS.is_parsed():
        qdiag.FLAGS(["test_bmm_pointmaze_graph"])
    qdiag.FLAGS.num_trans_witnesses = 2
    qdiag.FLAGS.trans_witness_mode = "slack_balanced"
    qdiag.FLAGS.trans_pos_boundary_frac = 0.5
    qdiag.FLAGS.trans_endpoint_epsilon = 1e-6
    qdiag.FLAGS.trans_boundary_beta = 0.25
    qv_batch = qdiag.sample_graph_qv_transitive_pairs(
        train_dataset,
        dict(
            kind="graph",
            graph=graph,
            adjacency=adjacency,
            distance_matrix=distance_matrix,
        ),
        "train",
        budgets=(4,),
        batch_size=8,
        rng=rng,
    )
    assert qv_batch["qv_observations"].shape == (8, 2)
    assert qv_batch["qv_midpoint_observations"].shape == (2, 8, 2)
    assert qv_batch["qv_valids"].shape == (2, 8)
    assert np.all(qv_batch["qv_valids"] == 1.0)
    assert np.all(qv_batch["qv_parent_distances"] <= 3.0)
    assert np.all(qv_batch["qv_left_distances"] <= 1.0)
    assert np.all(qv_batch["qv_right_distances"] <= 2.0)
    assert np.all(qv_batch["qv_parent_oracle_labels"] == 1.0)
    assert np.all(qv_batch["qv_left_oracle_labels"] == 1.0)
    assert np.all(qv_batch["qv_right_oracle_labels"] == 1.0)

    print("PointMaze graph utility checks passed.")


if __name__ == "__main__":
    main()
