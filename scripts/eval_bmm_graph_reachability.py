#!/usr/bin/env python
"""Evaluate PointMaze graph-distance reachability labels."""

import json
from pathlib import Path
import random
import sys

import numpy as np
from absl import app, flags

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets
from scripts.bmm_reachability_utils import binary_auc, format_metric, rank_metrics
from utils.pointmaze_graph import (
    adjacency_lists,
    build_dataset_position_graph,
    graph_distance_statistics,
    load_graph_npz,
    parse_xy_dims,
    sample_graph_budget_pairs,
    save_graph_npz,
)


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "pointmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_string("dataset_dir", None, "Optional OGBench dataset directory.")
flags.DEFINE_string("graph_path", "exp/bmm_pointmaze_graph.npz", "Graph npz path.")
flags.DEFINE_bool("rebuild_graph", False, "Rebuild the graph even if it exists.")
flags.DEFINE_string("xy_dims", "0,1", "Comma-separated observation dims to use as xy.")
flags.DEFINE_float("bin_size", None, "Optional graph bin size.")
flags.DEFINE_float("bin_size_factor", 2.0, "Bin size multiplier.")
flags.DEFINE_string("budgets", "64,128,256,512", "Comma-separated budgets.")
flags.DEFINE_integer("num_train_pairs", 8192, "Training pairs per budget for kNN.")
flags.DEFINE_integer("num_eval_pairs", 2048, "Evaluation pairs per budget.")
flags.DEFINE_integer("k", 32, "Number of nearest neighbors.")
flags.DEFINE_string(
    "features",
    "xy_pair,xy_delta,full_pair,full_pair_action",
    "Comma-separated feature modes for kNN.",
)
flags.DEFINE_float(
    "pos_boundary_frac",
    0.5,
    "Positive graph-distance lower bound as a fraction of budget.",
)
flags.DEFINE_float(
    "neg_max_factor",
    2.0,
    "Negative graph-distance upper bound as a factor of budget.",
)
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("eval_chunk_size", 256, "Eval chunk size for kNN.")
flags.DEFINE_string("output_json", None, "Optional JSON output path.")


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def dataset_path_from_dir(dataset_dir):
    if dataset_dir is None:
        return None
    candidates = [
        str(path)
        for path in sorted(Path(dataset_dir).glob("*.npz"))
        if "-val.npz" not in path.name
    ]
    return candidates[0] if candidates else None


def get_or_build_graph(train_dataset, val_dataset, xy_dims):
    graph_path = Path(FLAGS.graph_path)
    if graph_path.exists() and not FLAGS.rebuild_graph:
        return load_graph_npz(graph_path)
    graph = build_dataset_position_graph(
        train_dataset,
        val_dataset,
        xy_dims=xy_dims,
        bin_size=FLAGS.bin_size,
        bin_size_factor=FLAGS.bin_size_factor,
    )
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph_npz(graph_path, graph)
    return graph


def features_for(pair_batch, mode):
    observations = np.asarray(pair_batch["observations"], dtype=np.float32)
    goals = np.asarray(pair_batch["goals"], dtype=np.float32)
    actions = np.asarray(pair_batch["actions"], dtype=np.float32)

    if mode == "xy_pair":
        return np.concatenate([observations[:, :2], goals[:, :2]], axis=-1)
    if mode == "xy_delta":
        return np.concatenate(
            [observations[:, :2], goals[:, :2], goals[:, :2] - observations[:, :2]],
            axis=-1,
        )
    if mode == "full_pair":
        return np.concatenate([observations, goals], axis=-1)
    if mode == "full_pair_action":
        return np.concatenate([observations, actions, goals], axis=-1)
    raise ValueError(f"Unsupported feature mode: {mode}")


def standardize(train_features, eval_features):
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((train_features - mean) / std).astype(np.float32),
        ((eval_features - mean) / std).astype(np.float32),
    )


def knn_label_probs(train_features, train_labels, eval_features, k, eval_chunk_size):
    train_features = np.asarray(train_features, dtype=np.float32)
    eval_features = np.asarray(eval_features, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.float32)
    k = min(int(k), len(train_features))
    train_norms = np.sum(train_features * train_features, axis=1)
    probs = np.empty(len(eval_features), dtype=np.float32)

    for start in range(0, len(eval_features), int(eval_chunk_size)):
        end = min(start + int(eval_chunk_size), len(eval_features))
        chunk = eval_features[start:end]
        dists = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + train_norms[None, :]
            - 2.0 * chunk @ train_features.T
        )
        neighbor_idxs = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        probs[start:end] = train_labels[neighbor_idxs].mean(axis=1)
    return probs


def probability_metrics(probs, labels):
    labels = np.asarray(labels, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)
    pos_mask = labels == 1.0
    neg_mask = ~pos_mask
    pos_mean = float(probs[pos_mask].mean()) if pos_mask.any() else np.nan
    neg_mean = float(probs[neg_mask].mean()) if neg_mask.any() else np.nan
    return dict(
        auc=float(binary_auc(probs, labels)),
        gap=float(pos_mean - neg_mean)
        if np.isfinite(pos_mean) and np.isfinite(neg_mean)
        else np.nan,
        pos_mean=pos_mean,
        neg_mean=neg_mean,
        pos_count=int(pos_mask.sum()),
        neg_count=int(neg_mask.sum()),
    )


def baseline_rows(pair_batch):
    observations = np.asarray(pair_batch["observations"], dtype=np.float32)
    goals = np.asarray(pair_batch["goals"], dtype=np.float32)
    labels = np.asarray(pair_batch["labels"], dtype=np.float32)
    graph_distances = np.asarray(pair_batch["graph_distances"], dtype=np.float32)
    euclidean = np.linalg.norm(goals[:, :2] - observations[:, :2], axis=-1)
    return dict(
        graph_distance_oracle=rank_metrics(-graph_distances, labels),
        euclidean=rank_metrics(-euclidean, labels),
    )


def print_budget_report(budget, row):
    print(f"\nH={budget}")
    print(
        "metric                  | auc    | gap    | pos_mean | neg_mean | pos_n | neg_n"
    )
    print(
        "------------------------|--------|--------|----------|----------|-------|------"
    )
    for name, metrics in row["baselines"].items():
        print_metric_row(name, metrics)
    for feature, metrics in row["knn"].items():
        print_metric_row(f"knn_{feature}", metrics)


def print_metric_row(name, metrics):
    print(
        f"{name:24s} | {format_metric(metrics.get('auc')):6s} | "
        f"{format_metric(metrics.get('gap')):6s} | "
        f"{format_metric(metrics.get('pos_mean')):8s} | "
        f"{format_metric(metrics.get('neg_mean')):8s} | "
        f"{int(metrics.get('pos_count', 0)):5d} | "
        f"{int(metrics.get('neg_count', 0)):5d}"
    )


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    rng = np.random.default_rng(FLAGS.seed)
    xy_dims = parse_xy_dims(FLAGS.xy_dims)
    budgets = parse_int_list(FLAGS.budgets)
    feature_modes = parse_str_list(FLAGS.features)
    dataset_path = dataset_path_from_dir(FLAGS.dataset_dir)
    _, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, dataset_path=dataset_path
    )
    graph = get_or_build_graph(train_dataset, val_dataset, xy_dims)
    adjacency = adjacency_lists(
        len(graph["bin_centers"]), graph["edge_src"], graph["edge_dst"]
    )
    distance_stats = graph_distance_statistics(adjacency, graph)

    print("PointMaze graph reachability diagnostic")
    print(f"  env_name: {FLAGS.env_name}")
    print(f"  graph_path: {FLAGS.graph_path}")
    print(f"  nodes: {len(graph['bin_centers'])}")
    print(f"  edges: {len(graph['edge_src'])}")
    print(f"  metadata: {graph['metadata']}")
    print(f"  graph_diameter_steps: {distance_stats['max_steps']:.2f}")

    report = dict(
        env_name=FLAGS.env_name,
        graph_path=FLAGS.graph_path,
        graph_metadata=graph["metadata"],
        graph_distance_stats=distance_stats,
        budgets=budgets,
        num_train_pairs=int(FLAGS.num_train_pairs),
        num_eval_pairs=int(FLAGS.num_eval_pairs),
        k=int(FLAGS.k),
        features=feature_modes,
        rows=[],
    )

    for budget in budgets:
        train_pairs = sample_graph_budget_pairs(
            train_dataset,
            graph["train_state_to_bin"],
            graph,
            budget,
            FLAGS.num_train_pairs,
            rng,
            pos_boundary_frac=FLAGS.pos_boundary_frac,
            neg_max_factor=FLAGS.neg_max_factor,
            adjacency=adjacency,
        )
        eval_pairs = sample_graph_budget_pairs(
            val_dataset,
            graph["val_state_to_bin"],
            graph,
            budget,
            FLAGS.num_eval_pairs,
            rng,
            pos_boundary_frac=FLAGS.pos_boundary_frac,
            neg_max_factor=FLAGS.neg_max_factor,
            adjacency=adjacency,
        )
        if train_pairs is None or eval_pairs is None:
            row = dict(
                budget=int(budget),
                skipped=True,
                reason="no graph-distance pairs available",
            )
            report["rows"].append(row)
            print(f"\nH={budget}: skipped, no graph-distance pairs available")
            continue

        eval_labels = np.asarray(eval_pairs["labels"], dtype=np.float32)
        row = dict(
            budget=int(budget),
            skipped=False,
            train_pos_count=int((train_pairs["labels"] == 1.0).sum()),
            train_neg_count=int((train_pairs["labels"] == 0.0).sum()),
            eval_pos_count=int((eval_labels == 1.0).sum()),
            eval_neg_count=int((eval_labels == 0.0).sum()),
            baselines=baseline_rows(eval_pairs),
            knn={},
        )

        for mode in feature_modes:
            train_features = features_for(train_pairs, mode)
            eval_features = features_for(eval_pairs, mode)
            train_features, eval_features = standardize(train_features, eval_features)
            probs = knn_label_probs(
                train_features,
                train_pairs["labels"],
                eval_features,
                FLAGS.k,
                FLAGS.eval_chunk_size,
            )
            row["knn"][mode] = probability_metrics(probs, eval_labels)

        report["rows"].append(row)
        print_budget_report(budget, row)

    if FLAGS.output_json is not None:
        output_path = Path(FLAGS.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote graph reachability report to {output_path}")


if __name__ == "__main__":
    app.run(main)
