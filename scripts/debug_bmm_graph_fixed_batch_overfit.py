#!/usr/bin/env python
"""Overfit BMM supervised critic fields on PointMaze graph-distance labels."""

import ast
import json
from pathlib import Path
import random
import sys

import jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents import agents
from envs.env_utils import make_env_and_datasets
from scripts.bmm_reachability_utils import binary_metrics, format_metric
from utils.datasets import Dataset, GCDataset
from utils.pointmaze_graph import (
    adjacency_lists,
    build_dataset_position_graph,
    load_graph_npz,
    parse_xy_dims,
    sample_graph_budget_pairs,
    save_graph_npz,
)


FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "pointmaze-medium-navigate-v0", "Environment name.")
flags.DEFINE_string("dataset_dir", None, "Optional directory of OGBench npz files.")
flags.DEFINE_string("graph_path", "exp/bmm_pointmaze_graph.npz", "Graph npz path.")
flags.DEFINE_bool("rebuild_graph", False, "Rebuild the graph even if it exists.")
flags.DEFINE_string("xy_dims", "0,1", "Comma-separated observation dims to use as xy.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("budgets", "(32, 64, 96, 128)", "Budgets to train/evaluate.")
flags.DEFINE_integer("batch_size", 256, "Frozen transition batch size.")
flags.DEFINE_integer("steps", 1000, "Maximum fixed-batch update steps.")
flags.DEFINE_integer("eval_interval", 100, "Evaluate every N update steps.")
flags.DEFINE_string(
    "diagnostic_critic_mode", "state", "Diagnostic critic mode: state or action."
)
flags.DEFINE_float("target_auc", 0.98, "Early-stop train AUC target.")
flags.DEFINE_float("target_gap", 0.5, "Early-stop train score-gap target.")
flags.DEFINE_string("output_json", None, "Optional path to write final metrics.")

config_flags.DEFINE_config_file("agent", "agents/bmm_trl.py", lock_config=False)


def parse_budgets(value):
    parsed = ast.literal_eval(value)
    if isinstance(parsed, int):
        parsed = (parsed,)
    budgets = tuple(int(x) for x in parsed)
    if not budgets:
        raise ValueError("--budgets must contain at least one budget.")
    return budgets


def dataset_path_from_dir(dataset_dir):
    if dataset_dir is None:
        return None
    candidates = [
        str(path)
        for path in sorted(Path(dataset_dir).glob("*.npz"))
        if "-val.npz" not in path.name
    ]
    return candidates[0] if candidates else None


def configure_agent(config):
    budgets = parse_budgets(FLAGS.budgets)
    config.budgets = budgets
    config.max_budget = max(budgets)
    config.batch_size = FLAGS.batch_size
    config.diagnostic_critic_mode = FLAGS.diagnostic_critic_mode
    config.value_only = True
    config.lambda_sup = 1.0
    config.lambda_rank = 0.0
    config.lambda_mono = 0.0
    config.lambda_trans = 0.0
    config.lambda_pos = 0.0
    config.lambda_budget_neg = 0.0
    config.lambda_hard_neg = 0.0
    config.lambda_rand_hinge = 0.0
    config.num_rank_pairs = 0
    config.actor_hidden_dims = tuple(config.actor_hidden_dims)
    config.value_hidden_dims = tuple(config.value_hidden_dims)
    return budgets


def get_or_build_graph(train_dataset, val_dataset):
    graph_path = Path(FLAGS.graph_path)
    if graph_path.exists() and not FLAGS.rebuild_graph:
        return load_graph_npz(graph_path)
    graph = build_dataset_position_graph(
        train_dataset,
        val_dataset,
        xy_dims=parse_xy_dims(FLAGS.xy_dims),
    )
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph_npz(graph_path, graph)
    return graph


def make_graph_sup_fields(dataset, state_to_bin, graph, budgets, batch_size, rng, adjacency):
    rows = []
    for budget in budgets:
        pair_batch = sample_graph_budget_pairs(
            dataset,
            state_to_bin,
            graph,
            int(budget),
            int(batch_size),
            rng,
            adjacency=adjacency,
        )
        if pair_batch is None or len(pair_batch["labels"]) < batch_size:
            raise ValueError(
                f"Could not sample {batch_size} graph pairs for H={budget}; "
                f"got {0 if pair_batch is None else len(pair_batch['labels'])}."
            )
        rows.append(pair_batch)

    return dict(
        value_sup_observations=np.stack([row["observations"] for row in rows], axis=0),
        value_sup_actions=np.stack([row["actions"] for row in rows], axis=0),
        value_sup_goals=np.stack([row["goals"] for row in rows], axis=0),
        value_sup_budgets=np.stack([row["budgets"] for row in rows], axis=0),
        value_sup_offsets=np.stack(
            [np.rint(row["graph_distances"]).astype(np.int32) for row in rows],
            axis=0,
        ),
        value_sup_labels=np.stack([row["labels"] for row in rows], axis=0),
        value_sup_valids=np.ones((len(rows), batch_size), dtype=np.float32),
        value_sup_graph_distances=np.stack(
            [row["graph_distances"] for row in rows], axis=0
        ),
    )


def score_sup_batch(agent, batch, budgets):
    logits = agent.critic_logits_for_pair_grid(
        batch["value_sup_observations"],
        batch["value_sup_actions"],
        batch["value_sup_goals"],
        batch["value_sup_budgets"],
        offsets=batch["value_sup_offsets"],
    )
    scores = np.asarray(jax.nn.sigmoid(logits))
    mean_scores = scores.mean(axis=0)
    min_scores = scores.min(axis=0)
    labels = np.asarray(batch["value_sup_labels"])
    valids = np.asarray(batch["value_sup_valids"]) > 0
    sup_budgets = np.asarray(batch["value_sup_budgets"])

    report = {
        "mean": binary_metrics(mean_scores[valids], labels[valids]),
        "ensemble_min": binary_metrics(min_scores[valids], labels[valids]),
        "budget_rows": [],
    }
    for budget in budgets:
        mask = valids & (sup_budgets == int(budget))
        if not mask.any():
            continue
        report["budget_rows"].append(
            {
                "budget": int(budget),
                "mean": binary_metrics(mean_scores[mask], labels[mask]),
                "ensemble_min": binary_metrics(min_scores[mask], labels[mask]),
            }
        )
    return report


def passed(report):
    for row in report["budget_rows"]:
        for key in ("mean", "ensemble_min"):
            metrics = row[key]
            if metrics["pos_count"] == 0 or metrics["neg_count"] == 0:
                return False
            gap = metrics["pos_mean"] - metrics["neg_mean"]
            if metrics["auc"] < FLAGS.target_auc or gap < FLAGS.target_gap:
                return False
    return bool(report["budget_rows"])


def print_report(title, step, report, loss=None):
    loss_text = "" if loss is None else f" loss={format_metric(loss)}"
    print(f"\n{title} step={step}{loss_text}")
    print("H | auc | gap | pos | neg | min_auc | min_gap | pos_n | neg_n")
    print("--|-----|-----|-----|-----|---------|---------|-------|------")
    for row in report["budget_rows"]:
        mean = row["mean"]
        ens_min = row["ensemble_min"]
        gap = mean["pos_mean"] - mean["neg_mean"]
        min_gap = ens_min["pos_mean"] - ens_min["neg_mean"]
        print(
            f"{row['budget']:4d} | {format_metric(mean['auc'])} | "
            f"{format_metric(gap)} | {format_metric(mean['pos_mean'])} | "
            f"{format_metric(mean['neg_mean'])} | {format_metric(ens_min['auc'])} | "
            f"{format_metric(min_gap)} | {mean['pos_count']:5d} | {mean['neg_count']:5d}"
        )


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    rng = np.random.default_rng(FLAGS.seed)
    config = FLAGS.agent
    if config["agent_name"] != "bmm_trl":
        raise ValueError("debug_bmm_graph_fixed_batch_overfit.py requires bmm_trl.")
    budgets = configure_agent(config)

    dataset_path = dataset_path_from_dir(FLAGS.dataset_dir)
    _, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, dataset_path=dataset_path
    )
    graph = get_or_build_graph(train_dataset, val_dataset)
    adjacency = adjacency_lists(
        len(graph["bin_centers"]), graph["edge_src"], graph["edge_dst"]
    )
    dataset_class = {"GCDataset": GCDataset}[config["dataset"]["dataset_class"]]
    gc_train = dataset_class(Dataset.create(**train_dataset), config)

    train_batch = gc_train.sample(config.batch_size)
    train_batch.update(
        make_graph_sup_fields(
            train_dataset,
            graph["train_state_to_bin"],
            graph,
            budgets,
            config.batch_size,
            rng,
            adjacency,
        )
    )
    eval_batch = make_graph_sup_fields(
        val_dataset,
        graph["val_state_to_bin"],
        graph,
        budgets,
        config.batch_size,
        rng,
        adjacency,
    )

    example_batch = gc_train.sample(1)
    agent = agents[config["agent_name"]].create(FLAGS.seed, example_batch, config)
    train_report = score_sup_batch(agent, train_batch, budgets)
    eval_report = score_sup_batch(agent, eval_batch, budgets)
    print_report("train", 0, train_report)
    print_report("eval", 0, eval_report)

    final_loss = None
    for step in range(1, FLAGS.steps + 1):
        agent, info = agent.update(train_batch)
        final_loss = float(info["critic/loss_sup"])
        if step % FLAGS.eval_interval == 0 or step == FLAGS.steps:
            train_report = score_sup_batch(agent, train_batch, budgets)
            eval_report = score_sup_batch(agent, eval_batch, budgets)
            print_report("train", step, train_report, loss=final_loss)
            print_report("eval", step, eval_report)
            if passed(train_report):
                print(f"\nPassed graph fixed-batch train target at step {step}.")
                break

    final_report = dict(
        train=train_report,
        eval=eval_report,
        config=dict(
            env_name=FLAGS.env_name,
            graph_path=FLAGS.graph_path,
            graph_metadata=graph["metadata"],
            budgets=[int(x) for x in budgets],
            diagnostic_critic_mode=FLAGS.diagnostic_critic_mode,
            steps=int(step),
            target_auc=float(FLAGS.target_auc),
            target_gap=float(FLAGS.target_gap),
            final_loss_sup=final_loss,
        ),
    )
    if FLAGS.output_json is not None:
        output_path = Path(FLAGS.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(final_report, f, indent=2)
        print(f"\nWrote graph fixed-batch report to {output_path}")

    if not passed(train_report):
        raise SystemExit("Graph fixed-batch overfit target was not reached.")


if __name__ == "__main__":
    app.run(main)
