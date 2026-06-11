#!/usr/bin/env python
"""Offline action-ranking diagnostic for budgeted BMM Q critics."""

import argparse
import ast
import json
from pathlib import Path
import sys

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents import agents
from agents.bmm_trl import get_config
from envs.env_utils import make_env_and_datasets
from scripts.bmm_reachability_utils import binary_auc, binary_metrics, format_metric
from utils.datasets import Dataset, GCDataset
from utils.flax_utils import restore_agent
from utils.pointmaze_graph import median_step_xy, parse_xy_dims, valid_transition_indices
from utils.pointmaze_grid import (
    free_cell_distance_matrix,
    free_cell_to_state_indices,
    state_to_free_cell_indices,
    unwrap_maze_env,
)


def parse_tuple(value):
    parsed = ast.literal_eval(str(value))
    if isinstance(parsed, int):
        parsed = (parsed,)
    return tuple(int(x) for x in parsed)


def parse_int_list(value):
    value = str(value).strip()
    if not value:
        return tuple()
    if value.startswith("(") or value.startswith("["):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, int):
            parsed = (parsed,)
        return tuple(int(x) for x in parsed)
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "y"):
        return True
    if value in ("0", "false", "no", "n"):
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


def parse_critic_spec(value):
    if "=" not in value:
        raise ValueError(
            "Critic specs must have form NAME=RESTORE_PATH:EPOCH, "
            f"got {value!r}."
        )
    name, restore = value.split("=", 1)
    if ":" not in restore:
        raise ValueError(
            "Critic specs must have form NAME=RESTORE_PATH:EPOCH, "
            f"got {value!r}."
        )
    path, epoch = restore.rsplit(":", 1)
    return name, path, int(epoch)


def dataset_path_from_dir(dataset_dir):
    if dataset_dir is None:
        return None
    candidates = [
        str(path)
        for path in sorted(Path(dataset_dir).glob("*.npz"))
        if "-val.npz" not in path.name
    ]
    return candidates[0] if candidates else None


def make_grid_context(env, train_dataset, val_dataset, xy_dims, budget_unit):
    maze_env = unwrap_maze_env(env)
    median_step = median_step_xy((train_dataset, val_dataset), xy_dims)
    steps_per_cell = float(maze_env._maze_unit) / float(median_step)
    distance_scale = 1.0 if budget_unit == "grid_cells" else steps_per_cell
    free_cells, cell_distances = free_cell_distance_matrix(maze_env.maze_map)
    val_state_to_cell = state_to_free_cell_indices(
        val_dataset,
        maze_env.maze_map,
        free_cells,
        xy_dims=xy_dims,
        maze_unit=maze_env._maze_unit,
        offset_x=maze_env._offset_x,
        offset_y=maze_env._offset_y,
    )
    train_state_to_cell = state_to_free_cell_indices(
        train_dataset,
        maze_env.maze_map,
        free_cells,
        xy_dims=xy_dims,
        maze_unit=maze_env._maze_unit,
        offset_x=maze_env._offset_x,
        offset_y=maze_env._offset_y,
    )
    return dict(
        free_cells=free_cells,
        cell_distances=cell_distances,
        distance_scale=float(distance_scale),
        steps_per_cell=float(steps_per_cell),
        val_state_to_cell=val_state_to_cell,
        train_state_to_cell=train_state_to_cell,
        val_goal_by_cell=free_cell_to_state_indices(
            val_state_to_cell, len(free_cells)
        ),
        train_goal_by_cell=free_cell_to_state_indices(
            train_state_to_cell, len(free_cells)
        ),
        maze_type=maze_env._maze_type,
        maze_unit=float(maze_env._maze_unit),
        median_step_xy=float(median_step),
    )


def transition_indices_by_cell(dataset, state_to_cell):
    idxs = valid_transition_indices(dataset)
    idxs = idxs[idxs + 1 < len(state_to_cell)]
    idxs = idxs[(state_to_cell[idxs] >= 0) & (state_to_cell[idxs + 1] >= 0)]
    by_cell = [[] for _ in range(int(np.max(state_to_cell)) + 1)]
    for idx in idxs:
        by_cell[int(state_to_cell[idx])].append(int(idx))
    return [np.asarray(items, dtype=np.int32) for items in by_cell]


def sample_action_queries(
    dataset,
    context,
    budget,
    num_queries,
    candidate_count,
    rng,
    split="val",
    require_both_classes=True,
):
    """Sample same-cell candidate-action sets with grid oracle distances."""
    state_to_cell = np.asarray(context[f"{split}_state_to_cell"], dtype=np.int32)
    goal_by_cell = context[f"{split}_goal_by_cell"]
    by_cell = transition_indices_by_cell(dataset, state_to_cell)
    has_goal = np.asarray([len(items) > 0 for items in goal_by_cell])
    step_distances = np.asarray(context["cell_distances"], dtype=np.float32) * float(
        context["distance_scale"]
    )
    remaining_budget = max(float(int(budget) - 1), 1.0)
    eligible_cells = np.asarray(
        [idx for idx, items in enumerate(by_cell) if len(items) >= 2],
        dtype=np.int32,
    )
    if len(eligible_cells) == 0:
        raise ValueError("No validation cells have at least two candidate actions.")

    observations = []
    actions = []
    goals = []
    labels = []
    distances = []
    source_distances = []
    source_cells = []
    goal_cells = []
    candidate_source_idxs = []
    attempts = 0
    max_attempts = max(2000, int(num_queries) * 500)

    while len(observations) < int(num_queries) and attempts < max_attempts:
        attempts += 1
        source_cell = int(rng.choice(eligible_cells))
        cell_idxs = by_cell[source_cell]
        logged_idx = int(rng.choice(cell_idxs))
        other_idxs = cell_idxs[cell_idxs != logged_idx]
        sample_pool = other_idxs if len(other_idxs) >= int(candidate_count) - 1 else cell_idxs
        replace = len(sample_pool) < int(candidate_count) - 1
        sampled = rng.choice(
            sample_pool,
            size=max(int(candidate_count) - 1, 0),
            replace=replace,
        )
        candidate_idxs = np.concatenate(
            [np.asarray([logged_idx], dtype=np.int32), sampled.astype(np.int32)]
        )[: int(candidate_count)]
        if len(candidate_idxs) < int(candidate_count):
            continue

        next_cells = state_to_cell[candidate_idxs + 1]
        finite_from_all = np.all(
            np.asarray(context["cell_distances"])[next_cells] >= 0, axis=0
        )
        source_finite = np.asarray(context["cell_distances"])[source_cell] >= 0
        goal_mask = finite_from_all & source_finite & has_goal
        if not goal_mask.any():
            continue
        candidate_distances = step_distances[next_cells][:, goal_mask]
        candidate_goal_cells = np.nonzero(goal_mask)[0]
        candidate_labels = candidate_distances <= remaining_budget
        if require_both_classes:
            mixed = candidate_labels.any(axis=0) & (~candidate_labels).any(axis=0)
            if not mixed.any():
                continue
            candidate_distances = candidate_distances[:, mixed]
            candidate_goal_cells = candidate_goal_cells[mixed]
        spread = candidate_distances.max(axis=0) - candidate_distances.min(axis=0)
        useful = spread > 1e-6
        if not useful.any():
            continue
        candidate_distances = candidate_distances[:, useful]
        candidate_goal_cells = candidate_goal_cells[useful]
        spread = spread[useful]
        top_count = min(len(candidate_goal_cells), max(8, len(candidate_goal_cells) // 4))
        order = np.argsort(spread)[::-1][:top_count]
        goal_pos = int(rng.choice(order))
        goal_cell = int(candidate_goal_cells[goal_pos])
        goal_idx = int(rng.choice(goal_by_cell[goal_cell]))
        d_next = candidate_distances[:, goal_pos].astype(np.float32)

        observations.append(
            np.repeat(
                np.asarray(dataset["observations"])[logged_idx][None, :],
                int(candidate_count),
                axis=0,
            )
        )
        actions.append(np.asarray(dataset["actions"])[candidate_idxs])
        goals.append(
            np.repeat(
                np.asarray(dataset["observations"])[goal_idx][None, :],
                int(candidate_count),
                axis=0,
            )
        )
        labels.append((d_next <= remaining_budget).astype(np.float32))
        distances.append(d_next)
        source_distances.append(float(step_distances[source_cell, goal_cell]))
        source_cells.append(source_cell)
        goal_cells.append(goal_cell)
        candidate_source_idxs.append(candidate_idxs.astype(np.int32))

    if len(observations) < int(num_queries):
        raise ValueError(
            f"Could only sample {len(observations)} action-ranking queries "
            f"for H={budget} after {attempts} attempts."
        )
    return dict(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        goals=np.asarray(goals, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        distances=np.asarray(distances, dtype=np.float32),
        source_distances=np.asarray(source_distances, dtype=np.float32),
        source_cells=np.asarray(source_cells, dtype=np.int32),
        goal_cells=np.asarray(goal_cells, dtype=np.int32),
        candidate_source_idxs=np.asarray(candidate_source_idxs, dtype=np.int32),
        budget=int(budget),
        remaining_budget=float(remaining_budget),
    )


def pairwise_ranking_accuracy(scores, distances, eps=1e-6):
    scores = np.asarray(scores, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    correct = 0.0
    total = 0
    for query_scores, query_distances in zip(scores, distances):
        count = len(query_scores)
        for i in range(count):
            for j in range(i + 1, count):
                delta_d = query_distances[j] - query_distances[i]
                if abs(delta_d) <= eps:
                    continue
                delta_s = query_scores[i] - query_scores[j]
                if abs(delta_s) <= eps:
                    correct += 0.5
                elif (delta_d > 0.0 and delta_s > 0.0) or (
                    delta_d < 0.0 and delta_s < 0.0
                ):
                    correct += 1.0
                total += 1
    return float(correct / total) if total else float("nan"), int(total)


def selected_action_stats(scores, distances, labels, source_distances):
    scores = np.asarray(scores, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    source_distances = np.asarray(source_distances, dtype=np.float64)
    selected = np.argmax(scores, axis=1)
    rows = np.arange(scores.shape[0])
    selected_dist = distances[rows, selected]
    selected_label = labels[rows, selected]
    logged_dist = distances[:, 0]
    logged_label = labels[:, 0]
    oracle_dist = distances.min(axis=1)
    return dict(
        selected_distance_mean=float(selected_dist.mean()),
        selected_success_frac=float(selected_label.mean()),
        selected_improves_frac=float((selected_dist < source_distances).mean()),
        logged_distance_mean=float(logged_dist.mean()),
        logged_success_frac=float(logged_label.mean()),
        logged_improves_frac=float((logged_dist < source_distances).mean()),
        random_distance_mean=float(distances.mean()),
        random_success_frac=float(labels.mean()),
        random_improves_frac=float((distances < source_distances[:, None]).mean()),
        oracle_best_distance_mean=float(oracle_dist.mean()),
    )


def action_ranking_metrics(scores, labels, distances, source_distances):
    labels = np.asarray(labels, dtype=np.float32)
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    flat_labels = labels.reshape(-1)
    pair_acc, pair_count = pairwise_ranking_accuracy(scores, distances)
    metrics = binary_metrics(flat_scores, flat_labels)
    metrics.update(
        pairwise_accuracy=pair_acc,
        pairwise_count=pair_count,
        action_auc=float(binary_auc(flat_scores, flat_labels)),
    )
    metrics.update(selected_action_stats(scores, distances, labels, source_distances))
    return metrics


def score_flat(agent, observations, actions, goals, budgets, batch_size):
    mean_scores = []
    min_scores = []
    count = len(observations)
    for start in range(0, count, int(batch_size)):
        end = min(start + int(batch_size), count)
        logits = agent.critic_logits_for(
            observations[start:end],
            actions[start:end],
            goals[start:end],
            budgets[start:end],
            offsets=budgets[start:end],
        )
        scores = np.asarray(jax.nn.sigmoid(logits))
        mean_scores.append(scores.mean(axis=0))
        min_scores.append(scores.min(axis=0))
    return np.concatenate(mean_scores), np.concatenate(min_scores)


def score_queries(agent, queries, budget, interp_budgets, batch_size):
    shape = queries["labels"].shape
    observations = queries["observations"].reshape((-1, queries["observations"].shape[-1]))
    actions = queries["actions"].reshape((-1, queries["actions"].shape[-1]))
    goals = queries["goals"].reshape((-1, queries["goals"].shape[-1]))
    budgets = np.full(len(observations), int(budget), dtype=np.int32)
    parent_mean, parent_min = score_flat(
        agent, observations, actions, goals, budgets, batch_size
    )
    result = {
        "parent_mean": parent_mean.reshape(shape),
        "parent_ensemble_min": parent_min.reshape(shape),
    }
    if interp_budgets:
        interp_mean_rows = []
        interp_min_rows = []
        for interp_budget in interp_budgets:
            interp_arr = np.full(len(observations), int(interp_budget), dtype=np.int32)
            mean_scores, min_scores = score_flat(
                agent, observations, actions, goals, interp_arr, batch_size
            )
            interp_mean_rows.append(mean_scores.reshape(shape))
            interp_min_rows.append(min_scores.reshape(shape))
        result["interp_mean"] = np.max(np.stack(interp_mean_rows, axis=0), axis=0)
        result["interp_ensemble_min"] = np.max(
            np.stack(interp_min_rows, axis=0), axis=0
        )
    return result


def configure_restore_agent(args, train_dataset, budgets):
    config = get_config()
    config.budgets = tuple(int(x) for x in budgets)
    config.max_budget = max(int(x) for x in budgets)
    config.batch_size = 1
    config.diagnostic_critic_mode = "action"
    config.value_only = True
    config.lambda_sup = 1.0
    config.lambda_rank = 0.0
    config.lambda_mono = 0.0
    config.lambda_trans = 0.0
    config.lambda_pos = 0.0
    config.lambda_budget_neg = 0.0
    config.lambda_hard_neg = 0.0
    config.lambda_rand_hinge = 0.0
    config.num_sup_pairs = 0
    config.num_rank_pairs = 0
    config.dataset.reachability_label_type = "grid_geodesic"
    config.actor_hidden_dims = parse_tuple(args.actor_hidden_dims)
    config.value_hidden_dims = parse_tuple(args.value_hidden_dims)
    config.layer_norm = parse_bool(args.layer_norm)
    gc_train = GCDataset(Dataset.create(**train_dataset), config)
    example_batch = gc_train.sample(1)
    return agents[config["agent_name"]].create(args.seed, example_batch, config)


def summarize_markdown(result):
    lines = [
        "# BMM action-ranking diagnostic",
        "",
        f"env: `{result['env_name']}`",
        f"budget: `{result['budget']}`",
        f"queries: `{result['num_queries']}`, candidates/query: `{result['candidate_count']}`",
        "",
        "| critic | score | pair_acc | AUC | gap | selected_d | logged_d | random_d | selected_improve | selected_success |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for critic in result["critics"]:
        for score_name, metrics in critic["scores"].items():
            gap = metrics["pos_mean"] - metrics["neg_mean"]
            lines.append(
                "| {critic} | {score} | {pair_acc} | {auc} | {gap} | "
                "{selected_d} | {logged_d} | {random_d} | {improve} | {success} |".format(
                    critic=critic["name"],
                    score=score_name,
                    pair_acc=format_metric(metrics["pairwise_accuracy"]),
                    auc=format_metric(metrics["action_auc"]),
                    gap=format_metric(gap),
                    selected_d=format_metric(metrics["selected_distance_mean"]),
                    logged_d=format_metric(metrics["logged_distance_mean"]),
                    random_d=format_metric(metrics["random_distance_mean"]),
                    improve=format_metric(metrics["selected_improves_frac"]),
                    success=format_metric(metrics["selected_success_frac"]),
                )
            )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", default="pointmaze-medium-navigate-v0")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--geodesic_budget_unit", default="env_steps")
    parser.add_argument("--xy_dims", default="0,1")
    parser.add_argument("--budgets", default="40,80,160")
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--interp_budgets", default="40,80")
    parser.add_argument("--critics", nargs="+", required=True)
    parser.add_argument("--num_queries", type=int, default=512)
    parser.add_argument("--candidate_count", type=int, default=8)
    parser.add_argument("--score_batch_size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor_hidden_dims", default="(256, 256)")
    parser.add_argument("--value_hidden_dims", default="(256, 256)")
    parser.add_argument("--layer_norm", default="False")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    budgets = parse_int_list(args.budgets)
    interp_budgets = parse_int_list(args.interp_budgets)
    rng = np.random.default_rng(args.seed)
    dataset_path = dataset_path_from_dir(args.dataset_dir)
    env, train_dataset, val_dataset = make_env_and_datasets(
        args.env_name, dataset_path=dataset_path
    )
    context = make_grid_context(
        env,
        train_dataset,
        val_dataset,
        parse_xy_dims(args.xy_dims),
        args.geodesic_budget_unit,
    )
    queries = sample_action_queries(
        val_dataset,
        context,
        args.budget,
        args.num_queries,
        args.candidate_count,
        rng,
    )
    result = dict(
        env_name=args.env_name,
        geodesic_budget_unit=args.geodesic_budget_unit,
        budget=int(args.budget),
        interp_budgets=[int(x) for x in interp_budgets],
        budgets=[int(x) for x in budgets],
        num_queries=int(args.num_queries),
        candidate_count=int(args.candidate_count),
        remaining_budget=float(queries["remaining_budget"]),
        label_positive_frac=float(queries["labels"].mean()),
        source_cell_count=int(len(np.unique(queries["source_cells"]))),
        goal_cell_count=int(len(np.unique(queries["goal_cells"]))),
        context=dict(
            maze_type=context["maze_type"],
            maze_unit=context["maze_unit"],
            median_step_xy=context["median_step_xy"],
            steps_per_cell=context["steps_per_cell"],
            distance_scale=context["distance_scale"],
        ),
        critics=[],
    )

    for spec in args.critics:
        name, restore_path, restore_epoch = parse_critic_spec(spec)
        agent = configure_restore_agent(args, train_dataset, budgets)
        agent = restore_agent(agent, restore_path, restore_epoch)
        scores = score_queries(
            agent,
            queries,
            args.budget,
            interp_budgets,
            args.score_batch_size,
        )
        score_metrics = {
            key: action_ranking_metrics(
                value,
                queries["labels"],
                queries["distances"],
                queries["source_distances"],
            )
            for key, value in scores.items()
        }
        result["critics"].append(
            dict(
                name=name,
                restore_path=restore_path,
                restore_epoch=int(restore_epoch),
                scores=score_metrics,
            )
        )

    text = summarize_markdown(result)
    print(text)
    if args.output_json is not None:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
    if args.output_markdown is not None:
        path = Path(args.output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


if __name__ == "__main__":
    main()
