#!/usr/bin/env python
"""Graph-support value-subgoal diagnostic for a state-only BMM critic."""

import argparse
import json
from pathlib import Path
import sys

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.env_utils import make_env_and_datasets
from scripts import eval_bmm_action_ranking as ar
from scripts.bmm_reachability_utils import format_metric
from utils.flax_utils import restore_agent
from utils.pointmaze_graph import (
    adjacency_lists,
    bin_to_state_indices,
    build_dataset_position_graph,
    graph_step_distance_matrix,
    load_graph_npz,
    parse_xy_dims,
    save_graph_npz,
    source_indices,
)


def parse_int_list(value):
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]


def finite_mean(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def get_or_build_graph(args, train_dataset, val_dataset):
    graph_path = Path(args.graph_path)
    if graph_path.exists() and not args.rebuild_graph:
        return load_graph_npz(graph_path)
    graph = build_dataset_position_graph(
        train_dataset,
        val_dataset,
        xy_dims=parse_xy_dims(args.xy_dims),
        bin_size_factor=args.bin_size_factor,
    )
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    save_graph_npz(graph_path, graph)
    return graph


def sample_queries(val_dataset, graph, distance_matrix, args, rng):
    state_to_bin = np.asarray(graph["val_state_to_bin"], dtype=np.int32)
    by_bin = bin_to_state_indices(state_to_bin, len(graph["bin_centers"]))
    has_state = np.asarray([len(items) > 0 for items in by_bin])
    src_idxs = source_indices(val_dataset)
    src_idxs = src_idxs[state_to_bin[src_idxs] >= 0]

    observations = []
    actions = []
    goals = []
    source_bins = []
    goal_bins = []
    direct_distances = []
    attempts = 0
    max_attempts = max(1000, int(args.num_queries) * 200)
    while len(observations) < int(args.num_queries) and attempts < max_attempts:
        attempts += 1
        src_idx = int(rng.choice(src_idxs))
        src_bin = int(state_to_bin[src_idx])
        distances = np.asarray(distance_matrix[src_bin], dtype=np.float32)
        finite = np.isfinite(distances) & has_state
        lo = float(args.query_pos_boundary_frac) * float(args.budget)
        mask = finite & (distances >= lo) & (distances <= float(args.budget))
        if not mask.any() and lo > 0.0:
            mask = finite & (distances <= float(args.budget))
        candidate_goal_bins = np.nonzero(mask)[0]
        if len(candidate_goal_bins) == 0:
            continue
        goal_bin = int(rng.choice(candidate_goal_bins))
        goal_idx = int(rng.choice(by_bin[goal_bin]))
        observations.append(np.asarray(val_dataset["observations"])[src_idx])
        actions.append(np.asarray(val_dataset["actions"])[src_idx])
        goals.append(np.asarray(val_dataset["observations"])[goal_idx])
        source_bins.append(src_bin)
        goal_bins.append(goal_bin)
        direct_distances.append(float(distances[goal_bin]))

    if len(observations) < int(args.num_queries):
        raise ValueError(
            f"Could only sample {len(observations)} graph queries after {attempts} attempts."
        )
    return dict(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        goals=np.asarray(goals, dtype=np.float32),
        source_bins=np.asarray(source_bins, dtype=np.int32),
        goal_bins=np.asarray(goal_bins, dtype=np.int32),
        direct_distances=np.asarray(direct_distances, dtype=np.float32),
    )


def sample_candidates(val_dataset, graph, distance_matrix, queries, args, rng):
    state_to_bin = np.asarray(graph["val_state_to_bin"], dtype=np.int32)
    by_bin = bin_to_state_indices(state_to_bin, len(graph["bin_centers"]))
    has_state = np.asarray([len(items) > 0 for items in by_bin])

    candidate_observations = []
    candidate_bins = []
    source_distances = []
    right_distances = []
    state_valids = []
    oracle_valid_exists = []
    for source_bin, goal_bin in zip(queries["source_bins"], queries["goal_bins"]):
        source_bin = int(source_bin)
        goal_bin = int(goal_bin)
        d_source = np.asarray(distance_matrix[source_bin], dtype=np.float32)
        d_right = np.asarray(distance_matrix[:, goal_bin], dtype=np.float32)
        finite = np.isfinite(d_source) & np.isfinite(d_right) & has_state
        state_valid = (
            finite
            & (d_source <= float(args.left_budget))
            & (d_right <= float(args.right_budget))
        )
        oracle_valid_exists.append(float(state_valid.any()))
        fallback = np.nonzero(finite)[0]
        if len(fallback) == 0:
            raise ValueError("No finite graph subgoal candidates for query.")
        preferred = np.nonzero(state_valid)[0]
        chosen = []
        preferred_count = min(len(preferred), int(args.num_candidates) // 2)
        if preferred_count > 0:
            chosen.extend(
                rng.choice(preferred, size=preferred_count, replace=False).tolist()
            )
        remaining = int(args.num_candidates) - len(chosen)
        pool = np.asarray([idx for idx in fallback if int(idx) not in set(chosen)])
        if len(pool) == 0:
            pool = fallback
        chosen.extend(rng.choice(pool, size=remaining, replace=len(pool) < remaining))
        chosen = np.asarray(chosen[: int(args.num_candidates)], dtype=np.int32)
        subgoal_idxs = [int(rng.choice(by_bin[int(bin_idx)])) for bin_idx in chosen]

        candidate_observations.append(np.asarray(val_dataset["observations"])[subgoal_idxs])
        candidate_bins.append(chosen)
        source_distances.append(d_source[chosen])
        right_distances.append(d_right[chosen])
        state_valids.append(state_valid[chosen].astype(np.float32))

    return dict(
        source_observations=queries["observations"],
        source_actions=queries["actions"],
        goals=queries["goals"],
        source_bins=queries["source_bins"],
        goal_bins=queries["goal_bins"],
        direct_distances=queries["direct_distances"],
        subgoal_observations=np.asarray(candidate_observations, dtype=np.float32),
        subgoal_bins=np.asarray(candidate_bins, dtype=np.int32),
        source_distances=np.asarray(source_distances, dtype=np.float32),
        right_distances=np.asarray(right_distances, dtype=np.float32),
        state_valids=np.asarray(state_valids, dtype=np.float32),
        oracle_valid_exists=np.asarray(oracle_valid_exists, dtype=np.float32),
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
    )


def score_flat(agent, observations, actions, goals, budgets, batch_size):
    chunks = []
    for start in range(0, len(observations), int(batch_size)):
        end = min(start + int(batch_size), len(observations))
        logits = agent.critic_logits_for(
            observations[start:end],
            actions[start:end],
            goals[start:end],
            budgets[start:end],
            offsets=budgets[start:end],
        )
        chunks.append(np.asarray(jax.nn.sigmoid(logits)).mean(axis=0))
    return np.concatenate(chunks)


def score_bmm_v(value_agent, batch, batch_size):
    num_queries, num_candidates = batch["subgoal_bins"].shape
    source_obs = np.repeat(batch["source_observations"][:, None, :], num_candidates, axis=1)
    source_actions = np.repeat(batch["source_actions"][:, None, :], num_candidates, axis=1)
    goals = np.repeat(batch["goals"][:, None, :], num_candidates, axis=1)
    subgoals = batch["subgoal_observations"]
    left = np.full(num_queries * num_candidates, int(batch["left_budget"]), np.int32)
    right = np.full(num_queries * num_candidates, int(batch["right_budget"]), np.int32)
    first = score_flat(
        value_agent,
        source_obs.reshape((-1, source_obs.shape[-1])),
        source_actions.reshape((-1, source_actions.shape[-1])),
        subgoals.reshape((-1, subgoals.shape[-1])),
        left,
        batch_size,
    ).reshape((num_queries, num_candidates))
    second = score_flat(
        value_agent,
        subgoals.reshape((-1, subgoals.shape[-1])),
        source_actions.reshape((-1, source_actions.shape[-1])),
        goals.reshape((-1, goals.shape[-1])),
        right,
        batch_size,
    ).reshape((num_queries, num_candidates))
    return np.minimum(first, second)


def score_rows(value_agent, batch, graph, batch_size, rng):
    centers = np.asarray(graph["bin_centers"], dtype=np.float32)
    source_xy = centers[batch["source_bins"]]
    goal_xy = centers[batch["goal_bins"]]
    candidate_xy = centers[batch["subgoal_bins"]]
    midpoint = 0.5 * (source_xy + goal_xy)
    euclidean_scores = -np.linalg.norm(candidate_xy - midpoint[:, None, :], axis=-1)
    midpoint_error = -(
        np.abs(batch["source_distances"] - float(batch["left_budget"]))
        + np.abs(batch["right_distances"] - float(batch["right_budget"]))
    )
    return [
        ("random", rng.random(batch["subgoal_bins"].shape)),
        ("euclidean_midpoint", euclidean_scores),
        ("oracle_graph_midpoint", midpoint_error),
        ("BMM_V_graph", score_bmm_v(value_agent, batch, batch_size)),
    ]


def row_metrics(scores, batch):
    scores = np.asarray(scores, dtype=np.float32)
    selected = np.argmax(scores, axis=1)
    rows = np.arange(len(selected))
    source_d = batch["source_distances"][rows, selected]
    right_d = batch["right_distances"][rows, selected]
    valid = batch["state_valids"][rows, selected]
    direct_d = batch["direct_distances"]
    midpoint_error = (
        np.abs(source_d - float(batch["left_budget"]))
        + np.abs(right_d - float(batch["right_budget"]))
    )
    return dict(
        state_valid_frac=float(valid.mean()),
        oracle_valid_exists_frac=float(batch["oracle_valid_exists"].mean()),
        selected_source_distance=finite_mean(source_d),
        selected_right_distance=finite_mean(right_d),
        source_path_stretch=finite_mean(source_d + right_d - direct_d),
        midpoint_error=finite_mean(midpoint_error),
        selected_unique_bins=int(len(np.unique(batch["subgoal_bins"][rows, selected]))),
    )


def markdown(result):
    lines = [
        "# BMM graph value-subgoal diagnostic",
        "",
        f"env: `{result['env_name']}`",
        f"graph path: `{result['graph_path']}`",
        f"budget: `{result['budget']}` split `{result['left_budget']}/{result['right_budget']}`",
        f"queries: `{result['num_queries']}`, candidates/query: `{result['num_candidates']}`",
        f"value checkpoint: `{result['value_restore_path']}:{result['value_restore_epoch']}`",
        "",
        "| scorer | state_valid | oracle_valid_exists | path_stretch | midpoint_err | source_d | right_d | unique_bins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        m = row["metrics"]
        lines.append(
            "| {name} | {sv} | {ove} | {ps} | {me} | {sd} | {rd} | {ub} |".format(
                name=row["name"],
                sv=format_metric(m["state_valid_frac"]),
                ove=format_metric(m["oracle_valid_exists_frac"]),
                ps=format_metric(m["source_path_stretch"]),
                me=format_metric(m["midpoint_error"]),
                sd=format_metric(m["selected_source_distance"]),
                rd=format_metric(m["selected_right_distance"]),
                ub=m["selected_unique_bins"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_name", default="pointmaze-medium-navigate-v0")
    parser.add_argument("--dataset_dir", default=None)
    parser.add_argument("--graph_path", default="exp/bmm_pointmaze_graph.npz")
    parser.add_argument("--rebuild_graph", action="store_true")
    parser.add_argument("--xy_dims", default="0,1")
    parser.add_argument("--bin_size_factor", type=float, default=2.0)
    parser.add_argument("--budgets", default="40,80,120,500")
    parser.add_argument("--budget", type=int, default=160)
    parser.add_argument("--left_budget", type=int, default=80)
    parser.add_argument("--right_budget", type=int, default=80)
    parser.add_argument("--query_pos_boundary_frac", type=float, default=0.5)
    parser.add_argument("--num_queries", type=int, default=128)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value_restore_path", required=True)
    parser.add_argument("--value_restore_epoch", type=int, required=True)
    parser.add_argument("--actor_hidden_dims", default="(256, 256)")
    parser.add_argument("--value_hidden_dims", default="(256, 256)")
    parser.add_argument("--layer_norm", default="False")
    parser.add_argument("--score_batch_size", type=int, default=8192)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    rng = np.random.default_rng(args.seed)
    dataset_path = ar.dataset_path_from_dir(args.dataset_dir)
    _, train_dataset, val_dataset = make_env_and_datasets(
        args.env_name, dataset_path=dataset_path
    )
    graph = get_or_build_graph(args, train_dataset, val_dataset)
    adjacency = adjacency_lists(
        len(graph["bin_centers"]), graph["edge_src"], graph["edge_dst"]
    )
    distance_matrix = graph_step_distance_matrix(adjacency, graph)
    queries = sample_queries(val_dataset, graph, distance_matrix, args, rng)
    batch = sample_candidates(val_dataset, graph, distance_matrix, queries, args, rng)

    budgets = parse_int_list(args.budgets)
    value_agent = ar.configure_restore_agent(
        args, train_dataset, budgets, critic_mode="state"
    )
    value_agent = restore_agent(
        value_agent, args.value_restore_path, args.value_restore_epoch
    )
    rows = [
        dict(name=name, metrics=row_metrics(scores, batch))
        for name, scores in score_rows(value_agent, batch, graph, args.score_batch_size, rng)
    ]
    result = dict(
        env_name=args.env_name,
        graph_path=args.graph_path,
        budget=int(args.budget),
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        num_queries=int(batch["subgoal_bins"].shape[0]),
        num_candidates=int(batch["subgoal_bins"].shape[1]),
        value_restore_path=args.value_restore_path,
        value_restore_epoch=int(args.value_restore_epoch),
        rows=rows,
    )
    text = markdown(result)
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
