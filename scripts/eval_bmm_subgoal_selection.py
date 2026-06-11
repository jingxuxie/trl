#!/usr/bin/env python
"""Subgoal-selection diagnostic for BMM Q/V critics."""

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


def score_flat(agent, observations, actions, goals, budgets, batch_size):
    mean_scores = []
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
    return np.concatenate(mean_scores)


def sample_subgoal_candidates(
    val_dataset,
    context,
    queries,
    budget,
    left_budget,
    right_budget,
    num_candidates,
    rng,
):
    state_to_cell = np.asarray(context["val_state_to_cell"], dtype=np.int32)
    goal_by_cell = context["val_goal_by_cell"]
    has_state = np.asarray([len(items) > 0 for items in goal_by_cell])
    cell_distances_raw = np.asarray(context["cell_distances"], dtype=np.int32)
    step_distances = cell_distances_raw.astype(np.float32) * float(
        context["distance_scale"]
    )

    source_cells = np.asarray(queries["source_cells"], dtype=np.int32)
    goal_cells = np.asarray(queries["goal_cells"], dtype=np.int32)
    logged_idxs = np.asarray(queries["candidate_source_idxs"], dtype=np.int32)[:, 0]
    next_cells = state_to_cell[logged_idxs + 1]
    observations = queries["observations"][:, 0]
    actions = queries["actions"][:, 0]
    goals = queries["goals"][:, 0]

    candidate_observations = []
    candidate_cells = []
    source_distances = []
    next_distances = []
    right_distances = []
    direct_source_distances = []
    direct_next_distances = []
    state_valids = []
    action_valids = []

    for source_cell, next_cell, goal_cell in zip(source_cells, next_cells, goal_cells):
        source_cell = int(source_cell)
        next_cell = int(next_cell)
        goal_cell = int(goal_cell)
        finite = (
            (cell_distances_raw[source_cell] >= 0)
            & (cell_distances_raw[next_cell] >= 0)
            & (cell_distances_raw[:, goal_cell] >= 0)
            & has_state
        )
        d_source = step_distances[source_cell]
        d_next = step_distances[next_cell]
        d_right = step_distances[:, goal_cell]
        state_valid = finite & (d_source <= float(left_budget)) & (
            d_right <= float(right_budget)
        )
        action_valid = finite & (d_next <= max(float(left_budget) - 1.0, 1.0)) & (
            d_right <= float(right_budget)
        )
        preferred = np.nonzero(action_valid | state_valid)[0]
        fallback = np.nonzero(finite)[0]
        if len(fallback) == 0:
            raise ValueError("No finite subgoal candidates for a cached query.")

        chosen = []
        preferred_count = min(len(preferred), int(num_candidates) // 2)
        if preferred_count > 0:
            chosen.extend(
                rng.choice(preferred, size=preferred_count, replace=False).tolist()
            )
        remaining = int(num_candidates) - len(chosen)
        pool = np.asarray([cell for cell in fallback if cell not in set(chosen)])
        if len(pool) == 0:
            pool = fallback
        chosen.extend(
            rng.choice(pool, size=remaining, replace=len(pool) < remaining).tolist()
        )
        chosen = np.asarray(chosen[: int(num_candidates)], dtype=np.int32)

        subgoal_idxs = [
            int(rng.choice(goal_by_cell[int(cell)]))
            for cell in chosen
        ]
        candidate_observations.append(np.asarray(val_dataset["observations"])[subgoal_idxs])
        candidate_cells.append(chosen)
        source_distances.append(d_source[chosen])
        next_distances.append(d_next[chosen])
        right_distances.append(d_right[chosen])
        direct_source_distances.append(float(step_distances[source_cell, goal_cell]))
        direct_next_distances.append(float(step_distances[next_cell, goal_cell]))
        state_valids.append(state_valid[chosen].astype(np.float32))
        action_valids.append(action_valid[chosen].astype(np.float32))

    return dict(
        source_observations=np.asarray(observations, dtype=np.float32),
        source_actions=np.asarray(actions, dtype=np.float32),
        goals=np.asarray(goals, dtype=np.float32),
        subgoal_observations=np.asarray(candidate_observations, dtype=np.float32),
        subgoal_cells=np.asarray(candidate_cells, dtype=np.int32),
        source_distances=np.asarray(source_distances, dtype=np.float32),
        next_distances=np.asarray(next_distances, dtype=np.float32),
        right_distances=np.asarray(right_distances, dtype=np.float32),
        direct_source_distances=np.asarray(direct_source_distances, dtype=np.float32),
        direct_next_distances=np.asarray(direct_next_distances, dtype=np.float32),
        state_valids=np.asarray(state_valids, dtype=np.float32),
        action_valids=np.asarray(action_valids, dtype=np.float32),
        budget=int(budget),
        left_budget=int(left_budget),
        right_budget=int(right_budget),
    )


def tiled_source(batch):
    num_queries, num_candidates = batch["subgoal_cells"].shape
    observations = np.repeat(
        batch["source_observations"][:, None, :], num_candidates, axis=1
    )
    actions = np.repeat(batch["source_actions"][:, None, :], num_candidates, axis=1)
    goals = np.repeat(batch["goals"][:, None, :], num_candidates, axis=1)
    return observations, actions, goals


def score_vv(value_agent, batch, batch_size):
    num_queries, num_candidates = batch["subgoal_cells"].shape
    source_obs, source_actions, goals = tiled_source(batch)
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


def score_qv(agent, value_agent, batch, batch_size):
    num_queries, num_candidates = batch["subgoal_cells"].shape
    source_obs, source_actions, goals = tiled_source(batch)
    subgoals = batch["subgoal_observations"]
    left = np.full(num_queries * num_candidates, int(batch["left_budget"]), np.int32)
    right = np.full(num_queries * num_candidates, int(batch["right_budget"]), np.int32)
    first = score_flat(
        agent,
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


def selection_metrics(scores, batch):
    scores = np.asarray(scores, dtype=np.float64)
    selected = np.argmax(scores, axis=1)
    rows = np.arange(scores.shape[0])
    source_d = batch["source_distances"][rows, selected]
    next_d = batch["next_distances"][rows, selected]
    right_d = batch["right_distances"][rows, selected]
    state_valid = batch["state_valids"][rows, selected]
    action_valid = batch["action_valids"][rows, selected]
    direct_source = batch["direct_source_distances"]
    direct_next = batch["direct_next_distances"]
    left_budget = float(batch["left_budget"])
    right_budget = float(batch["right_budget"])
    midpoint_error = np.abs(source_d - left_budget) + np.abs(right_d - right_budget)
    action_midpoint_error = np.abs(next_d - max(left_budget - 1.0, 1.0)) + np.abs(
        right_d - right_budget
    )
    return dict(
        state_valid_frac=float(state_valid.mean()),
        action_valid_frac=float(action_valid.mean()),
        selected_source_distance=float(source_d.mean()),
        selected_next_distance=float(next_d.mean()),
        selected_right_distance=float(right_d.mean()),
        source_path_stretch=float((source_d + right_d - direct_source).mean()),
        next_path_stretch=float((next_d + right_d - direct_next).mean()),
        midpoint_error=float(midpoint_error.mean()),
        action_midpoint_error=float(action_midpoint_error.mean()),
        selected_unique_cells=int(len(np.unique(batch["subgoal_cells"][rows, selected]))),
    )


def oracle_baselines(batch, rng):
    state_error = np.abs(batch["source_distances"] - float(batch["left_budget"])) + np.abs(
        batch["right_distances"] - float(batch["right_budget"])
    )
    action_error = np.abs(
        batch["next_distances"] - max(float(batch["left_budget"]) - 1.0, 1.0)
    ) + np.abs(batch["right_distances"] - float(batch["right_budget"]))
    return {
        "oracle_state_midpoint": -state_error,
        "oracle_action_midpoint": -action_error,
        "oracle_state_valid": batch["state_valids"],
        "oracle_action_valid": batch["action_valids"],
        "random": rng.random(batch["subgoal_cells"].shape),
    }


def markdown(result):
    lines = [
        "# BMM subgoal-selection diagnostic",
        "",
        f"env: `{result['env_name']}`",
        f"budget: `{result['budget']}` split `{result['left_budget']}/{result['right_budget']}`",
        f"queries: `{result['num_queries']}`, candidates/query: `{result['num_candidates']}`",
        "",
        "| scorer | state_valid | action_valid | source_d | next_d | right_d | source_stretch | next_stretch | midpoint_err | action_mid_err | unique_cells |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        m = row["metrics"]
        lines.append(
            "| {name} | {sv} | {av} | {sd} | {nd} | {rd} | {ss} | {ns} | {me} | {ame} | {uc} |".format(
                name=row["name"],
                sv=format_metric(m["state_valid_frac"]),
                av=format_metric(m["action_valid_frac"]),
                sd=format_metric(m["selected_source_distance"]),
                nd=format_metric(m["selected_next_distance"]),
                rd=format_metric(m["selected_right_distance"]),
                ss=format_metric(m["source_path_stretch"]),
                ns=format_metric(m["next_path_stretch"]),
                me=format_metric(m["midpoint_error"]),
                ame=format_metric(m["action_midpoint_error"]),
                uc=m["selected_unique_cells"],
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
    parser.add_argument("--left_budget", type=int, default=80)
    parser.add_argument("--right_budget", type=int, default=80)
    parser.add_argument("--query_cache_path", required=True)
    parser.add_argument("--num_queries", type=int, default=512)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--critics", nargs="+", required=True)
    parser.add_argument("--value_restore_path", required=True)
    parser.add_argument("--value_restore_epoch", type=int, required=True)
    parser.add_argument("--actor_hidden_dims", default="(256, 256)")
    parser.add_argument("--value_hidden_dims", default="(256, 256)")
    parser.add_argument("--layer_norm", default="False")
    parser.add_argument("--score_batch_size", type=int, default=8192)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    budgets = ar.parse_int_list(args.budgets)
    rng = np.random.default_rng(args.seed)
    dataset_path = ar.dataset_path_from_dir(args.dataset_dir)
    env, train_dataset, val_dataset = make_env_and_datasets(
        args.env_name, dataset_path=dataset_path
    )
    context = ar.make_grid_context(
        env,
        train_dataset,
        val_dataset,
        ar.parse_xy_dims(args.xy_dims),
        args.geodesic_budget_unit,
    )
    queries = ar.load_query_cache(args.query_cache_path)
    if args.num_queries > 0:
        for key in ar.QUERY_CACHE_KEYS:
            queries[key] = queries[key][: args.num_queries]
    batch = sample_subgoal_candidates(
        val_dataset,
        context,
        queries,
        args.budget,
        args.left_budget,
        args.right_budget,
        args.num_candidates,
        rng,
    )

    value_agent = ar.configure_restore_agent(
        args, train_dataset, budgets, critic_mode="state"
    )
    value_agent = restore_agent(
        value_agent, args.value_restore_path, args.value_restore_epoch
    )

    rows = []
    for name, scores in oracle_baselines(batch, rng).items():
        rows.append(dict(name=name, metrics=selection_metrics(scores, batch)))
    rows.append(dict(name="V/V_teacher", metrics=selection_metrics(score_vv(value_agent, batch, args.score_batch_size), batch)))

    for spec in args.critics:
        name, restore_path, restore_epoch = ar.parse_critic_spec(spec)
        agent = ar.configure_restore_agent(args, train_dataset, budgets, critic_mode="action")
        agent = restore_agent(agent, restore_path, restore_epoch)
        rows.append(
            dict(
                name=f"{name}_Q/V",
                metrics=selection_metrics(
                    score_qv(agent, value_agent, batch, args.score_batch_size),
                    batch,
                ),
            )
        )

    result = dict(
        env_name=args.env_name,
        geodesic_budget_unit=args.geodesic_budget_unit,
        budget=int(args.budget),
        left_budget=int(args.left_budget),
        right_budget=int(args.right_budget),
        num_queries=int(batch["subgoal_cells"].shape[0]),
        num_candidates=int(batch["subgoal_cells"].shape[1]),
        query_cache_path=args.query_cache_path,
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
