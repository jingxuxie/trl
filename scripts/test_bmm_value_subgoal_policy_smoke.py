#!/usr/bin/env python
"""Synthetic checks for value-subgoal policy-smoke selector comparison."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_bmm_value_subgoal_policy_smoke as smoke


def make_policy(selector):
    policy = object.__new__(smoke.ValueSubgoalNNPolicy)
    policy.selector = smoke.normalize_selector(selector)
    policy.left_budget = 10
    policy.right_budget = 10
    policy.step_distances = np.asarray(
        [
            [0.0, 10.0, 2.0, 8.0],
            [10.0, 0.0, 8.0, 2.0],
            [2.0, 8.0, 0.0, 6.0],
            [8.0, 2.0, 6.0, 0.0],
        ],
        dtype=np.float32,
    )
    policy.rng = np.random.default_rng(0)
    return policy


def main():
    candidate_cells = np.asarray([1, 2, 3], dtype=np.int32)
    subgoals = np.asarray(
        [
            [10.0, 0.0],
            [5.0, 0.0],
            [8.0, 0.0],
        ],
        dtype=np.float32,
    )
    observation = np.asarray([0.0, 0.0], dtype=np.float32)
    goal = np.asarray([10.0, 0.0], dtype=np.float32)

    geometric = make_policy("geometric")
    geometric_scores = geometric.selector_scores(
        observation, goal, candidate_cells, subgoals, source_cell=0, goal_cell=1
    )
    assert int(np.argmax(geometric_scores)) == 1

    oracle = make_policy("oracle_midpoint")
    oracle_scores = oracle.selector_scores(
        observation, goal, candidate_cells, subgoals, source_cell=0, goal_cell=1
    )
    assert int(np.argmax(oracle_scores)) == 0

    random = make_policy("random")
    random_scores = random.selector_scores(
        observation, goal, candidate_cells, subgoals, source_cell=0, goal_cell=1
    )
    assert random_scores.shape == (3,)

    assert smoke.normalize_selector("bmm-v") == "BMM_V"
    assert smoke.parse_str_list("random, geometric, BMM_V") == [
        "random",
        "geometric",
        "BMM_V",
    ]

    rows = [
        dict(success=0.0, final_goal_distance=1.0, goal_distance_improvement=2.0),
        dict(success=1.0, final_goal_distance=3.0, goal_distance_improvement=4.0),
    ]
    for row in rows:
        row.update(
            steps=5,
            start_goal_distance=5.0,
            mean_step_goal_improvement=0.5,
            mean_step_subgoal_improvement=0.25,
            subgoal_reduce_frac=1.0,
            goal_reduce_frac=0.5,
            subgoal_valid_frac=1.0,
            selected_score_mean=0.0,
            selected_source_to_subgoal=10.0,
            selected_subgoal_to_goal=10.0,
        )
    agg = smoke.aggregate(rows)
    assert np.isclose(agg["success"], 0.5)
    assert np.isclose(agg["goal_distance_improvement"], 3.0)

    print("BMM value-subgoal policy-smoke selector checks passed.")


if __name__ == "__main__":
    main()
