#!/usr/bin/env python
"""Synthetic checks for the BMM offline action-ranking diagnostic."""

from pathlib import Path
import sys
import tempfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_bmm_action_ranking as ranking


def main():
    name, path, epoch = ranking.parse_critic_spec("B=exp/run/critic:1000")
    assert name == "B"
    assert path == "exp/run/critic"
    assert epoch == 1000
    assert ranking.parse_int_list("(40, 80, 160)") == (40, 80, 160)
    assert ranking.parse_int_list("40,80") == (40, 80)

    distances = np.asarray(
        [
            [3.0, 1.0, 6.0, 2.0],
            [7.0, 4.0, 5.0, 8.0],
        ],
        dtype=np.float32,
    )
    labels = (distances <= 4.0).astype(np.float32)
    source_distances = np.asarray([4.0, 6.0], dtype=np.float32)
    good_scores = -distances
    bad_scores = distances
    acc, count = ranking.pairwise_ranking_accuracy(good_scores, distances)
    assert count == 12
    assert np.isclose(acc, 1.0)
    bad_acc, _ = ranking.pairwise_ranking_accuracy(bad_scores, distances)
    assert np.isclose(bad_acc, 0.0)

    metrics = ranking.action_ranking_metrics(
        good_scores, labels, distances, source_distances
    )
    assert np.isclose(metrics["pairwise_accuracy"], 1.0)
    assert metrics["action_auc"] > 0.99
    assert np.isclose(metrics["selected_distance_mean"], 2.5)
    assert np.isclose(metrics["logged_distance_mean"], 5.0)
    assert np.isclose(metrics["selected_improves_frac"], 1.0)

    interp = np.max(
        np.stack(
            [
                np.asarray([[0.1, 0.7], [0.4, 0.3]]),
                np.asarray([[0.5, 0.2], [0.2, 0.9]]),
            ],
            axis=0,
        ),
        axis=0,
    )
    np.testing.assert_allclose(interp, np.asarray([[0.5, 0.7], [0.4, 0.9]]))

    queries = {
        "observations": np.zeros((2, 4, 3), dtype=np.float32),
        "candidate_observations": np.arange(24, dtype=np.float32).reshape(2, 4, 3),
        "actions": np.zeros((2, 4, 2), dtype=np.float32),
        "next_observations": np.ones((2, 4, 3), dtype=np.float32),
        "goals": np.ones((2, 4, 3), dtype=np.float32),
        "labels": labels,
        "distances": distances,
        "source_distances": source_distances,
        "source_cells": np.asarray([1, 2], dtype=np.int32),
        "goal_cells": np.asarray([3, 4], dtype=np.int32),
        "candidate_source_idxs": np.arange(8, dtype=np.int32).reshape(2, 4),
        "candidate_next_cells": np.asarray([[1, 2, 2, 3], [4, 4, 5, 6]], dtype=np.int32),
        "budget": 4,
        "remaining_budget": 3.0,
    }
    baselines = ranking.baseline_score_tables(queries, np.random.default_rng(0))
    oracle_metrics = ranking.action_ranking_metrics(
        baselines["oracle_distance"], labels, distances, source_distances
    )
    assert np.isclose(oracle_metrics["pairwise_accuracy"], 1.0)
    source_metrics = ranking.action_ranking_metrics(
        baselines["source_distance"], labels, distances, source_distances
    )
    assert np.isclose(source_metrics["pairwise_accuracy"], 0.5)
    diag = ranking.candidate_diagnostics(queries, xy_dims=(0, 1))
    assert np.isclose(diag["candidate_next_distance_spread_mean"], 4.5)
    assert np.isclose(diag["oracle_best_distance_mean"], 2.5)
    assert diag["candidate_source_position_spread_mean"] > 0.0
    assert np.isclose(diag["candidate_unique_next_cell_count_mean"], 3.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "queries.npz"
        ranking.save_query_cache(cache_path, queries)
        loaded = ranking.load_query_cache(cache_path)
        assert loaded["budget"] == 4
        assert np.isclose(loaded["remaining_budget"], 3.0)
        for key in ranking.QUERY_CACHE_KEYS:
            np.testing.assert_allclose(loaded[key], queries[key])

    print("BMM action-ranking diagnostic checks passed.")


if __name__ == "__main__":
    main()
