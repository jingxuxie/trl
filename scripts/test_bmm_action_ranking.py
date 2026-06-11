#!/usr/bin/env python
"""Synthetic checks for the BMM offline action-ranking diagnostic."""

from pathlib import Path
import sys

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

    print("BMM action-ranking diagnostic checks passed.")


if __name__ == "__main__":
    main()
