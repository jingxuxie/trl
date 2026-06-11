#!/usr/bin/env python
"""Synthetic checks for graph value-subgoal diagnostic metrics."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_bmm_graph_value_subgoal as diag


def main():
    batch = dict(
        subgoal_bins=np.asarray([[1, 2], [2, 1]], dtype=np.int32),
        source_distances=np.asarray([[5.0, 8.0], [8.0, 5.0]], dtype=np.float32),
        right_distances=np.asarray([[5.0, 9.0], [9.0, 5.0]], dtype=np.float32),
        direct_distances=np.asarray([10.0, 10.0], dtype=np.float32),
        state_valids=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        oracle_valid_exists=np.asarray([1.0, 1.0], dtype=np.float32),
        left_budget=5,
        right_budget=5,
    )
    scores = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    metrics = diag.row_metrics(scores, batch)
    assert np.isclose(metrics["state_valid_frac"], 1.0)
    assert np.isclose(metrics["oracle_valid_exists_frac"], 1.0)
    assert np.isclose(metrics["selected_source_distance"], 5.0)
    assert np.isclose(metrics["selected_right_distance"], 5.0)
    assert np.isclose(metrics["source_path_stretch"], 0.0)
    assert np.isclose(metrics["midpoint_error"], 0.0)
    assert metrics["selected_unique_bins"] == 1

    bad_scores = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    bad_metrics = diag.row_metrics(bad_scores, batch)
    assert np.isclose(bad_metrics["state_valid_frac"], 0.0)
    assert bad_metrics["source_path_stretch"] > metrics["source_path_stretch"]

    print("BMM graph value-subgoal diagnostic checks passed.")


if __name__ == "__main__":
    main()
