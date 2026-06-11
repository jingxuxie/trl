#!/usr/bin/env python
"""Synthetic checks for BMM subgoal-selection diagnostics."""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import eval_bmm_subgoal_selection as subgoal


def main():
    batch = {
        "subgoal_cells": np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "source_distances": np.asarray(
            [[80.0, 20.0, 120.0], [30.0, 80.0, 140.0]], dtype=np.float32
        ),
        "next_distances": np.asarray(
            [[79.0, 19.0, 119.0], [29.0, 79.0, 139.0]], dtype=np.float32
        ),
        "right_distances": np.asarray(
            [[80.0, 140.0, 20.0], [130.0, 80.0, 20.0]], dtype=np.float32
        ),
        "direct_source_distances": np.asarray([160.0, 160.0], dtype=np.float32),
        "direct_next_distances": np.asarray([159.0, 159.0], dtype=np.float32),
        "state_valids": np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        "action_valids": np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        "left_budget": 80,
        "right_budget": 80,
    }
    baselines = subgoal.oracle_baselines(batch, np.random.default_rng(0))
    metrics = subgoal.selection_metrics(baselines["oracle_state_midpoint"], batch)
    assert np.isclose(metrics["state_valid_frac"], 1.0)
    assert np.isclose(metrics["action_valid_frac"], 1.0)
    assert np.isclose(metrics["midpoint_error"], 0.0)
    assert np.isclose(metrics["source_path_stretch"], 0.0)
    assert metrics["selected_unique_cells"] == 2

    random_metrics = subgoal.selection_metrics(
        np.zeros_like(baselines["oracle_state_midpoint"]), batch
    )
    assert random_metrics["state_valid_frac"] == 0.5

    print("BMM subgoal-selection diagnostic checks passed.")


if __name__ == "__main__":
    main()
