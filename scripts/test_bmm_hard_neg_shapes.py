#!/usr/bin/env python
"""Synthetic shape and validity checks for BMM hard negatives."""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.bmm_trl import get_config
from utils.datasets import Dataset, GCDataset


def make_fake_dataset(num_trajs=3, traj_len=40, obs_dim=5, action_dim=2):
    size = num_trajs * traj_len
    observations = np.arange(size * obs_dim, dtype=np.float32).reshape(size, obs_dim)
    observations = observations / observations.max()
    actions = np.linspace(-1.0, 1.0, size * action_dim, dtype=np.float32).reshape(
        size, action_dim
    )
    terminals = np.zeros(size, dtype=np.float32)
    terminals[traj_len - 1 :: traj_len] = 1.0
    return Dataset.create(
        observations=observations,
        actions=actions,
        terminals=terminals,
    )


def make_config(batch_size):
    config = get_config()
    config.batch_size = batch_size
    config.discount = 0.9
    config.budgets = (1, 2, 4, 8, 16)
    config.max_budget = 16
    config.hard_neg_min_factor = 1.25
    config.hard_neg_max_factor = 4.0
    return config


def main():
    np.random.seed(0)
    batch_size = 128
    config = make_config(batch_size)
    dataset = GCDataset(make_fake_dataset(), config)
    batch = dataset.sample(batch_size, idxs=np.zeros(batch_size, dtype=np.int32))

    required_keys = {
        "value_hard_neg_budgets",
        "value_hard_neg_goals",
        "value_hard_neg_offsets",
        "value_hard_neg_valids",
    }
    missing = required_keys - set(batch.keys())
    assert not missing, f"Missing hard-negative fields: {sorted(missing)}"

    for key in required_keys:
        assert batch[key].shape[0] == batch_size, (key, batch[key].shape)

    valid = batch["value_hard_neg_valids"] > 0
    assert valid.any(), "Expected valid hard negatives from trajectory starts."
    assert np.all(batch["value_hard_neg_offsets"][valid] > 0)
    assert np.all(
        batch["value_hard_neg_offsets"][valid]
        > batch["value_hard_neg_budgets"][valid]
    )
    assert np.all(
        batch["value_hard_neg_offsets"][valid]
        >= np.ceil(config.hard_neg_min_factor * batch["value_hard_neg_budgets"][valid])
    )
    assert np.all(
        batch["value_hard_neg_offsets"][valid]
        <= np.floor(config.hard_neg_max_factor * batch["value_hard_neg_budgets"][valid])
    )
    assert np.any(np.isin(batch["value_hard_neg_budgets"][valid], [4, 8, 16]))
    assert np.any(batch["value_hard_neg_budgets"][valid] == 16)

    print("BMM hard-negative shape checks passed.")


if __name__ == "__main__":
    main()
