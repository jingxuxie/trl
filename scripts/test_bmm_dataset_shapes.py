#!/usr/bin/env python
"""Synthetic batch-shape checks for BMM-TRL dataset fields."""

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


def make_fake_dataset(num_trajs=4, traj_len=12, obs_dim=5, action_dim=2):
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


def make_config():
    config = get_config()
    config.batch_size = 16
    config.discount = 0.9
    config.budgets = (1, 2, 4, 8)
    config.max_budget = 8
    return config


def main():
    np.random.seed(0)
    config = make_config()
    dataset = GCDataset(make_fake_dataset(), config)
    batch = dataset.sample(config.batch_size)

    required_keys = {
        "value_offsets",
        "value_budgets",
        "value_left_budgets",
        "value_right_budgets",
        "value_midpoint_offsets",
        "value_midpoint_observations",
        "value_midpoint_actions",
        "value_midpoint_goals",
        "trans_valids",
        "value_neg_budgets",
        "value_neg_valids",
        "value_random_goals",
        "value_random_goal_observations",
        "value_random_budgets",
        "mono_low_budgets",
        "mono_high_budgets",
    }
    missing = required_keys - set(batch.keys())
    assert not missing, f"Missing BMM fields: {sorted(missing)}"

    for key in required_keys:
        assert batch[key].shape[0] == config.batch_size, (key, batch[key].shape)

    assert np.all(batch["value_offsets"] >= 1)
    assert np.all(batch["value_offsets"] <= batch["value_budgets"])
    assert np.all(batch["value_budgets"] <= config.max_budget)
    assert np.all(batch["value_left_budgets"] >= 1)
    assert np.all(batch["value_right_budgets"] >= 1)
    assert np.all(batch["mono_low_budgets"] <= batch["mono_high_budgets"])

    neg_valid = batch["value_neg_valids"] > 0
    assert np.all(
        batch["value_neg_budgets"][neg_valid]
        < config.budget_neg_frac * batch["value_offsets"][neg_valid]
    )

    print("BMM dataset shape checks passed.")


if __name__ == "__main__":
    main()
