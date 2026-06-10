#!/usr/bin/env python
"""Synthetic checks for BMM supervised reachability and ranking fields."""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.bmm_trl import (
    augment_goal_with_budget,
    config_budgets,
    get_config,
    masked_mean,
)
from utils.datasets import Dataset, GCDataset


def make_fake_dataset(num_trajs=4, traj_len=80, obs_dim=5, action_dim=2):
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
    config.batch_size = 64
    config.discount = 0.9
    config.budgets = (1, 2, 4, 8, 16, 32)
    config.max_budget = 32
    config.num_sup_pairs = 12
    config.num_rank_pairs = 10
    return config


def main():
    np.random.seed(0)
    config = make_config()
    dataset = GCDataset(make_fake_dataset(), config)
    batch = dataset.sample(config.batch_size)

    assert np.allclose(batch["actor_goals"], batch["actor_goal_observations"])

    sup_keys = {
        "value_sup_observations",
        "value_sup_actions",
        "value_sup_goals",
        "value_sup_budgets",
        "value_sup_offsets",
        "value_sup_labels",
        "value_sup_valids",
    }
    rank_keys = {
        "value_rank_observations",
        "value_rank_actions",
        "value_rank_goals",
        "value_rank_pos_budgets",
        "value_rank_neg_budgets",
        "value_rank_offsets",
        "value_rank_valids",
    }
    missing = (sup_keys | rank_keys) - set(batch.keys())
    assert not missing, f"Missing BMM supervised fields: {sorted(missing)}"

    assert batch["value_sup_budgets"].shape == (
        config.num_sup_pairs,
        config.batch_size,
    )
    assert batch["value_sup_observations"].shape[:2] == (
        config.num_sup_pairs,
        config.batch_size,
    )
    assert batch["value_rank_pos_budgets"].shape == (
        config.num_rank_pairs,
        config.batch_size,
    )

    sup_valid = batch["value_sup_valids"] > 0
    sup_pos = sup_valid & (batch["value_sup_labels"] == 1)
    sup_neg = sup_valid & (batch["value_sup_labels"] == 0)
    assert np.all(batch["value_sup_offsets"][sup_pos] <= batch["value_sup_budgets"][sup_pos])
    assert np.all(batch["value_sup_offsets"][sup_neg] > batch["value_sup_budgets"][sup_neg])
    assert np.any(
        sup_neg & (batch["value_sup_budgets"] == config.max_budget)
    ), "Expected at least one valid high-budget supervised negative."

    rank_valid = batch["value_rank_valids"] > 0
    assert np.all(
        batch["value_rank_neg_budgets"][rank_valid]
        < batch["value_rank_offsets"][rank_valid]
    )
    assert np.all(
        batch["value_rank_offsets"][rank_valid]
        <= batch["value_rank_pos_budgets"][rank_valid]
    )

    budgets = config_budgets(config)
    goals = batch["value_sup_goals"][0]
    goal_budgets = batch["value_sup_budgets"][0]
    augmented = augment_goal_with_budget(
        goals,
        goal_budgets,
        config.max_budget,
        budgets=budgets,
        budget_feature="log_scalar_onehot",
    )
    assert augmented.shape[-1] == goals.shape[-1] + 1 + len(budgets)
    one_hot = np.asarray(augmented[:, -len(budgets) :])
    expected_idxs = np.searchsorted(np.asarray(budgets), goal_budgets, side="left")
    assert np.array_equal(one_hot.argmax(axis=-1), expected_idxs)
    assert np.allclose(one_hot.sum(axis=-1), 1.0)

    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    mask = jnp.asarray(
        [[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.float32
    )
    got = float(masked_mean(x, mask))
    expected = float((x * mask[None, ...]).sum() / (mask.sum() * x.shape[0]))
    assert np.isclose(got, expected), (got, expected)

    print("BMM supervised and ranking shape checks passed.")


if __name__ == "__main__":
    main()
