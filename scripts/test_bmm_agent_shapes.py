#!/usr/bin/env python
"""Tiny BMM-TRL agent update and action-sampling smoke test."""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.bmm_trl import BMMTRLAgent, get_config
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
    config.batch_size = 8
    config.discount = 0.9
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.layer_norm = False
    config.budgets = (1, 2, 4, 8)
    config.max_budget = 8
    config.frs.flow_steps = 2
    config.frs.num_samples = 4
    return config


def main():
    np.random.seed(0)
    config = make_config()
    dataset = GCDataset(make_fake_dataset(), config)
    example_batch = dataset.sample(1)
    batch = dataset.sample(config.batch_size)

    agent = BMMTRLAgent.create(0, example_batch, config)
    agent, info = agent.update(batch)
    required_metrics = {
        "critic/total_loss",
        "critic/loss_trans",
        "critic/loss_pos",
        "critic/loss_budget_neg",
        "critic/loss_hard_neg",
        "critic/loss_rand_hinge",
        "critic/loss_mono",
        "actor/actor_loss",
    }
    missing = required_metrics - set(info.keys())
    assert not missing, f"Missing update metrics: {sorted(missing)}"
    for key, value in info.items():
        assert bool(jnp.all(jnp.isfinite(value))), f"Non-finite metric {key}: {value}"

    actions = agent.sample_actions(
        batch["observations"],
        goals=batch["actor_goals"],
        seed=jax.random.PRNGKey(1),
    )
    assert actions.shape == batch["actions"].shape, (actions.shape, batch["actions"].shape)
    assert bool(jnp.all(jnp.isfinite(actions)))

    single_action = agent.sample_actions(
        batch["observations"][0],
        goals=batch["actor_goals"][0],
        seed=jax.random.PRNGKey(2),
    )
    assert single_action.shape == batch["actions"][0].shape
    assert bool(jnp.all(jnp.isfinite(single_action)))

    one_batch_action = agent.sample_actions(
        batch["observations"][:1],
        goals=batch["actor_goals"][:1],
        seed=jax.random.PRNGKey(3),
    )
    assert one_batch_action.shape == batch["actions"][:1].shape
    assert bool(jnp.all(jnp.isfinite(one_batch_action)))

    zero_config = make_config()
    zero_config.lambda_hard_neg = 0.0
    zero_dataset = GCDataset(make_fake_dataset(), zero_config)
    zero_example_batch = zero_dataset.sample(1)
    zero_batch = zero_dataset.sample(zero_config.batch_size)
    zero_agent = BMMTRLAgent.create(4, zero_example_batch, zero_config)
    _, zero_info = zero_agent.update(zero_batch)
    assert "critic/loss_hard_neg" in zero_info
    for key, value in zero_info.items():
        assert bool(jnp.all(jnp.isfinite(value))), f"Non-finite metric {key}: {value}"

    print("BMM agent update and sample_actions smoke checks passed.")


if __name__ == "__main__":
    main()
