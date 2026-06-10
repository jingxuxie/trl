import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    ActorVectorField,
    GCActor,
    GCDiscreteActor,
    GCDiscreteCritic,
    GCValue,
)


BMM_DEFAULT_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def normalize_budget(budget, max_budget):
    """Map positive step budgets to a scalar log feature in [0, 1]."""
    budget = jnp.asarray(budget, dtype=jnp.float32)
    budget = jnp.maximum(budget, 1.0)
    denom = jnp.maximum(jnp.log2(float(max_budget)), 1.0)
    return jnp.log2(budget) / denom


def augment_goal_with_budget(goal, budget, max_budget):
    """Append a normalized log-budget scalar to vector goals."""
    if isinstance(goal, (dict, flax.core.FrozenDict)):
        raise ValueError("BMM-TRL first pass supports vector goals only, not dict goals.")

    goal = jnp.asarray(goal)
    if goal.ndim not in (2, 3):
        raise ValueError(
            "BMM-TRL first pass supports vector goals with shape [B, G] or "
            f"[N, B, G], got shape {goal.shape}."
        )

    b = normalize_budget(budget, max_budget)
    if b.ndim == len(goal.shape[:-1]) + 1 and b.shape[-1] == 1:
        b = jnp.squeeze(b, axis=-1)
    if b.ndim == 0:
        b = jnp.full(goal.shape[:-1], b, dtype=goal.dtype)
    else:
        b = jnp.broadcast_to(b, goal.shape[:-1]).astype(goal.dtype)
    return jnp.concatenate([goal, b[..., None]], axis=-1)


def actor_budget(goals, max_budget):
    """Return a max-budget array matching the leading dimensions of goals."""
    return jnp.full(goals.shape[:-1], max_budget, dtype=jnp.float32)


def masked_mean(x, mask, eps=1e-8):
    """Mean of x under a batch mask for either [B] or [E, B] tensors."""
    mask = jnp.asarray(mask, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask[None, :]
    denom = mask.sum()
    if x.ndim == 2:
        denom = denom * x.shape[0]
    return (x * mask).sum() / (denom + eps)


class BMMTRLAgent(flax.struct.PyTreeNode):
    """Budgeted Max-Min TRL prototype agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def bce_loss(pred_logit, target):
        log_pred = jax.nn.log_sigmoid(pred_logit)
        log_not_pred = jax.nn.log_sigmoid(-pred_logit)
        loss = -(log_pred * target + log_not_pred * (1 - target))
        return loss

    def critic_loss(self, batch, grad_params):
        if self.config["oracle_distill"]:
            raise ValueError("BMM-TRL prototype does not support oracle_distill=True.")

        goal_key = "value_goals"
        max_budget = self.config["max_budget"]
        aug_value_goals = augment_goal_with_budget(
            batch[goal_key], batch["value_budgets"], max_budget
        )
        r_logits = self.network.select("critic")(
            batch["observations"],
            goals=aug_value_goals,
            actions=batch["actions"],
            params=grad_params,
        )
        r = jax.nn.sigmoid(r_logits)

        aug_midpoint_goals = augment_goal_with_budget(
            batch["value_midpoint_goals"],
            batch["value_left_budgets"],
            max_budget,
        )
        first_logits = self.network.select("target_critic")(
            batch["observations"],
            goals=aug_midpoint_goals,
            actions=batch["actions"],
        )
        first_r = jax.nn.sigmoid(first_logits)

        aug_right_goals = augment_goal_with_budget(
            batch[goal_key],
            batch["value_right_budgets"],
            max_budget,
        )
        second_logits = self.network.select("target_critic")(
            batch["value_midpoint_observations"],
            goals=aug_right_goals,
            actions=batch["value_midpoint_actions"],
        )
        second_r = jax.nn.sigmoid(second_logits)

        y_trans = jax.lax.stop_gradient(jnp.minimum(first_r, second_r))
        loss_trans = masked_mean(
            self.bce_loss(r_logits, y_trans), batch["trans_valids"]
        )
        loss_pos = self.bce_loss(r_logits, jnp.ones_like(r_logits)).mean()

        aug_neg_goals = augment_goal_with_budget(
            batch[goal_key], batch["value_neg_budgets"], max_budget
        )
        neg_logits = self.network.select("critic")(
            batch["observations"],
            goals=aug_neg_goals,
            actions=batch["actions"],
            params=grad_params,
        )
        neg_r = jax.nn.sigmoid(neg_logits)
        loss_budget_neg = masked_mean(
            self.bce_loss(neg_logits, jnp.zeros_like(neg_logits)),
            batch["value_neg_valids"],
        )

        aug_rand_goals = augment_goal_with_budget(
            batch["value_random_goals"], batch["value_random_budgets"], max_budget
        )
        rand_logits = self.network.select("critic")(
            batch["observations"],
            goals=aug_rand_goals,
            actions=batch["actions"],
            params=grad_params,
        )
        rand_r = jax.nn.sigmoid(rand_logits)
        loss_rand_hinge = jnp.mean(
            jnp.maximum(rand_r - self.config["rand_hinge_rho"], 0.0) ** 2
        )

        aug_low_goals = augment_goal_with_budget(
            batch[goal_key], batch["mono_low_budgets"], max_budget
        )
        aug_high_goals = augment_goal_with_budget(
            batch[goal_key], batch["mono_high_budgets"], max_budget
        )
        low_logits = self.network.select("critic")(
            batch["observations"],
            goals=aug_low_goals,
            actions=batch["actions"],
            params=grad_params,
        )
        high_logits = self.network.select("critic")(
            batch["observations"],
            goals=aug_high_goals,
            actions=batch["actions"],
            params=grad_params,
        )
        low_r = jax.nn.sigmoid(low_logits)
        high_r = jax.nn.sigmoid(high_logits)
        loss_mono = jnp.mean(jnp.maximum(low_r - high_r, 0.0) ** 2)

        total_loss = (
            self.config["lambda_trans"] * loss_trans
            + self.config["lambda_pos"] * loss_pos
            + self.config["lambda_budget_neg"] * loss_budget_neg
            + self.config["lambda_rand_hinge"] * loss_rand_hinge
            + self.config["lambda_mono"] * loss_mono
        )

        return total_loss, {
            "total_loss": total_loss,
            "loss_trans": loss_trans,
            "loss_pos": loss_pos,
            "loss_budget_neg": loss_budget_neg,
            "loss_rand_hinge": loss_rand_hinge,
            "loss_mono": loss_mono,
            "r_mean": r.mean(),
            "r_min": r.min(),
            "r_max": r.max(),
            "y_trans_mean": masked_mean(y_trans, batch["trans_valids"]),
            "first_r_mean": first_r.mean(),
            "second_r_mean": second_r.mean(),
            "pos_r_mean": r.mean(),
            "neg_r_mean": neg_r.mean(),
            "rand_r_mean": rand_r.mean(),
            "mono_violation": (low_r > high_r).mean(),
            "trans_valid_frac": batch["trans_valids"].mean(),
            "neg_valid_frac": batch["value_neg_valids"].mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the actor loss."""
        pe_info = self.config[self.config["pe_type"]]

        if self.config["pe_type"] == "rpg":
            dist = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], params=grad_params
            )
            if pe_info["const_std"]:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

            aug_actor_goals = augment_goal_with_budget(
                batch["actor_goals"],
                actor_budget(batch["actor_goals"], self.config["max_budget"]),
                self.config["max_budget"],
            )
            q1, q2 = self.network.select("critic")(
                batch["observations"], aug_actor_goals, q_actions
            )
            q = jnp.minimum(q1, q2)

            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch["actions"])
            bc_loss = -(pe_info["alpha"] * log_prob).mean()
            actor_loss = q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
                "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                "std": jnp.mean(dist.scale_diag),
            }

        elif self.config["pe_type"] == "discrete":
            dist = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], params=grad_params
            )

            n_actions = jnp.repeat(
                jnp.expand_dims(jnp.arange(0, pe_info["action_ct"]), 1),
                self.config["batch_size"],
                axis=1,
            )
            n_obs = jnp.repeat(
                jnp.expand_dims(batch["observations"], 0), pe_info["action_ct"], axis=0
            )
            n_goals = jnp.repeat(
                jnp.expand_dims(batch["actor_goals"], 0), pe_info["action_ct"], axis=0
            )
            aug_n_goals = augment_goal_with_budget(
                n_goals,
                actor_budget(n_goals, self.config["max_budget"]),
                self.config["max_budget"],
            )

            q = self.network.select("critic")(n_obs, aug_n_goals, n_actions).mean(axis=0)
            v = jnp.sum(q * dist.probs.T, axis=0)
            q_loss = -v.mean()

            log_prob = dist.log_prob(batch["actions"])
            bc_loss = -(pe_info["alpha"] * log_prob).mean()
            actor_loss = q_loss + bc_loss

            return actor_loss, {
                "actor_loss": actor_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
                "q_mean": q.mean(),
                "q_abs_mean": jnp.abs(q).mean(),
                "bc_log_prob": log_prob.mean(),
            }

        elif self.config["pe_type"] == "frs":
            batch_size, action_dim = batch["actions"].shape
            x_rng, t_rng = jax.random.split(rng, 2)

            x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
            x_1 = batch["actions"]
            t = jax.random.uniform(t_rng, (batch_size, 1))
            x_t = (1 - t) * x_0 + t * x_1
            y = x_1 - x_0

            pred = self.network.select("actor")(
                batch["observations"], batch["actor_goals"], x_t, t, params=grad_params
            )
            actor_loss = jnp.mean((pred - y) ** 2)

            return actor_loss, {
                "actor_loss": actor_loss,
            }

        raise ValueError(f"Unsupported pe_type: {self.config['pe_type']}")

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        pe_info = self.config[self.config["pe_type"]]

        if self.config["pe_type"] == "frs":
            if goals is None:
                raise ValueError("BMM-TRL FRS sampling requires goals.")
            n_observations = jnp.repeat(
                jnp.expand_dims(observations, 0), pe_info["num_samples"], axis=0
            )
            n_goals = jnp.repeat(
                jnp.expand_dims(goals, 0), pe_info["num_samples"], axis=0
            )

            n_actions = jax.random.normal(
                seed,
                (
                    pe_info["num_samples"],
                    *observations.shape[:-1],
                    self.config["action_dim"],
                ),
            )
            for i in range(pe_info["flow_steps"]):
                t = jnp.full(
                    (pe_info["num_samples"], *observations.shape[:-1], 1),
                    i / pe_info["flow_steps"],
                )
                vels = self.network.select("actor")(
                    n_observations, n_goals, n_actions, t
                )
                n_actions = n_actions + vels / pe_info["flow_steps"]
            n_actions = jnp.clip(n_actions, -1, 1)

            aug_n_goals = augment_goal_with_budget(
                n_goals,
                actor_budget(n_goals, self.config["max_budget"]),
                self.config["max_budget"],
            )
            q = self.network.select("critic")(
                n_observations, goals=aug_n_goals, actions=n_actions
            )
            q = jnp.min(q, axis=0)

            if len(observations.shape) == 2:
                actions = n_actions[
                    jnp.argmax(q, axis=0), jnp.arange(observations.shape[0])
                ]
            else:
                actions = n_actions[jnp.argmax(q)]

            return actions

        else:
            dist = self.network.select("actor")(
                observations, goals, temperature=temperature
            )
            actions = dist.sample(seed=seed)

            if self.config["pe_type"] != "discrete":
                actions = jnp.clip(actions, -1, 1)

            return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        if config["oracle_distill"]:
            raise ValueError("BMM-TRL prototype does not support oracle_distill=True.")
        if config["budget_feature"] != "log_scalar":
            raise ValueError("BMM-TRL prototype only supports budget_feature='log_scalar'.")
        if config["split_mode"] != "half":
            raise ValueError("BMM-TRL prototype only supports split_mode='half'.")

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch["observations"]
        ex_actions = example_batch["actions"]
        ex_goals = example_batch["actor_goals"]
        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]
        pe_info = config[config["pe_type"]]

        if config["pe_type"] == "discrete":
            critic_def = GCDiscreteCritic(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                num_ensembles=2,
                action_dim=config["discrete"]["action_ct"],
            )
        else:
            critic_def = GCValue(
                hidden_dims=config["value_hidden_dims"],
                layer_norm=config["layer_norm"],
                num_ensembles=2,
            )

        if config["pe_type"] == "frs":
            actor_def = ActorVectorField(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                layer_norm=config["layer_norm"],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions, ex_times)
        elif config["pe_type"] == "discrete":
            actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=config["discrete"]["action_ct"],
                layer_norm=config["layer_norm"],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)
        else:
            actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                layer_norm=config["layer_norm"],
                state_dependent_std=False,
                const_std=pe_info["const_std"],
            )
            ex_actor_in = (ex_observations, ex_goals, ex_actions)

        ex_critic_goals = augment_goal_with_budget(
            ex_goals,
            actor_budget(ex_goals, config["max_budget"]),
            config["max_budget"],
        )
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_critic_goals, ex_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_critic_goals, ex_actions),
            ),
            actor=(actor_def, ex_actor_in),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]

        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params["modules_target_critic"] = params["modules_critic"]

        config["action_dim"] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="bmm_trl",
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(1024,) * 4,
            value_hidden_dims=(1024,) * 4,
            layer_norm=True,
            discount=0.999,
            tau=0.005,
            lam=0.0,
            expectile=0.7,
            oracle_distill=False,
            pe_type="frs",  # frs (flow rejection sampling), rpg (reparameterized grads), discrete
            frs=mlc.ConfigDict(dict(flow_steps=10, num_samples=32)),
            rpg=mlc.ConfigDict(dict(alpha=0.03, const_std=True)),
            discrete=mlc.ConfigDict(dict(alpha=0.03, action_ct=0)),
            budgets=BMM_DEFAULT_BUDGETS,
            max_budget=1024,
            use_budget_goal_aug=True,
            budget_feature="log_scalar",
            split_mode="half",
            split_min_frac=0.25,
            num_split_samples=1,
            num_witnesses=1,
            lambda_trans=0.5,
            lambda_pos=1.0,
            lambda_budget_neg=0.05,
            lambda_rand_hinge=0.02,
            lambda_mono=0.05,
            budget_neg_frac=0.5,
            rand_hinge_rho=0.3,
            actor_budget_mode="max",
            actor_budget_threshold=0.5,
            dataset=mlc.ConfigDict(
                dict(
                    dataset_class="GCDataset",
                    value_p_curgoal=0.0,
                    value_p_trajgoal=1.0,
                    value_p_randomgoal=0.0,
                    value_geom_sample=True,
                    actor_p_curgoal=0.0,
                    actor_p_trajgoal=0.5,
                    actor_p_randomgoal=0.5,
                    actor_geom_sample=True,
                )
            ),
        )
    )
    return config
