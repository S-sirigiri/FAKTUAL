import functools
from typing import Any, Dict, Tuple

import jax
import optax
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp

from .core import Algorithm


def _sim(z_obs, z_goal, temperature=1.0):
    # We need to implement a custom norm to avoid NaN gradients when obs == goal.
    diff = z_obs - z_goal
    norm = jnp.sum(diff * diff, axis=-1)
    norm = jnp.where(norm == 0, 1e-14, norm)  # Avoid sqrt on zero
    norm = jnp.sqrt(norm)
    return -norm / temperature


class VIPModel(nn.Module):
    obs_encoder: nn.Module
    z_dim: int

    def setup(self):
        self.obs_proj = nn.Dense(self.z_dim)

    def __call__(self, obs: Dict[str, Any], train: bool = True) -> Tuple[jax.Array, jax.Array]:
        return self.obs_proj(self.obs_encoder(obs, train=train))


class VIP(Algorithm):
    def __init__(
        self, obs_encoder: nn.Module, z_dim: int, temperature: float = 1.0, gamma: float = 0.98, num_negatives: int = 0
    ):
        self.temperature = temperature
        self.model = obs_encoder
        self.gamma = gamma
        self.num_negatives = num_negatives
        self.model = VIPModel(obs_encoder=obs_encoder, z_dim=z_dim)

    def init(self, batch, tx: optax.GradientTransformation, rng: jax.Array) -> train_state.TrainState:
        """
        Returns a state object and initializes the model.
        """
        params = jax.jit(functools.partial(self.model.init, train=False))(rng, batch["observation"])
        return train_state.TrainState.create(apply_fn=self._loss, params=params, tx=tx)

    def train_step(
        self, state: train_state.TrainState, batch: Dict, rng: jax.Array
    ) -> Tuple[train_state.TrainState, Dict[str, jax.Array]]:
        """
        The main train step function. This should return a dictionary of metrics.
        """
        grads, info = jax.grad(state.apply_fn, has_aux=True)(state.params, batch, rng, train=True)
        new_state = state.apply_gradients(grads=grads)
        return new_state, info

    def val_step(self, state: train_state.TrainState, batch: Dict, rng: jax.Array) -> Dict[str, jax.Array]:
        """
        Take a validation step and return a dictionary of metrics.
        """
        _, info = state.apply_fn(state.params, batch, rng, train=False)
        return info

    def predict(self, state: train_state.TrainState, batch: Dict, rng: jax.Array) -> Any:
        params_rng, dropout_rng = jax.random.split(rng)
        obs_all = jax.tree.map(
            lambda *x: jnp.concatenate(x, axis=0), batch["observation"], batch["next_observation"], batch["goal"]
        )
        z_all = self.model.apply(state.params, obs_all, rngs=dict(params=params_rng, dropout=dropout_rng), train=False)
        z_obs, z_next_obs, z_goal = jnp.split(z_all, 3, axis=0)
        vs = _sim(z_obs, z_goal, temperature=self.temperature)
        vs_next = _sim(z_next_obs, z_goal, temperature=self.temperature)
        return vs_next - vs

    def _loss(self, params, batch, rng: jax.Array, train: bool = True):
        """
        A helper method to compute the losses. Shared across train and val.
        Taken from https://github.com/facebookresearch/vip/blob/main/vip/trainer.py
        """
        params_rng, dropout_rng, negatives_rng = jax.random.split(rng, num=3)
        # Stack everything together to pass in a single batch.
        obs_all = jax.tree.map(
            lambda *x: jnp.concatenate(x, axis=0),
            batch["initial_observation"],
            batch["observation"],
            batch["next_observation"],
            batch["goal"],
        )
        z_all = self.model.apply(params, obs_all, rngs=dict(params=params_rng, dropout=dropout_rng), train=train)
        z_init, z_obs, z_obs_next, z_goal = jnp.split(z_all, 4, axis=0)
        batch_size = z_obs.shape[0]

        # Compute the loss.
        v0 = _sim(z_init, z_goal, temperature=self.temperature)
        vs = _sim(z_obs, z_goal, temperature=self.temperature)
        vs_next = _sim(z_obs_next, z_goal, temperature=self.temperature)
        reward = (batch["horizon"] == 1).astype(jnp.float32) - 1
        target = reward + self.gamma * vs_next
        v_loss = (1 - self.gamma) * -jnp.mean(v0) + jax.nn.logsumexp(vs - target, b=1 / batch_size)
        if self.num_negatives > 0:
            # If we have negatives, add an extra loss for it. We do this by randomly permuting the goals
            # This was used in the original paper for Ego4D.
            idx = jax.random.uniform(
                negatives_rng, shape=(self.num_negatives * batch_size), minval=0, maxval=batch_size, dtype=jnp.int32
            )
            vs_neg = _sim(z_obs[idx], jnp.tile(z_goal, self.num_negatives), temperature=self.temperature)
            vs_next_neg = _sim(z_obs_next[idx], jnp.tile(z_goal, self.num_negatives), temperature=self.temperature)
            neg_reward = -jnp.ones((self.num_negatives * batch_size,))
            neg_target = neg_reward + self.gamma * vs_next_neg
            v_loss += jax.nn.logsumexp(vs_neg - neg_target, b=1 / (batch_size * self.num_negatives))

        return v_loss, dict(v_loss=v_loss)
