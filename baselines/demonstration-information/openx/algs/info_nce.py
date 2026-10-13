import functools
from typing import Any, Dict, Tuple

import jax
import optax
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp

from .core import Algorithm


class InfoNCEModel(nn.Module):
    obs_encoder: nn.Module
    action_encoder: nn.Module
    z_dim: int

    def setup(self):
        self.obs_proj = nn.Dense(self.z_dim)
        self.action_proj = nn.Dense(self.z_dim)

    def __call__(self, batch: Dict[str, Any], train: bool = True) -> Tuple[jax.Array, jax.Array]:
        z_obs = self.obs_proj(self.obs_encoder(batch, train=train))
        z_action = self.action_proj(self.action_encoder(batch, train=train))
        return z_obs, z_action


class InfoNCE(Algorithm):
    def __init__(
        self,
        obs_encoder: nn.Module,
        action_encoder: nn.Module,
        z_dim: int,
        loss_type: str = "cosine",
        temperature: float = 1.0,
        normalize: bool = False,
    ):
        assert loss_type in {"cosine", "l2"}
        self.loss_type = loss_type
        self.temperature = temperature
        self.normalize = normalize
        self.model = InfoNCEModel(obs_encoder, action_encoder, z_dim)

    def init(self, batch, tx: optax.GradientTransformation, rng: jax.Array) -> train_state.TrainState:
        """
        Returns a state object and initializes the model.
        """
        params = jax.jit(functools.partial(self.model.init, train=False))(rng, batch)
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
        z_obs, z_action = self.model.apply(
            state.params, batch, rngs=dict(params=params_rng, dropout=dropout_rng), train=False
        )
        if self.normalize:
            z_obs = z_obs / jnp.linalg.norm(z_obs, ord=2, axis=-1, keepdims=True)
            z_action = z_action / jnp.linalg.norm(z_action, ord=2, axis=-1, keepdims=True)

        if self.loss_type == "cosine":
            return jnp.sum(z_obs * z_action, axis=-1)
        if self.loss_type == "l2":
            return -jnp.linalg.norm(z_obs - z_action, axis=-1)
        raise ValueError("Invalid loss_type selected.")

    def encode(self, state: train_state.TrainState, batch: Dict, rng: jax.Array):
        params_rng, dropout_rng = jax.random.split(rng)
        return self.model.apply(state.params, batch, rngs=dict(params=params_rng, dropout=dropout_rng), train=False)

    def _loss(self, params, batch, rng: jax.Array, train: bool = True):
        """
        A helper method to compute the losses. Shared across train and val.
        """
        params_rng, dropout_rng = jax.random.split(rng)
        z_obs, z_action = self.model.apply(
            params,
            batch,
            rngs=dict(params=params_rng, dropout=dropout_rng),
            train=train,
        )

        if self.normalize:
            z_obs = z_obs / jnp.linalg.norm(z_obs, ord=2, axis=-1, keepdims=True)
            z_action = z_action / jnp.linalg.norm(z_action, ord=2, axis=-1, keepdims=True)

        if self.loss_type == "cosine":
            logits = jnp.sum(z_obs[:, None, :] * z_action[None, :, :], axis=-1)
        elif self.loss_type == "l2":
            logits = -jnp.linalg.norm(z_obs[:, None] - z_action[None, :], axis=-1)
        logits = logits / self.temperature

        # Obs loss
        obs_loss = optax.losses.softmax_cross_entropy(logits, jnp.eye(logits.shape[0])).mean()
        action_loss = optax.losses.softmax_cross_entropy(logits.T, jnp.eye(logits.shape[0])).mean()
        obs_accuracy = jnp.mean(jnp.argmax(logits, axis=1) == jnp.arange(logits.shape[0]))
        action_accuracy = jnp.mean(jnp.argmax(logits, axis=0) == jnp.arange(logits.shape[0]))

        loss = obs_loss + action_loss
        info = dict(
            loss=loss,
            obs_loss=obs_loss,
            action_loss=action_loss,
            obs_accuracy=obs_accuracy,
            action_accuracy=action_accuracy,
        )

        return loss, info
