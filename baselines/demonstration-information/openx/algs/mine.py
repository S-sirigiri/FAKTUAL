import functools
from typing import Any, Dict, Tuple

import jax
import optax
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp

from .core import Algorithm


class MINETrainState(train_state.TrainState):
    log_denom: jax.Array


class MINEModel(nn.Module):
    obs_action_encoder: nn.Module

    def setup(self):
        self.proj = nn.Dense(1)

    @nn.compact
    def __call__(self, batch: Dict[str, Any], train: bool = True) -> Tuple[jax.Array, jax.Array]:
        x = self.obs_action_encoder(batch, train=train)
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


class MINE(Algorithm):
    def __init__(
        self,
        obs_action_encoder: nn.Module,
        alpha: float = 0.9,
    ):
        self.model = MINEModel(obs_action_encoder)
        self.alpha = alpha

    def init(self, batch, tx: optax.GradientTransformation, rng: jax.Array) -> train_state.TrainState:
        """
        Returns a state object and initializes the model.
        """
        params = jax.jit(functools.partial(self.model.init, train=False))(rng, batch)
        # We will also need to add the denominator EMA to the state.
        log_denom = -jnp.inf
        return MINETrainState.create(apply_fn=self.model.apply, params=params, tx=tx, log_denom=log_denom)

    def train_step(
        self, state: train_state.TrainState, batch: Dict, rng: jax.Array
    ) -> Tuple[train_state.TrainState, Dict[str, jax.Array]]:
        """
        The main train step function. This should return a dictionary of metrics.
        """
        (
            perm_rng,
            pair_p_rng,
            pair_d_rng,
            marginal_p_rng,
            marginal_d_rng,
        ) = jax.random.split(rng, 5)

        batch_marginal = {
            "observation": batch["observation"],
            "action": jax.random.permutation(perm_rng, batch["action"], axis=0),
        }
        batch_size = batch["action"].shape[0]

        paired_mean, paired_grad = jax.value_and_grad(
            lambda params: jnp.mean(
                state.apply_fn(params, batch, rngs=dict(params=pair_p_rng, dropout=pair_d_rng), train=True)
            )
        )(state.params)
        lse_mean, lse_grad = jax.value_and_grad(
            lambda params: jax.nn.logsumexp(
                state.apply_fn(
                    params, batch_marginal, rngs=dict(params=marginal_p_rng, dropout=marginal_d_rng), train=True
                ),
                b=1 / batch_size,
            )
        )(state.params)

        log_denom = jnp.logaddexp(jnp.log(self.alpha) + state.log_denom, jnp.log(1 - self.alpha) + lse_mean)

        # Now merge both terms to get the correct gradient.
        correction = jnp.exp(lse_mean - log_denom)
        grads = jax.tree.map(lambda mean, lse: -mean + correction * lse, paired_grad, lse_grad)

        new_state = state.apply_gradients(grads=grads, log_denom=log_denom)
        info = dict(loss=-paired_mean + lse_mean)
        return new_state, info

    def val_step(self, state: train_state.TrainState, batch: Dict, rng: jax.Array) -> Dict[str, jax.Array]:
        """
        Take a validation step and return a dictionary of metrics.
        """
        (
            perm_rng,
            pair_p_rng,
            pair_d_rng,
            marginal_p_rng,
            marginal_d_rng,
        ) = jax.random.split(rng, 5)

        batch_marginal = {
            "observation": batch["observation"],
            "action": jax.random.permutation(perm_rng, batch["action"], axis=0),
        }
        batch_size = batch["action"].shape[0]

        mean = jnp.mean(
            state.apply_fn(state.params, batch, rngs=dict(params=pair_p_rng, dropout=pair_d_rng), train=True)
        )
        lse = jax.nn.logsumexp(
            state.apply_fn(
                state.params, batch_marginal, rngs=dict(params=marginal_p_rng, dropout=marginal_d_rng), train=True
            ),
            b=1 / batch_size,
        )

        return dict(loss=-mean + lse)

    def predict(self, state: train_state.TrainState, batch: Dict, rng: jax.Array) -> Any:
        params_rng, dropout_rng = jax.random.split(rng)
        # We just return the paired value as the un-paired term does not affect the value.
        return state.apply_fn(state.params, batch, rngs=dict(params=params_rng, dropout=dropout_rng), train=False)
