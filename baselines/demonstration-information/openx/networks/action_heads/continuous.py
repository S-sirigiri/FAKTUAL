import jax
from flax import linen as nn
from jax import numpy as jnp

from . import core


def _ensemblize(ensemble_size: int, fn: str | None = None):
    if fn is None:
        fn = "__call__"
    return nn.vmap(
        lambda module, *args: getattr(module, fn)(*args),
        in_axes=None,
        out_axes=0,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        axis_size=ensemble_size,
    )


class L2ActionHead(core.ActionHead):
    ensemble_size: int | None = None

    @nn.compact
    def __call__(self, obs: jax.Array, train: bool = True):
        x = self.model(obs, train=train) if self.model is not None else obs
        if self.action_horizon is None:
            return nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.Dense(self.action_dim * self.action_horizon, kernel_init=nn.initializers.xavier_uniform())(x)
        return jnp.reshape(x, (x.shape[0], self.action_horizon, self.action_dim))

    def predict(self, obs: jax.Array, train: bool = True):
        if self.ensemble_size is not None:
            return jnp.mean(_ensemblize(self.ensemble_size)(self, obs, train), axis=0)
        return self(obs, train=train)

    def loss(self, obs: jax.Array, action: jax.Array, train: bool = True):
        if self.ensemble_size is not None:
            pred = _ensemblize(self.ensemble_size)(self, obs, train)
        else:
            pred = self(obs, train=train)
        loss = jnp.square(pred - action).sum(axis=-1)  # (B, T, D) --> (B, T)
        if self.ensemble_size is not None:
            loss = jnp.mean(loss, axis=0)
        return loss

    def predict_with_confidence(self, obs: jax.Array, train: bool = True):
        assert self.ensemble_size is not None
        pred = _ensemblize(self.ensemble_size)(self, obs, train)
        mean = jnp.mean(pred, axis=0)  # (B, T, D)
        stddev = jnp.mean(jnp.std(pred, axis=0), axis=(1, 2))
        return mean, stddev


class L1ActionHead(core.ActionHead):
    @nn.compact
    def __call__(self, obs: jax.Array, train: bool = True):
        x = self.model(obs, train=train) if self.model is not None else obs
        if self.action_horizon is None:
            return nn.Dense(self.action_dim, kernel_init=nn.initializers.xavier_uniform())(x)
        x = nn.Dense(self.action_dim * self.action_horizon, kernel_init=nn.initializers.xavier_uniform())(x)
        return jnp.reshape(x, (x.shape[0], self.action_horizon, self.action_dim))

    def predict(self, obs: jax.Array, train: bool = True):
        return self(obs, train=train)

    def loss(self, obs: jax.Array, action: jax.Array, train: bool = True):
        pred = self(obs, train=train)
        return jnp.abs(pred - action).sum(axis=-1)  # (B, T, D) --> (B, T)
