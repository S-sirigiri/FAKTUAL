import functools
from typing import Any, Dict, Tuple

import flax
import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp

from .core import Algorithm


class VAEEncoder(nn.Module):
    model: nn.Module
    z_dim: int

    @nn.compact
    def __call__(self, batch, train: bool = True):
        x = self.model(batch, train=train)
        mean, logvar = jnp.split(
            nn.Dense(2 * self.z_dim, kernel_init=nn.initializers.xavier_uniform(), name="z_proj")(x), 2, axis=-1
        )
        return mean, logvar


class VAEDecoder(nn.Module):
    model: nn.Module

    @nn.compact
    def __call__(self, z, batch, train: bool = True):
        x_hat = self.model(z, batch, train=train)

        def _output_proj(x_hat, x, name):
            matching_dims = 0
            for dim_x, dim_x_hat in zip(x.shape, x_hat.shape, strict=False):
                if dim_x == dim_x_hat:
                    matching_dims += 1
                else:
                    break

            if matching_dims < len(x.shape):
                x_hat = nn.Dense(
                    np.prod(x.shape[matching_dims:]), kernel_init=nn.initializers.xavier_uniform(), name=name + "_proj"
                )(x_hat)
            return jnp.reshape(x_hat, x.shape)

        x_flat = flax.traverse_util.flatten_dict(batch)
        x_hat_flat = flax.traverse_util.flatten_dict(x_hat)
        x_hat_flat = {k: _output_proj(v, x_flat[k], str(k)) for k, v in x_hat_flat.items()}
        return flax.traverse_util.unflatten_dict(x_hat_flat)


class BetaVaeModel(nn.Module):
    """
    This is a Dummy Module that allows for factoring of the encoder and
    decoder into a single module, and implements the reparameterization trick.
    """

    encoder: nn.Module
    decoder: nn.Module

    def __call__(self, batch, train: bool = True):
        # Does the reparameterization trick
        mean, logvar = self.encoder(batch, train=train)
        stddev = jnp.exp(0.5 * logvar)
        # Reparameterize.
        z = mean + stddev * jax.random.normal(self.make_rng("params"), mean.shape)
        x_hat = self.decoder(z, batch, train=train)
        return x_hat, mean, logvar

    def encode(self, batch, train: bool = True):
        return self.encoder(batch, train=train)[0]  # Return just the mean

    def decode(self, z, batch, train: bool = True):
        return self.decoder(z, batch, train=train)


class BetaVAE(Algorithm):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, z_dim: int, beta: float = 1.0, weights: dict | None = None
    ):
        self.model = BetaVaeModel(VAEEncoder(encoder, z_dim), VAEDecoder(decoder))
        self.weights = None if weights is None else {k.replace("->", "."): v for k, v in weights.items()}
        self.beta = beta

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
        return self.model.apply(
            state.params, batch, rngs=dict(params=params_rng, dropout=dropout_rng), train=False, method="encode"
        )

    def _loss(self, params, batch, rng: jax.Array, train: bool = True):
        """
        A helper method to compute the losses. Shared across train and val.
        """
        params_rng, dropout_rng = jax.random.split(rng)
        x_hat, mean, logvar = self.model.apply(
            params,
            batch,
            rngs=dict(params=params_rng, dropout=dropout_rng),
            train=train,
        )

        kl = 0.5 * jnp.sum(-logvar - 1.0 + jnp.exp(logvar) + jnp.square(mean), axis=-1)
        kl_loss = jnp.mean(kl)  # minimize the KL

        # Compute the reconstruction loss per-component, if we have actions, apply the mask.
        def _recon_loss(x_hat, x, k):
            reduction_axes = -1 if k == "action" else tuple(range(1, len(x.shape)))
            loss = jnp.sum((x - x_hat) ** 2, axis=reduction_axes)
            if k == "action":  # Then apply second reduction.
                loss = jnp.sum(loss * batch["mask"], axis=-1)
            return jnp.mean(loss)

        x_hat = flax.traverse_util.flatten_dict(x_hat, sep=".")  # Avoid tuple to allow for return
        x = flax.traverse_util.flatten_dict(batch, sep=".")  # Avoid tuple to allow for return
        reconstruction_loss_per_key = {k: _recon_loss(x_hat[k], x[k], k) for k in x_hat}
        if self.weights is not None:
            reconstruction_loss_per_key = {k: self.weights[k] * v for k, v in reconstruction_loss_per_key.items()}
        reconstruction_loss = sum(reconstruction_loss_per_key.values())
        loss = reconstruction_loss + self.beta * kl_loss

        info = {"recon_loss/" + k: v for k, v in reconstruction_loss_per_key.items()}
        info["loss"] = loss
        info["kl_loss"] = kl_loss
        info["recon_loss/total"] = reconstruction_loss

        return loss, info
