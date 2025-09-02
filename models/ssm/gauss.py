from typing import TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from models.distributions import Gaussian
from models.transition import GaussTr
from models.vae import GaussVAE


class Result(TypedDict):
    posterior: Gaussian
    prior: Gaussian
    reconst: Array


class GaussSSM(eqx.Module):
    """State-Space Model with Gaussian Prior and Posterior."""

    vae: GaussVAE
    tr: GaussTr

    def __call__(
        self,
        data: tuple[Array, Array, Array],
        *,
        key: PRNGKeyArray,
    ) -> Result:
        """
        Forward the input data through the model.

        Args:
            data (`tuple[Array, Array, Array]`):
               A 3-tuple containing the current state, action, and next state.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A dictionary containing model outputs.
        """
        s, a, ns = data  # state, action, next_state
        key1, key2 = jr.split(key)
        # prior (transition)
        z = self.vae.encode(s).sample(key=key1)
        prior = self.tr(z, a)
        # posterior
        posterior = self.vae.encode(ns)
        # reconstruction
        z = posterior.sample(key=key2)
        reconst = self.vae.decode(z)
        # return
        return {
            'posterior': posterior,
            'prior': prior,
            'reconst': reconst,
        }

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(
        self,
        data: tuple[Array, Array, Array],
        *,
        key: PRNGKeyArray,
    ) -> tuple[Array, dict[str, Array]]:
        """
        Compute the loss and associated metrics of the model.

        Args:
            data (`tuple[Array, Array, Array]`):
                A 3-tuple containing the current state, action and next state.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A 2-tuple containing the loss and asscoiated metrics.
        """
        batch_size = data[0].shape[0]
        results = jax.vmap(self)(data, key=jr.split(key, batch_size))
        # reconstruction error
        x = data[-1]
        x_hat = results['reconst'].reshape(x.shape)
        reconst = jnp.sum((x - x_hat) ** 2, axis=range(1, x.ndim)).mean()
        # kld (posterior || prior)
        posterior, prior = results['posterior'].to(), results['prior'].to()
        kld = posterior.kl_divergence(prior).mean()
        # return loss + metrics
        loss = reconst + kld
        return loss, {'train/loss': loss, 'train/reconst': reconst, 'train/kld': kld}
