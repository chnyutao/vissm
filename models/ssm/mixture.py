from typing import TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from models.distributions import Gaussian, GaussianMixture

from .transition import MixtureTr
from .vae import GaussVAE


class Result(TypedDict):
    posterior: Gaussian
    prior: GaussianMixture
    reconst: Array


class MixtureSSM(eqx.Module):
    """State-Space Model with Gaussian Mixture Prior and Gaussian Posterior."""

    vae: GaussVAE
    tr: MixtureTr

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
        key, *keys = jr.split(key, 1 + batch_size)
        results = jax.vmap(self)(data, key=jnp.array(keys))
        # reconstruction error
        x = data[-1]
        x_hat = results['reconst'].reshape(x.shape)
        reconst = jnp.sum((x - x_hat) ** 2, axis=range(1, x.ndim)).mean()
        # kld (posterior || prior)
        posterior, prior = results['posterior'], results['prior']
        ## 1. monte-carlo estimation
        z = posterior.sample(key=key)
        logp = jnp.array(prior.to().log_prob(z))
        logq = jnp.array(posterior.to().log_prob(z))
        logr = logp - logq
        kld = ((jnp.exp(logr) - 1) - logr).mean()
        ## 2. upper bound
        pi = jnp.exp(prior.weight.logits)
        klds = jax.vmap(lambda q, p: q.to().kl_divergence(p.to()), in_axes=(None, 1))(
            posterior,  # batch_size * latent_size
            prior.components,  # batch_size * n_components * latent_size
        )
        kld = jnp.einsum('bk,kb->b', pi, klds).mean()
        ## 3. monte-carlo upper bound
        indices = (jnp.arange(batch_size), prior.weight.sample(key=key).argmax(axis=-1))
        component = jax.tree.map(lambda x: x[indices], prior.components)
        kld = posterior.to().kl_divergence(component.to()).mean()
        # return loss + metrics
        loss = reconst + kld
        return loss, {
            'train/loss': loss,
            'train/reconst': reconst,
            'train/kld': kld,
            'train/entropy': prior.weight.to().entropy().mean(),
        }
