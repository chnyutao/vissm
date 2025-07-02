from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from distrax import MultivariateNormalDiag as MvNormal
from jaxtyping import Array, PRNGKeyArray, PyTree


class VAE(eqx.Module):
    """Variational Auto-Encodr."""

    encoder: Callable[[Array], Array]
    decoder: Callable[[Array], Array]

    def encode(self, x: Array, *, key: PRNGKeyArray) -> tuple[Array, PyTree]:
        """Compute the variational posterior q(z|x).

        Args:
            x (`Array`): Input array.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A 2-tuple containing the latent variable z, and the parameters of q(z|x).
        """
        mean, log_std = jnp.split(self.encoder(x), 2)
        std = jnp.exp(log_std)
        z = mean + std * jr.normal(key, std.shape)
        return z, (mean, std)

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> tuple[Array, PyTree]:
        """Forward the input through the variational auto-encoder.

        Args:
            x (Array): Input array.
            key (PRNGKeyArray): JAX random key.

        Returns:
            A 2-tuple containing the reconstructed input, and the parameters of
            the posterior and prior distributions.
        """
        z, posterior = self.encode(x, key=key)
        return self.decoder(z), {
            'posterior': posterior,
            'prior': (jnp.zeros_like(z), jnp.ones_like(z)),
        }


def loss_fn(model: VAE, x: Array, *, key: PRNGKeyArray) -> tuple[Array, PyTree]:
    """negative evidence lower bound (elbo)."""
    batch_size = x.shape[0]
    x_hat, dists = jax.vmap(model)(x, key=jr.split(key, batch_size))
    # reconstruction error
    reconst = jnp.sum((x - x_hat) ** 2, axis=range(1, len(x.shape))).mean()
    # gaussian kld(q(z|x) || p(z))
    posteriors, priors = dists['posterior'], dists['prior']
    kld = MvNormal(*posteriors).kl_divergence(MvNormal(*priors)).mean()
    # return
    loss = reconst + kld
    return loss, {'loss': loss, 'reconst': reconst, 'kld': kld}
