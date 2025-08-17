from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from .distributions import GaussianMixture


class GMVAE(eqx.Module):
    """Gaussian-Mixture Variational Auto-Encoder."""

    encoder: Callable[[Array], Array]
    decoder: Callable[[Array], Array]
    k: int = eqx.field(static=True)
    tau: float = eqx.field(static=True)

    def decode(self, z: Array) -> Array:
        """Compute the likelihood p(x|z).

        Args:
            z (`Array`): Latent variable z.

        Returns:
            Parameters of p(x|z), also known as the reconstruction.
        """
        return self.decoder(z)

    def encode(self, x: Array, *, key: PRNGKeyArray) -> tuple[Array, GaussianMixture]:
        """Compute the variational posterior q(y,z|x) and sample z ~ q(z|x,y).

        Args:
            x (`Array`): Input array.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A 2-tuple containing the latent variable z, and the parameters of q(y,z|x).
        """
        key1, key2 = jr.split(key)
        posterior = self.split(self.encoder(x))
        # categorical posterior q(y|x)
        logits = posterior.logits
        gumbel = jr.gumbel(key1, logits.shape)
        y = jnp.exp(jax.nn.log_softmax((logits + gumbel) / self.tau))
        # gaussian posterior q(z|x,y)
        means, stds = posterior.means, posterior.stds
        mean = jnp.einsum('k,kn->n', y, means)
        std = jnp.einsum('k,kn->n', y, stds)
        z = mean + std * jr.normal(key2, std.shape)
        # return
        return z, posterior

    def split(self, embedding: Array) -> GaussianMixture:
        """Split encoder embeddings into the parameters of p(y) and p(z|y).

        Args:
            embedding (`Array`): Encoder embeddings.

        Returns:
            Parameters of p(y) and p(z|y).
        """
        logits, gaussian = jnp.split(embedding, [self.k])
        means, log_stds = jnp.split(gaussian, 2)
        return GaussianMixture(
            logits=jax.nn.log_softmax(logits),
            means=means.reshape(*logits.shape, -1),
            stds=jnp.exp(log_stds).reshape(*logits.shape, -1),
        )
