from collections.abc import Callable
from typing import TypedDict

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class Distribution(TypedDict):
    mean: Array
    std: Array


class VAE(eqx.Module):
    """Variational Auto-Encodr."""

    encoder: Callable[[Array], Array]
    decoder: Callable[[Array], Array]

    def decode(self, z: Array) -> Array:
        """Compute the likelihood p(x|z).

        Args:
            z (`Array`): Latent variable z.

        Returns:
            Parameters of p(x|z), also known as the reconstruction.
        """
        return self.decoder(z)

    def encode(self, x: Array, *, key: PRNGKeyArray) -> tuple[Array, Distribution]:
        """Compute the variational posterior q(z|x) and sample z ~ q(z|x).

        Args:
            x (`Array`): Input array.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A 2-tuple containing the latent variable z, and the parameters of q(z|x).
        """
        posterior = self.split(self.encoder(x))
        mean, std = posterior['mean'], posterior['std']
        z = mean + std * jr.normal(key, std.shape)
        return z, posterior

    def split(self, embedding: Array) -> Distribution:
        """Split encoder embeddings into the parameters of p(z).

        Args:
            embedding (`Array`): Encoder embeddings.

        Returns:
            Parameters of p(z).
        """
        mean, log_std = jnp.split(embedding, 2)
        return {'mean': mean, 'std': jnp.exp(log_std)}
