import equinox as eqx
import equinox.nn as nn
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class MLPEncoder(eqx.Module):
    """MLP Encoder for 64x64 images."""

    layers: nn.Sequential

    def __init__(self, latent_size: int, *, key: PRNGKeyArray):
        """Initializes the MLP encoder.

        Args:
            latent_size (`int`): Latent Gaussian dimensionality.
            key (`PRNGKeyArray`): JAX random key.
        """
        key1, key2, key3 = jr.split(key, 3)
        self.layers = nn.Sequential(
            [
                nn.Linear(1 * 64 * 64, 256, key=key1),
                nn.Lambda(jax.nn.relu),
                nn.Linear(256, 128, key=key2),
                nn.Lambda(jax.nn.relu),
                nn.Linear(128, latent_size * 2, key=key3),
            ]
        )

    def __call__(self, x: Array):
        """Forward the input through the encoder.

        Args:
            x (`Array`): Input array.

        Returns:
            Encoded features of shape `(latent_size * 2,)`.
        """
        return self.layers(x.ravel())


class MLPDecoder(eqx.Module):
    """MLP Decoder for 64x64 images."""

    layers: nn.Sequential

    def __init__(self, latent_size: int, *, key: PRNGKeyArray):
        """Initializes the MLP decoder.

        Args:
            latent_size (`int`): Latent Gaussian dimensionality.
            key (`PRNGKeyArray`): JAX random key.
        """
        key1, key2, key3 = jr.split(key, 3)
        self.layers = nn.Sequential(
            [
                nn.Linear(latent_size, 128, key=key1),
                nn.Lambda(jax.nn.relu),
                nn.Linear(128, 256, key=key2),
                nn.Lambda(jax.nn.relu),
                nn.Linear(256, 1 * 64 * 64, key=key3),
            ]
        )

    def __call__(self, x: Array):
        """Forward the input through the decoder.

        Args:
            x (`Array`): Input array.

        Returns:
            Decoded image of shape `(1, 64, 64)`.
        """
        return self.layers(x).reshape(1, 64, 64)
