from collections.abc import Callable

import equinox as eqx
from jaxtyping import Array

from .distributions import Gaussian
from .utils import GaussNet


class GaussVAE(eqx.Module):
    """Gaussian Variational Auto-Encoder."""

    encoder: GaussNet
    decoder: Callable[[Array], Array]

    def __init__(
        self,
        encoder: Callable[[Array], Array],
        decoder: Callable[[Array], Array],
    ) -> None:
        """
        Initialize a variational auto-encoder.

        Args:
            encoder (`Callable[[Array], Array]`): The encoder net.
            decoder (`Callable[[Array], Array]`): The decoder net.
        """
        self.encoder = GaussNet(encoder)
        self.decoder = decoder

    def decode(self, z: Array) -> Array:
        """Reconstruct the input x ~ p(x|z).

        Args:
            z (`Array`): Latent variable z.

        Returns:
            Parameters of p(x|z), also known as the reconstruction.
        """
        return self.decoder(z)

    def encode(self, x: Array) -> Gaussian:
        """Compute the variational posterior q(z|x).

        Args:
            x (`Array`): Input array.

        Returns:
            Parameters of the Gaussian variational posterior q(z|x).
        """
        return self.encoder(x)


VAE = GaussVAE
