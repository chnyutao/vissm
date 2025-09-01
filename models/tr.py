from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .distributions import Gaussian
from .utils import GaussNet


class GaussTr(eqx.Module):
    """Gaussian Transition Function."""

    net: GaussNet

    def __init__(self, net: Callable[[Array], Array]):
        """
        Initialize the transition function.

        Args:
            net (`Callable[[Array], Array]`): The neural network.
        """
        self.net = GaussNet(net)

    def __call__(self, z: Array, a: Array) -> Gaussian:
        """Compute the next state distribution p(z'|z,a).

        Args:
            z (`Array`): The latent state z.
            a (`Array`): The action a.

        Returns:
            The next state distribution p(z'|z,a).
        """
        return self.net(jnp.concat([z, a]))
