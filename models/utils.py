from collections.abc import Callable, Sequence
from itertools import pairwise

import jax
import jax.random as jr
from equinox import nn
from jaxtyping import Array, PRNGKeyArray

from .distributions import Gaussian


class MLP(nn.Sequential):
    """Multi-layer Perceptron."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int] = (),
        *,
        key: PRNGKeyArray,
        act: Callable[[Array], Array] = jax.nn.relu,
    ) -> None:
        """Initialize an multi-layer perceptron.

        NOTE that the final activation will be dropped.

        Args:
            in_size (`int`): Input size.
            out_size (`int`): Output size.
            hidden_sizes (`Sequence[int]`): Hidden layer sizes.
            key (`PRNGKeyArray`): JAX random key.
            act (`Callable[[Array], Array]`, optional):
                The activation function. Default to `jax.nn.relu`.
        """
        layer_sizes = [in_size, *hidden_sizes, out_size]
        layers = []
        keys = iter(jr.split(key, len(layer_sizes) - 1))
        for in_size, out_size in pairwise(layer_sizes):
            layers.append(nn.Linear(in_size, out_size, key=next(keys)))
            layers.append(nn.Lambda(act))
        super().__init__(layers[:-1])  # drop last act

    def __call__(self, x: Array) -> Array:
        """Forward the flattened input through the layers.

        Args:
            x (`Array`): Input array.

        Returns:
            Outupt array, shape determined by the last layer.
        """
        return super().__call__(x.ravel())


@jax.custom_jvp
def ngd(dist: Gaussian) -> Gaussian:
    return dist


@ngd.defjvp
def _(primals: tuple[Gaussian], tangents: tuple[Gaussian]) -> tuple[Gaussian, Gaussian]:
    dist, grads = *primals, *tangents
    return dist, Gaussian(
        mean=grads.mean * (dist.std**2),
        std=grads.std / 2.0 * (dist.std**2),
    )
