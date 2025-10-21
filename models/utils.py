from collections.abc import Callable, Sequence
from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import nn
from jaxtyping import Array, PRNGKeyArray

from .distributions import Categorical, Gaussian


class CatNet(eqx.Module):
    """Categorical Distribution Network."""

    net: Callable[[Array], Array]

    def __call__(self, x: Array) -> Categorical:
        """
        Compute the parameters of a Categorical distribution `log_p = f(x)`,
        where `f(x)` is a parametrized neural network.

        Args:
            x (`Array`): Input array.

        Returns:
            A Categorical distribution.
        """
        log_p = jax.nn.log_softmax(self.net(x).ravel())
        return Categorical(log_p)


class GaussNet(eqx.Module):
    """Gaussian Distribution Network."""

    net: Callable[[Array], Array]

    def __call__(self, x: Array) -> Gaussian:
        """
        Compute the parameters of a Gaussian distribution `(mean, std) = f(x)`,
        where `f(x)` is a parametrized neural network.

        Args:
            x (`Array`): Input array.

        Returns:
            A Gaussian distribution.
        """
        mean, log_std = jnp.split(self.net(x).ravel(), 2)
        return Gaussian(mean, jnp.exp(log_std))


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
