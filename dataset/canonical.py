from functools import cache
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import jax_dataloader as jdl
from jaxtyping import Array, PRNGKeyArray


@cache
def make_data(n: int) -> tuple[Array, Array]:
    """Generate data points from an inverse sinusoid wave.

    Args:
        n (`int`): Number of data points.

    Returns:
        The generated `n` data points of `(y + 0.3 * sin(2 * pi * y), y)`.
    """
    y = jnp.linspace(0, 1, n)[:, jnp.newaxis]
    x = y + 0.3 * jnp.sin(2 * jnp.pi * y)
    return x, y


def make_canonical(n: int, *, key: PRNGKeyArray, **kwds: Any) -> jdl.DataLoader:
    """Generate the canonical dataset used by the original mixture density network.

    Args:
        n (`int`): Number of data points.
        key (`PRNGKeyArray`): JAX random key.
        **kwds: Extra keyword args forwarded to `jdl.DataLoader`.

    Returns:
        A dataset containing `n` data points, where each data point is a pair of
        `(x, y)` with `x = y + 0.3 * sin(2 * pi * y)` with noise.
    """
    key1, key2 = jr.split(key)
    # generate data
    x, y = make_data(n)
    x = x + jr.uniform(key1, x.shape, minval=-0.1, maxval=0.1)
    # return
    dataset = jdl.ArrayDataset(x, y, asnumpy=False)
    return jdl.DataLoader(dataset, backend='jax', generator=key2, **kwds)
