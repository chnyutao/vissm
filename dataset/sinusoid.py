from typing import Any

import jax.numpy as jnp
import jax.random as jr
import jax_dataloader as jdl
from jaxtyping import PRNGKeyArray


def make_sinusoid_waves(n: int, *, key: PRNGKeyArray, **kwds: Any) -> jdl.DataLoader:
    """Generate a dataset containing noisy sinusoid waves.

    Args:
        n (`int`): Number of data points.
        key (`PRNGKeyArray`): JAX random key.
        **kwds (`Any`): Extra keyword arguments for `jdl.DataLoader`.

    Returns:
        A dataset containing `2 * n` data points, where each data point is a pair of
        either `(x, sin(x))` or `(x, sin(x + pi))`, and the sinusoids are pertubed by
        a noise from `N(0, 0.1)`.
    """
    key1, key2, key3 = jr.split(key, 3)
    # generate data
    x = jr.uniform(key1, (n, 1), minval=0, maxval=4 * jnp.pi)
    y1 = jnp.sin(x)
    y2 = jnp.sin(x + jnp.pi)
    x, y = jnp.concat([x, x]), jnp.concat([y1, y2])
    y = y + 0.01 * jr.normal(key2, y.shape)
    # return
    dataset = jdl.ArrayDataset(x, y, asnumpy=False)
    return jdl.DataLoader(dataset, backend='jax', generator=key3, **kwds)
