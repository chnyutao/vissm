from typing import Any

import jax.numpy as jnp
import jax.random as jr
import jax_dataloader as jdl
from jaxtyping import PRNGKeyArray


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
    y = jr.uniform(key1, (n, 1))
    x = y + 0.3 * jnp.sin(2 * jnp.pi * y)
    x = x + 0.05 * jr.normal(key2, x.shape)
    # return
    dataset = jdl.ArrayDataset(x, y, asnumpy=False)
    return jdl.DataLoader(dataset, backend='jax', **kwds)
