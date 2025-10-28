from typing import Any

import jax.numpy as jnp
import jax.random as jr
import jax_dataloader as jdl
from jaxtyping import PRNGKeyArray

lims = (0.0, 4 * jnp.pi)
f = (lambda x: jnp.sin(x), lambda x: jnp.sin(x + jnp.pi))


def make_sinusoid_waves(n: int, *, key: PRNGKeyArray, **kwds: Any) -> jdl.DataLoader:
    """Generate a dataset containing noisy sinusoid waves.

    Args:
        n (`int`): The number of data points.
        key (`PRNGKeyArray`): JAX random key.
        **kwds (`Any`): Extra keyword arguments for `jdl.DataLoader`.

    Returns:
        A dataset containing `2 * n` data points, where each data point is a pair of
        either `(x, sin(x))` or `(x, sin(x + pi))`, and the sinusoids are pertubed by
        a noise from `N(0, 0.1)`.
    """
    key, key1, key2, key3 = jr.split(key, 4)
    # generate data
    x = jr.uniform(key1, (n, 1), minval=lims[0], maxval=lims[1])
    y1 = f[0](x) + jr.normal(key2, (n, 1)) * 0.01
    y2 = f[1](x) + jr.normal(key3, (n, 1)) * 0.01
    # return
    dataset = jdl.ArrayDataset(
        jnp.concat([x, x]),
        jnp.concat([y1, y2]),
        asnumpy=False,
    )
    return jdl.DataLoader(dataset, backend='jax', generator=key, **kwds)
