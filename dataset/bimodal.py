from typing import Any

import jax.numpy as jnp
import jax_dataloader as jdl
from distrax import MixtureOfTwo, Normal
from jaxtyping import PRNGKeyArray


def make_bimodal(n: int, *, key: PRNGKeyArray, **kwds: Any) -> jdl.DataLoader:
    """Generate a dataset containing samples from two 2D normal distributions.

    Args:
        n (`int`): The number of data points.
        key (`PRNGKeyArray`): JAX random key.
        **kwds (`Any`): Extra keyword arguments for `jdl.DataLoader`.

    Returns:
        A dataset containing `n` samples, each of which is randomly sampled from
        one of the two normal distributions.
    """
    # generate data
    dists = MixtureOfTwo(
        0.5,
        Normal(jnp.ones((2,)), jnp.ones((2,)) / 10),
        Normal(-jnp.ones((2,)), jnp.ones((2,)) / 10),
    )
    x = dists.sample(seed=key, sample_shape=(n,))
    # return
    dataset = jdl.ArrayDataset(x, asnumpy=False)
    return jdl.DataLoader(dataset, backend='jax', **kwds)
