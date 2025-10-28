from typing import Any

import jax.numpy as jnp
import jax.random as jr
import jax_dataloader as jdl
from distrax import MixtureOfTwo
from jaxtyping import PRNGKeyArray

from models.distributions import Gaussian

dists = [
    Gaussian(jnp.ones((2,)), jnp.ones((2,)) / 10),
    Gaussian(-jnp.ones((2,)), jnp.ones((2,)) / 10),
]


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
    key, subkey = jr.split(key)
    # generate data
    mixture = MixtureOfTwo(0.5, *[dist.to() for dist in dists])
    x = mixture.sample(seed=subkey, sample_shape=(n,))
    # return
    dataset = jdl.ArrayDataset(x, asnumpy=False)
    return jdl.DataLoader(dataset, backend='jax', generator=key, **kwds)
