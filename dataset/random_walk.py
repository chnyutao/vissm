from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_dataloader as jdl
from jax import lax
from jaxtyping import Array, PRNGKeyArray

SIZE = 64
STEP = 16


def tr(state: Array, action: Array) -> tuple[Array, Array]:
    """Compute the next state and current observation.

    Args:
        state (`Array`): Coordinates `(x, y)`.
        action (`Array`): Movement `(dx, dy)`.

    Returns:
        A 2-tuple containing the next state and current observation.
    """
    obs = lax.dynamic_update_slice(
        jnp.full((SIZE, SIZE), 0.0),
        jnp.full((STEP, STEP), 1.0),
        state,
    )[jnp.newaxis]
    next_state = (state + action).clip(0, SIZE - STEP)
    return next_state, obs


def make_random_walks(
    n: int,
    length: int,
    *,
    key: PRNGKeyArray,
    **kwds: Any,
) -> jdl.DataLoader:
    """Generate a dataset containing random walk tracjectories.

    A random walk trajectory is genrated following these rules:
    - An agent (square) lives in a 2D image and starts at the center;
    - At any discrete time step t, the agent can move w/a/s/d;
    - The consequence of each action is stochastic.
      Specifically, facing the direction of the given action, the agent may
        - move forward 1 units with probability `p`;
        - move forward 2 units with probability `1 - p`.

    Args:
        n (`int`): The number of trajectories (> 0).
        length (`int`): The length of each trajectory (> 1).
        key (`PRNGKeyArray`): JAX random key.
        **kwds (`Any`): Extra keyword arguments for `jdl.DataLoader`.

    Returns:
        A dataset containing `n * (length - 1)` transitions, where each transition is a
        3-tuple `(state, action, next_state)`. Each state is an 64x64 image and
        each action is a one-hot vector.
    """
    key1, key2 = jr.split(key)
    # generate actions
    A = jnp.array([[STEP, 0], [-STEP, 0], [0, STEP], [0, -STEP]])  # action space
    actions = jax.nn.one_hot(jr.randint(key1, (n, length), *(0, 4)), num_classes=4)
    # generate observations
    trajs = A[actions.argmax(axis=-1)]
    trajs *= jr.categorical(key2, jnp.array([0.5, 0.5]), shape=(n, length, 1)) + 1
    _, obs = jax.vmap(lax.scan, in_axes=(None, None, 0))(
        tr, jnp.full((2,), SIZE // 2), trajs
    )
    # return
    state = obs[:, :-1].reshape(-1, *obs.shape[2:])
    actions = actions[:, :-1].reshape(-1, *actions.shape[2:])
    next_states = obs[:, 1:].reshape(-1, *obs.shape[2:])
    dataset = jdl.ArrayDataset(state, actions, next_states, asnumpy=False)
    return jdl.DataLoader(dataset, backend='jax', **kwds)
