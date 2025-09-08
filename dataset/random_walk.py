from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_dataloader as jdl
from jax import lax
from jaxtyping import Array, PRNGKeyArray

SIZE = 64
STEP = 8
ACTION_SPACE = jnp.eye(N=4)


def obs(state: Array) -> Array:
    """Render the pixelated observation given the state.

    Args:
        state (`Array`): State `(x, y)`.

    Returns:
        Pixelated observation of shape `specs.obs_size`.
    """
    return lax.dynamic_update_slice(
        jnp.full((SIZE, SIZE), 0.0),
        jnp.full((STEP, STEP), 1.0),
        state,
    )[jnp.newaxis]


def make_random_walks(
    n: int,
    length: int,
    *,
    key: PRNGKeyArray,
    p: float = 0.5,
    **kwds: Any,
) -> jdl.DataLoader:
    """Generate a dataset containing random walk tracjectories.

    A random walk trajectory is genrated following these rules:
    - An agent (square) lives in a 2D image and starts at the center;
    - At any time step t, the agent can move w/a/s/d;
    - The consequence of each action is stochastic.
      Specifically, facing the direction of the given action, the agent may
        - move forward 1 step with probability `p`;
        - move forward 2 steps with probability `1 - p`;
    - The agent will freeze if attempting to move beyond the boundaries.

    Args:
        n (`int`): The number of trajectories (> 0).
        length (`int`): The length of each trajectory (> 1).
        key (`PRNGKeyArray`): JAX random key.
        p (`float`): Bernoulli transition probability. Default to 0.5;
        **kwds (`Any`): Extra keyword arguments for `jdl.DataLoader`.

    Returns:
        A dataset containing `n * (length - 1)` transitions, where each transition is a
        3-tuple `(state, action, next_state)`. Each state is an 64x64 image and
        each action is a one-hot vector.
    """
    key1, key2 = jr.split(key)
    # generate actions
    n_actions = len(ACTION_SPACE)
    actions = ACTION_SPACE[jr.randint(key1, (n, length), 0, n_actions)]
    # generate observations
    moves = jnp.array([[STEP, 0], [-STEP, 0], [0, STEP], [0, -STEP]])
    noise = jr.categorical(key2, jnp.array([p, 1 - p]), shape=(n, length, 1))
    _, images = jax.vmap(lax.scan, in_axes=(None, None, 0))(
        lambda state, move: ((state + move).clip(0, SIZE - STEP), obs(state)),
        jnp.array([SIZE // 2, SIZE // 2]),
        moves[actions.argmax(axis=-1)] * (noise + 1),
    )
    # return
    states = images[:, :-1].reshape(-1, *images.shape[2:])
    actions = actions[:, :-1].reshape(-1, *actions.shape[2:])
    next_states = images[:, 1:].reshape(-1, *images.shape[2:])
    dataset = jdl.ArrayDataset(states, actions, next_states, asnumpy=False)
    return jdl.DataLoader(dataset, backend='jax', **kwds)
