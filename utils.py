from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
import wandb
from distrax import MultivariateNormalDiag as MvNormal
from jaxtyping import Array, PRNGKeyArray

from dataset.random_walk import tr
from models import SSM
from models.ssm import loss_fn


@eqx.filter_jit
def train_step(
    model: SSM,
    batch: tuple[Array, Array, Array],
    opt_state: optax.OptState,
    *,
    key: PRNGKeyArray,
    opt: optax.GradientTransformation,
    callback: Callable[..., None] = lambda _: None,
) -> tuple[SSM, optax.OptState]:
    """Performs a single jitted training step.

    Args:
        model (`Model`): The current model.
        batch (`tuple[Array, Array, Array]`):
            A 3-tuple containing the batched states, actions, and next states.
        opt_state (`optax.OptState`): The current optimizer state.
        key (`PRNGKeyArray`): JAX random key.
        opt (`optax.GradientTransformation`): The current optimizer.
        callback (`Callable[..., None]`, optional):
            Callback function for processing metrics. Default to `lambda _: None`.

    Returns:
        A 2-tuple containing the updated model, the updated optimizer state.
    """
    [_, metrics], grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, batch, key=key
    )
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    jax.debug.callback(callback, metrics)
    return model, opt_state


def eval_step(
    model: SSM,
    *,
    callback: Callable[..., None] = lambda _: None,
) -> None:
    """Evaluate the model with visualizations and metrics.

    Args:
        model (`SSM`): The current model.
        callback (`Callable[..., None]`, optional):
            Callback function for processing metrics. Default to `lambda _: None`.
    """
    # generate states & actions
    s = [tr(jnp.array([i * 16, 16]), jnp.zeros([2]))[-1] for i in range(3)]
    a = jax.nn.one_hot(0, num_classes=4)
    # computing prior & posteriors
    dists = {}
    _, (z, _) = model.vae.encode(s[0], key=jr.key(0))
    mean, log_std = jnp.split(model.tr(jnp.concat([z, a])), 2)
    dists['prior'] = (mean, jnp.exp(log_std))
    _, posterior = model.vae.encode(s[1], key=jr.key(0))
    dists['posterior/1'] = posterior
    _, posterior = model.vae.encode(s[2], key=jr.key(0))
    dists['posterior/2'] = posterior
    # plotting
    colors = dict(zip(dists.keys(), ['red', 'green', 'teal']))
    for key, (mean, std) in dists.items():
        x, y = jnp.unstack(jnp.linspace(mean - 3 * std, mean + 3 * std), axis=-1)
        z = MvNormal(mean, std).prob(jnp.dstack(jnp.meshgrid(x, y)))
        plt.contour(x, y, z, colors=colors[key], levels=3)
        plt.plot([], [], color=colors[key], label=key)  # dummy label
    plt.legend()
    # callback
    metrics = {'distributions': wandb.Image(plt)}
    jax.debug.callback(callback, metrics)
