from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import wandb
from jaxtyping import Array, PRNGKeyArray

import plots
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
    [_, metrics], grads = loss_fn(model, batch, key=key)
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    jax.debug.callback(callback, metrics)
    return model, opt_state


def eval_step(
    model: SSM,
    *,
    key: PRNGKeyArray,
    callback: Callable[..., None] = lambda _: None,
) -> None:
    """Evaluate the model with visualizations and metrics.

    Args:
        model (`SSM`): The current model.
        key (`PRNGKeyArray`): JAX random key.
        callback (`Callable[..., None]`, optional):
            Callback function for processing metrics. Default to `lambda _: None`.
    """
    # generate states & actions
    # given the current state s[0] and action a,
    # both s[1] and s[2] are possible next states.
    s = [tr(jnp.array([i * 16, 16]), jnp.zeros([2]))[-1] for i in range(3)]
    a = jax.nn.one_hot(0, num_classes=4)
    # computing prior & posteriors
    dists = {}
    z, _ = model.vae.encode(s[0], key=key)
    dists['prior'] = model.vae.split(model.tr(jnp.concat([z, a])))
    dists['posterior/1'] = model.vae.split(model.vae.encoder(s[1]))
    dists['posterior/2'] = model.vae.split(model.vae.encoder(s[2]))
    # heatmap
    plt.clf()
    fig1, axes = plots.make_distribution_map()
    plots.heatmap(fig1, axes, dists['prior'])
    for label, kwds in [
        ('posterior/1', {'alpha': 0.4, 'color': 'darkorange'}),
        ('posterior/2', {'alpha': 0.8, 'color': 'lavender'}),
        ('prior', {'alpha': 0.2, 'color': 'black', 'hatch': '///'}),
    ]:
        plots.marginal(fig1, axes, dists[label], label=label, **kwds)
    for label, kwds in [
        ('posterior/1', {'c': 'darkorange'}),
        ('posterior/2', {'c': 'lavender'}),
    ]:
        plots.mean(fig1, axes, dists[label], **kwds)
    # bars
    fig2, axes = plots.make_distribution_bars()
    for label, kwds in [
        ('posterior/1', {'alpha': 0.4, 'color': 'darkorange'}),
        ('posterior/2', {'alpha': 0.8, 'color': 'lavender'}),
        ('prior', {'alpha': 0.2, 'color': 'black', 'hatch': '///'}),
    ]:
        plots.bars(fig2, axes, dists[label], label=label, **kwds)
    # callback
    metrics = {
        'eval/heatmap': wandb.Image(fig1),
        'eval/bars': wandb.Image(fig2),
    }
    jax.debug.callback(callback, metrics)
    plt.close(fig1)
    plt.close(fig2)
