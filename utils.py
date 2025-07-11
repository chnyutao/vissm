from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import wandb
from distrax import MultivariateNormalDiag as MvNormal
from jaxtyping import Array, PRNGKeyArray

from dataset.random_walk import tr
from models import GMVAE, SSM, VAE
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
    s = [tr(jnp.array([i * 16, 16]), jnp.zeros([2]))[-1] for i in range(3)]
    a = jax.nn.one_hot(0, num_classes=4)
    # computing prior & posteriors
    dists = {}
    z, _ = model.vae.encode(s[0], key=key)
    dists['prior'] = model.vae.split(model.tr(jnp.concat([z, a])))
    dists['posterior/1'] = model.vae.split(model.vae.encoder(s[1]))
    dists['posterior/2'] = model.vae.split(model.vae.encoder(s[2]))
    # plotting
    colors = dict(zip(dists.keys(), ['darkorange', 'limegreen', 'teal']))
    for key, dist in dists.items():
        if isinstance(model.vae, GMVAE):
            logits, means, stds = dist['logits'], dist['means'], dist['stds']
            for i in range(means.shape[0]) if key == 'prior' else [logits.argmax()]:
                plot_gaussian(means[i], stds[i], c=colors[key])
                label = f'{key} (p={jnp.exp(logits[i]):.2f})'
                plt.plot([], [], color=colors[key], label=label)  # dummy label
        elif isinstance(model.vae, VAE):
            mean, std = dist.values()
            plot_gaussian(mean, std, c=colors[key])
            plt.plot([], [], color=colors[key], label=key)  # dummy label
    plt.legend()
    # callback
    metrics = {'distributions': wandb.Image(plt)}
    jax.debug.callback(callback, metrics)


def plot_gaussian(mean: Array, std: Array, *, c: str) -> None:
    """Plot a Gaussian contour.

    Args:
        mean (`Array`): The mean of the Gaussian distribution.
        std (`Array`): The standard deviation of the Gaussian distribution.
        c (`str`): The color of the contour plot.
    """
    x, y = jnp.unstack(jnp.linspace(mean - 3 * std, mean + 3 * std), axis=-1)
    z = MvNormal(mean, std).prob(jnp.dstack(jnp.meshgrid(x, y)))
    plt.contour(x, y, z, colors=c, levels=3)
