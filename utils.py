from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax_dataloader as jdl
import optax
import wandb
from jaxtyping import Array, PRNGKeyArray, PyTree
from matplotlib import pyplot as plt

import dataset
import plots
from config import Config
from dataset import make_bimodal, make_canonical, make_sinusoid_waves
from models import GaussianMixtureModel, GaussianNetwork, MixtureDensityNetwork
from models.distributions import Distribution, Gaussian, GaussianMixture
from models.utils import MLP


def make_dataset(
    config: Config,
    *,
    key: PRNGKeyArray,
    train: bool = True,
) -> jdl.DataLoader:
    """Initialize a dataset specified by the configuration.

    Args:
        config (`Config`): Configuration.
        key (`PRNGKeyArray`): JAX random key.
        train (`bool`, optional): Train or eval set. Default to `True`.

    Returns:
        A `jax_dataloader.DataLoader`.
    """
    match config.dataset.name:
        case 'bimodal':
            dataset = make_bimodal(
                config.dataset.n,
                key=key,
                batch_size=config.dataset.batch_size,
                shuffle=config.dataset.shuffle,
            )
        case 'canonical':
            dataset = make_canonical(
                config.dataset.n,
                key=key,
                batch_size=config.dataset.batch_size,
                shuffle=config.dataset.shuffle,
            )
        case 'sinusoid':
            dataset = make_sinusoid_waves(
                config.dataset.n,
                key=key,
                batch_size=config.dataset.batch_size,
                shuffle=config.dataset.shuffle,
            )
    return dataset


def make_model(config: Config, *, key: PRNGKeyArray) -> PyTree:
    """Initialize a model specified by the configuration.

    Args:
        config (`Config`): Configuration.
        key (`PRNGKeyArray`): JAX random key.

    Returns:
        A PyTree representation of the parametrized model .
    """
    match config.dataset.name:
        case 'bimodal':
            model = GaussianMixtureModel(
                config.model.k,
                config.model.n,
                config.model.loss,
                key=key,
            )
        case 'canonical' | 'sinusoid':
            act = getattr(jax.nn, config.model.act)
            hidden_sizes = config.model.hidden_sizes
            k = config.model.k
            n = config.model.n
            if config.model.density == 'gaussian':
                model = GaussianNetwork(MLP(1, n * 2, hidden_sizes, key=key, act=act))
            elif config.model.density == 'mixture':
                key1, key2 = jr.split(key)
                model = MixtureDensityNetwork(
                    cat=MLP(1, k, hidden_sizes, key=key1, act=act),
                    gauss=MLP(1, k * n * 2, hidden_sizes, key=key2, act=act),
                    loss=config.model.loss,
                )
    return model


def make_opt(
    config: Config,
    model: PyTree,
) -> tuple[optax.GradientTransformation, optax.OptState]:
    """Initialize an optimizer specified by the configuration.

    Args:
        config (`Config`): Configuration.

    Returns:
        An `optax.GradientTransformation`.
    """
    match config.opt.name:
        case 'adam':
            opt = optax.adam(config.opt.lr)
        case 'rmsprop':
            opt = optax.rmsprop(config.opt.lr)
        case 'sgd':
            opt = optax.sgd(config.opt.lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    return opt, opt_state


@eqx.filter_jit
def train_step(
    model: PyTree,
    batch: tuple[Array, ...],
    opt_state: optax.OptState,
    *,
    callback: Callable[..., None] = lambda _: None,
    key: PRNGKeyArray,
    opt: optax.GradientTransformation,
):
    """Perform a single jitted training step.

    Args:
        model (`PyTree`): The current model.
        batch (`tuple[Array, ...]`): A 1-tuple containing the batched data.
        opt_state (`optax.OptState`): The current optimizer state.
        callback (`Callable[..., None]`, optional): Callback function.
            Default to `lambda _: None`.
        key (`PRNGKeyArray`): JAX random key.
        opt (`optax.GradientTransformation`): The current optimizer.

    Returns:
        A 2-tuple containing the updated model, and the updated optimizer state.
    """
    (_, metrics), grads = model.loss_fn(*batch)
    updates, opt_state = opt.update(grads, opt_state)
    jax.debug.callback(callback, metrics)
    return eqx.apply_updates(model, updates), opt_state


def eval_step(
    model: PyTree,
    *,
    callback: Callable[..., None] = lambda _: None,
    config: Config,
    eval_set: jdl.DataLoader,
    key: PRNGKeyArray,
) -> None:
    """Perform a single evaluation step.

    Args:
        model (`PyTree`): The current model.
        callback (`Callable[..., None]`, optional): Callback function.
            Default to `lambda _: None`.
        config (`Config`): The current configuration.
        eval_set (`jdl.DataLoader`): The evaluation dataset.
        key (`PRNGKeyArray`): JAX random key.
    """
    match config.dataset.name:
        case 'bimodal':
            options = (
                {'alpha': 0.4, 'color': 'darkorange', 'label': 'posterior/1'},
                {'alpha': 0.8, 'color': 'lavender', 'label': 'posterior/2'},
            )
            heatmap = plots.Heatmap().show(model, dataset.bimodal.dists, options)
            jax.debug.callback(callback, {'heatmap': wandb.Image(heatmap.fig)})
        case 'canonical':
            xs, ys = map(jnp.concat, zip(*eval_set))
            # visualization
            options = tuple({'color': f'tab:{c}'} for c in ('blue', 'orange', 'green'))
            with plots.Regression().show(model, (xs, ys), options) as plot:
                plot.ax.set_aspect('equal')
                plot.ax.set_xlim(0, 1)
                plot.ax.set_xticks([0, 1])
                plot.ax.set_ylim(0, 1)
                plot.ax.set_yticks([0, 1])
            jax.debug.callback(callback, {'canonical': wandb.Image(plot.fig)})
            # rmse
            dists = jax.vmap(model)(xs)
            jax.debug.callback(callback, {'rmse': rmse(ys, dists)})
        case 'sinusoid':
            xs, ys = map(jnp.concat, zip(*eval_set))
            # visualization
            options = ({'color': 'tab:blue'}, {'color': 'tab:orange'})
            with plots.Regression().show(model, (xs, ys), options) as plot:
                plot.ax.set_xlim(0, 4 * jnp.pi)
                plot.ax.set_xticks([0, 2 * jnp.pi, 4 * jnp.pi])
                plot.ax.set_xticklabels(['$0$', '$2\\pi$', '$4\\pi$'])
                plot.ax.set_ylim(-1.25, 1.25)
                plot.ax.set_yticks([-1, 0, 1])
                plot.ax.set_yticklabels(['$-1$', '$0$', '$1$'])
            jax.debug.callback(callback, {'sinusoid': wandb.Image(plot.fig)})
            # rmse
            dists = jax.vmap(model)(xs)
            jax.debug.callback(callback, {'rmse': rmse(ys, dists)})
    plt.close('all')


def save_model(model: PyTree, *, path: str = 'model.eqx') -> None:
    """Save the model weigths locally & on wandb.

    Args:
        model (`PyTree`): The model to be saved..
        path (`str`, optional): The local path to save the model.
            Defaults to `'model.eqx'`.
    """
    eqx.tree_serialise_leaves(path, model)
    wandb.save(path)


def rmse(y: Array, dists: Distribution) -> Array:
    """Compute the root mean square error (RMSE) between the targets and predictions.
    - `Gaussian`: RMSE is computed between the target and the mean;
    - `GaussianMixture`: RMSE is computed as the minimum among mixture comoponents.

    Args:
        y (`Array`): The target values.
        dists (`Distribution`): The predictive distributions

    Returns:
        RMSE between the targets and the predictions.
    """

    match dists:
        case Gaussian(mean, _):
            return jnp.sqrt(((y - mean) ** 2).sum(axis=-1)).mean()
        case GaussianMixture(_, Gaussian(mean, _)):
            y = y[:, jnp.newaxis]
            return jnp.sqrt(((y - mean) ** 2).sum(axis=-1).min(axis=-1)).mean()
        case _:
            raise NotImplementedError
