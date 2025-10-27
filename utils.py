from collections.abc import Callable

import equinox as eqx
import jax
import jax.random as jr
import jax_dataloader as jdl
import optax
import wandb
from jaxtyping import Array, PRNGKeyArray, PyTree
from matplotlib import pyplot as plt

import dataset
import plots
from config import Config
from dataset import make_bimodal, make_random_walks, make_sinusoid_waves
from models import GaussianMixtureModel, GaussianNetwork, MixtureDensityNetwork
from models.utils import MLP


def make_dataset(config: Config, *, key: PRNGKeyArray) -> jdl.DataLoader:
    """Initialize a dataset specified by the configuration.

    Args:
        config (`Config`): Configuration.
        key (`PRNGKeyArray`): JAX random key.

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
        case 'random_walk':
            dataset = make_random_walks(
                config.dataset.n,
                config.dataset.length,
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
        case 'random_walk':
            raise NotImplementedError
        case 'sinusoid':
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
    callback: Callable[..., None] = lambda x: wandb.log(x),
    config: Config,
    key: PRNGKeyArray,
) -> None:
    """Perform a single evaluation step.

    Args:
        model (`PyTree`): The current model.
        callback (`Callable[..., None]`, optional): Callback function.
            Default to `lambda _: None`.
        config (`Config`): The current configuration.
        key (`PRNGKeyArray`): JAX random key.
    """
    if config.dataset.name == 'bimodal':
        options = (
            {'alpha': 0.4, 'color': 'darkorange', 'label': 'posterior/1'},
            {'alpha': 0.8, 'color': 'lavender', 'label': 'posterior/2'},
        )
        heatmap = plots.Heatmap().show(model, dataset.bimodal.dists, options)
        jax.debug.callback(callback, {'heatmap': wandb.Image(heatmap.fig)})
    elif config.dataset.name == 'sinusoid':
        options = ({'color': 'tab:blue'}, {'color': 'tab:orange'})
        sinusoid = plots.Sinusoid().show(model, options)
        jax.debug.callback(callback, {'sinusoid': wandb.Image(sinusoid.fig)})
    plt.close('all')


def save_model(model: PyTree, *, path: str = 'model.eqx') -> None:
    """
    Save the model weigths locally & on wandb.

    Args:
        model (`PyTree`): The model to be saved..
        path (`str`, optional): The local path to save the model.
            Defaults to `'model.eqx'`.
    """
    eqx.tree_serialise_leaves(path, model)
    wandb.save(path)
