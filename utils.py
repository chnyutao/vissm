from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import wandb
from jaxtyping import Array, PRNGKeyArray
from matplotlib import pyplot as plt

from config import Config
from dataset import random_walk
from models.ssm import SSM, GaussSSM, MixtureSSM
from models.transition import GaussTr, MixtureTr, Tr
from models.utils import MLP
from models.vae import VAE, GaussVAE
from plots import Heatmap


def make_model(config: Config, *, key: PRNGKeyArray) -> SSM:
    """Initialize a state-space model specified by the configuration.

    Args:
        config (`Config`): Configuration.
        key (`PRNGKeyArray`): JAX random key.

    Returns:
        A state-space model.
    """
    key1, key2 = jr.split(key)
    vae = make_vae(config, key=key1)
    tr = make_tr(config, key=key2)
    match (vae, tr):
        case (GaussVAE(), GaussTr()):
            ssm = GaussSSM(vae=vae, tr=tr)
        case (GaussVAE(), MixtureTr()):
            ssm = MixtureSSM(vae=vae, tr=tr)
        case _:
            raise TypeError(f'Unsupported combination {type(vae)} and {type(tr)}.')
    return ssm


def make_tr(config: Config, *, key: PRNGKeyArray) -> Tr:
    """
    Initialize a transition function specified by the configuration.

    Args:
        config (`Config`): Configuration.
        key (`PRNGKeyArray`): JAX random key.

    Returns:
        A transition function.
    """
    kwds = {'act': getattr(jax.nn, config.model.act), 'key': key}
    match config.dataset.name:
        case 'random_walk':
            action_size = len(random_walk.ACTION_SPACE)
    match config.model.prior:
        case 'gaussian':
            tr = GaussTr(
                state_size=config.model.latent_size,
                action_size=action_size,
                **kwds,
            )
        case 'mixture':
            tr = MixtureTr(
                state_size=config.model.latent_size,
                action_size=action_size,
                k=config.model.k,
                **kwds,
            )
    return tr


def make_vae(config: Config, *, key: PRNGKeyArray) -> VAE:
    """
    Initialize a variational auto-encoder specified by the configuration.

    Args:
        config (`Config`): Configuration.
        key (`PRNGKeyArray`): JAX random key.

    Returns:
        A variational auto-encoder.
    """
    act = getattr(jax.nn, config.model.act)
    latent_size = config.model.latent_size
    match config.dataset.name:
        case 'random_walk':
            input_size = jnp.array([1, random_walk.SIZE, random_walk.SIZE])
    match config.model.posterior:
        case 'gaussian':
            key1, key2 = jr.split(key)
            input_size = int(input_size.prod())
            vae = GaussVAE(
                MLP(input_size, latent_size * 2, (256, 128), key=key1, act=act),
                MLP(latent_size, input_size, (128, 256), key=key2, act=act),
            )
    return vae


@eqx.filter_jit
def train_step(
    model: SSM,
    batch: tuple[Array, Array, Array],
    opt_state: optax.OptState,
    *,
    callback: Callable[..., None] = lambda _: None,
    key: PRNGKeyArray,
    opt: optax.GradientTransformation,
) -> tuple[SSM, optax.OptState]:
    """Perform a single jitted training step.

    Args:
        model (`SSM`): The current model.
        batch (`tuple[Array, Array, Array]`):
            A 3-tuple containing the batched states, actions, and next states.
        opt_state (`optax.OptState`): The current optimizer state.
        callback (`Callable[..., None]`, optional):
            Callback function for processing metrics. Default to `lambda _: None`.
        key (`PRNGKeyArray`): JAX random key.
        opt (`optax.GradientTransformation`): The current optimizer.

    Returns:
        A 2-tuple containing the updated model, the updated optimizer state.
    """
    [_, metrics], grads = model.loss_fn(batch, key=key)
    updates, opt_state = opt.update(
        grads,
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    model = eqx.apply_updates(model, updates)
    jax.debug.callback(callback, metrics)
    return model, opt_state


def eval_step(
    model: SSM,
    *,
    callback: Callable[..., None] = lambda _: None,
    key: PRNGKeyArray,
) -> None:
    """Perform a single evaluation step between or after training epochs.

    Args:
        model (`SSM`): The current model.
        callback (`Callable[..., None]`, optional):
            Callback function for processing metrics. Default to `lambda _: None`.
        key (`PRNGKeyArray`): JAX random key.
    """
    metrics = {}
    # heatmap
    size, step = random_walk.SIZE, random_walk.STEP
    s0 = random_walk.obs(jnp.array([size // 2, size // 2]))
    a = random_walk.ACTION_SPACE[0]
    s1 = random_walk.obs(jnp.array([size // 2 + step, size // 2]))
    s2 = random_walk.obs(jnp.array([size // 2 + step * 2, size // 2]))
    heatmap = Heatmap().show(
        prior=model.tr(model.vae.encode(s0).sample(key=key), a),
        posteriors=[
            model.vae.encode(s1),
            model.vae.encode(s2),
        ],
        cfgs=[
            {'alpha': 0.4, 'color': 'darkorange', 'label': 'posterior/1'},
            {'alpha': 0.8, 'color': 'lavender', 'label': 'posterior/2'},
        ],
    )
    metrics['eval/heatmap'] = wandb.Image(heatmap.fig)
    # callback
    jax.debug.callback(callback, metrics)
    plt.close('all')
