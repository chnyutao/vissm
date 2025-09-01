from collections.abc import Callable

import equinox as eqx
import jax
import jax.random as jr
import optax
from jaxtyping import Array, PRNGKeyArray

from config import Config
from models.ssm import SSM
from models.tr import GaussTr
from models.utils import MLP
from models.vae import VAE


def make_tr(config: Config, *, key: PRNGKeyArray) -> GaussTr:
    """
    Initialize a transition function specified by the configuration.

    Args:
        config (`Config`): Configuration.
        key (`PRNGKeyArray`): JAX random key.

    Returns:
        A transition function.
    """
    match config.model.tr:
        case 'gaussian':
            tr = GaussTr(
                MLP(
                    config.model.latent_size + 4,
                    config.model.latent_size * 2,
                    (config.model.latent_size * 2,) * 2,
                    key=key,
                    act=getattr(jax.nn, config.model.act),
                )
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
    ekey, dkey = jr.split(key)
    act = getattr(jax.nn, config.model.act)
    latent_size = config.model.latent_size
    match config.model.posterior:
        case 'gaussian':
            vae = VAE(
                MLP(
                    in_size=1 * 64 * 64,
                    out_size=latent_size * 2,
                    hidden_sizes=(256, 128),
                    key=ekey,
                    act=act,
                ),
                MLP(
                    in_size=latent_size,
                    out_size=1 * 64 * 64,
                    hidden_sizes=(128, 256),
                    key=dkey,
                    act=act,
                ),
            )
    return vae


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
    key: PRNGKeyArray,
    callback: Callable[..., None] = lambda _: None,
) -> None:
    """Perform a single evaluation step between or after training epochs.

    Args:
        model (`SSM`): The current model.
        key (`PRNGKeyArray`): JAX random key.
        callback (`Callable[..., None]`, optional):
            Callback function for processing metrics. Default to `lambda _: None`.
    """
    metrics = {}
    jax.debug.callback(callback, metrics)
