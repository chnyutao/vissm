from collections.abc import Callable

import equinox as eqx
import jax
import jax.random as jr
import optax
from jaxtyping import Array, PRNGKeyArray

from config import Config
from models.ssm import SSM, GaussSSM, MixtureSSM
from models.transition import GaussTr, MixtureTr, Tr
from models.utils import MLP
from models.vae import VAE, GaussVAE


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
    match config.model.prior:
        case 'gaussian':
            tr = GaussTr(state_size=config.model.latent_size, action_size=4, **kwds)
        case 'mixture':
            tr = MixtureTr(
                state_size=config.model.latent_size,
                action_size=4,
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
    match config.model.posterior:
        case 'gaussian':
            key1, key2 = jr.split(key)
            vae = GaussVAE(
                MLP(1 * 64 * 64, latent_size * 2, (256, 128), key=key1, act=act),
                MLP(latent_size, 1 * 64 * 64, (128, 256), key=key2, act=act),
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
        model (`SSM`): The current model.
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
