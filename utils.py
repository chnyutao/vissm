import equinox as eqx
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from models import VAE
from models.vae import loss_fn

Model = VAE


@eqx.filter_jit
def train_step(
    model: Model,
    batch: tuple[Array, Array, Array],
    opt_state: optax.OptState,
    *,
    opt: optax.GradientTransformation,
    key: PRNGKeyArray,
) -> tuple[VAE, optax.OptState, PyTree]:
    """Performs a single jitted training step.

    Args:
        model (`Model`): The current model.
        batch (`tuple[Array, Array, Array]`):
            A 3-tuple containing the batched states, actions, and next states.
        opt_state (`optax.OptState`): The current optimizer state.
        opt (`optax.GradientTransformation`): The current optimizer.
        key (`PRNGKeyArray`): JAX random key.

    Returns:
        A 3-tuple containing the updated model, the updated optimizer state,
        and the training metrics.
    """
    states, actions, next_states = batch
    [_, metrics], grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, states, key=key
    )
    updates, opt_state = opt.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, metrics
