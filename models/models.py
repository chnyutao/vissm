from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.lax import stop_gradient as sg
from jaxtyping import Array, PRNGKeyArray

from .distributions import Categorical, Gaussian, GaussianMixture
from .utils import ngd

Loss = Literal['mle', 'ngem', 'sgem']


class GaussianMixtureModel(GaussianMixture):
    """Gaussian Mixture Model (GMM)."""

    loss: Loss = eqx.field(static=True)

    def __init__(self, k: int, n: int, loss: Loss, *, key: PRNGKeyArray) -> None:
        """Initialze a Gaussian mixture model randomly.

        Args:
            k (`int`): Number of mixture components.
            n (`int`): Dimensionality of each mixture component.
            key (`PRNGKeyArray`): JAX random key.
        """
        key1, key2, key3 = jr.split(key, 3)
        super().__init__(
            weight=Categorical(jr.normal(key1, (k,))),
            components=Gaussian(
                mean=jr.normal(key2, (k, n)),
                std=jr.uniform(key3, (k, n)),
            ),
        )
        self.loss = loss

    def __call__(self) -> GaussianMixture:
        """Wrap the Gaussian mixture model with natural gradient.

        Returns:
            Gaussian mixture distribution.
        """
        match self.loss:
            case 'ngem':
                return GaussianMixture(self.weight, ngd(self.components))
        return self

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(self, x: Array) -> tuple[Array, dict[str, Array]]:
        """Calculate the loss function.

        Args:
            x (`Array`): Batch of data.

        Returns:
            The loss value and a dictionary of auxiliary information.
        """
        match self.loss:
            case 'mle':
                loss = -self.to().log_prob(x).mean()
            case _:
                # e-step: responsibilities
                log_weights = jax.nn.log_softmax(self.weight.logits)  # (k,)
                log_components = self().components.to().log_prob(x[:, jnp.newaxis])
                rho = sg(jax.nn.softmax(log_weights + log_components, axis=-1))
                # m-step: gradient descent
                loss = -(rho * (log_weights + log_components)).sum(axis=-1).mean()
        # return
        entropy = self.weight.to().entropy().mean()
        return loss, {'loss': loss, 'entropy': entropy}
