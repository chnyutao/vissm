from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.lax import stop_gradient as sg
from jaxtyping import Array, PRNGKeyArray

from .distributions import Categorical, Gaussian, GaussianMixture
from .utils import ngd

Loss = Literal['nll', 'ngem', 'sgem']


class GaussianMixtureModel(GaussianMixture):
    """Gaussian Mixture Model (GMM)."""

    loss: Loss = eqx.field(static=True)

    def __init__(self, k: int, n: int, loss: Loss, *, key: PRNGKeyArray) -> None:
        """Initialze a Gaussian mixture model randomly.

        Args:
            k (`int`): Number of mixture components.
            n (`int`): Dimensionality of each mixture component.
            loss (`Loss`): Type of loss function.
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
        if self.loss == 'ngem':
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


class GaussianNetwork(eqx.Module):
    """Gaussian Network."""

    net: Callable[[Array], Array]

    def __call__(self, x: Array) -> Gaussian:
        """Forward the input array through the Gaussian network.

        Args:
            x (`Array`): Input array.

        Returns:
            Gaussian distribution.
        """
        mean, log_std = jnp.split(self.net(x).ravel(), 2)
        return Gaussian(mean, jnp.exp(log_std))

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(self, x: Array, y: Array) -> tuple[Array, dict[str, Array]]:
        """Calculate the loss function.

        Args:
            x (`Array`): Batch of input array.
            y (`Array`): Batch of output array.

        Returns:
            The loss value and a dictionary of auxiliary information.
        """
        dists = jax.vmap(self)(x)
        loss = -dists.to().log_prob(y).mean()
        # return
        return loss, {'loss': loss}


class MixtureDensityNetwork(eqx.Module):
    """Mixture Density Network (MDN)."""

    cat: Callable[[Array], Array]
    gauss: Callable[[Array], Array]
    loss: Loss = eqx.field(static=True)

    def __init__(
        self,
        cat: Callable[[Array], Array],
        gauss: Callable[[Array], Array],
        loss: Loss,
    ) -> None:
        """Initialze a mixture density network.

        Args:
            cat (`Callable[[Array], Array]`): Categorical network.
            gauss (`Callable[[Array], Array]`): Gaussian network.
            loss (`Loss`): Type of loss function.
        """
        self.cat = cat
        self.gauss = gauss
        self.loss = loss

    def __call__(self, x: Array) -> GaussianMixture:
        """Forward the input array through the mixture density network.

        Args:
            x (`Array`): Input array.

        Returns:
            Gaussian mixture distribution.
        """
        weight = Categorical(self.cat(x))
        k = weight.logits.shape[-1]
        means, log_stds = jnp.split(self.gauss(x), 2)
        components = Gaussian(means.reshape(k, -1), jnp.exp(log_stds).reshape(k, -1))
        if self.loss == 'ngem':
            components = ngd(components)
        return GaussianMixture(weight, components)

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(self, x: Array, y: Array) -> tuple[Array, dict[str, Array]]:
        """Calculate the loss function.

        Args:
            x (`Array`): Batch of input array.
            y (`Array`): Batch of output array.

        Returns:
            The loss value and a dictionary of auxiliary information.
        """
        dists = jax.vmap(self)(x)
        match self.loss:
            case 'nll':
                loss = -dists.to().log_prob(y).mean()
            case _:
                # e-step: responsibilities
                log_weights = jax.nn.log_softmax(dists.weight.logits)
                log_components = dists.components.to().log_prob(y[:, jnp.newaxis])
                rho = sg(jax.nn.softmax(log_weights + log_components, axis=-1))
                # m-step: gradient descent
                loss = -(rho * (log_weights + log_components)).sum(axis=-1).mean()
        # return
        entropy = dists.weight.to().entropy().mean()
        return loss, {'loss': loss, 'entropy': entropy}
