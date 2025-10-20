from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from models.distributions import Categorical, Gaussian, GaussianMixture
from models.utils import MLP, CatNet, GaussNet


class GaussTr(eqx.Module):
    """Gaussian Transition Function."""

    net: GaussNet

    def __init__(
        self,
        state_size: int,
        action_size: int,
        *,
        key: PRNGKeyArray,
        **kwds: Any,
    ) -> None:
        """
        Initialize the gaussian transition function.

        Args:
            state_size (`int`): Size of latent state `z`.
            action_size (`int`): Size of action `a`.
            key (`PRNGKeyArray`): JAX random key.
            **kwds (`Any`, optional): Extra keyword arguments for `MLP`.
        """
        self.net = GaussNet(
            MLP(
                in_size=state_size + action_size,
                out_size=state_size * 2,
                hidden_sizes=(state_size * 2,) * 2,
                key=key,
                **kwds,
            )
        )

    def __call__(self, z: Array, a: Array) -> Gaussian:
        """Compute the next state distribution p(z'|z,a).

        Args:
            z (`Array`): The latent state z.
            a (`Array`): The action a.

        Returns:
            The next state distribution p(z'|z,a).
        """
        return self.net(jnp.concat([z, a]))


class MixtureTr(eqx.Module):
    """Gaussian Mixture Transition Function."""

    cat: CatNet
    gauss: GaussNet

    def __init__(
        self,
        state_size: int,
        action_size: int,
        k: int,
        *,
        key: PRNGKeyArray,
        **kwds: Any,
    ) -> None:
        """
        Initialize the gaussian mixture transition function.

        Args:
            state_size (`int`): Size of latent state `z`.
            action_size (`int`): Size of action `a`.
            k (`int`): Number of mixture components.
            key (`PRNGKeyArray`): JAX random key.
            **kwds (`Any`, optional): Extra keyword arguments for `MLP`.
        """
        self.cat = CatNet(
            MLP(
                in_size=state_size + action_size,
                out_size=k,
                hidden_sizes=(state_size * 2,) * 2,
                key=key,
                **kwds,
            )
        )
        self.gauss = GaussNet(
            MLP(
                in_size=state_size + action_size + k,
                out_size=state_size * 2,
                hidden_sizes=(state_size * 2,) * 2,
                key=key,
                **kwds,
            )
        )

    def __call__(self, z: Array, a: Array) -> GaussianMixture:
        """Compute the next state distribution p(z'|z,a).

        Args:
            z (`Array`): The latent state z.
            a (`Array`): The action a.

        Returns:
            The next state distribution p(z'|z,a).
        """
        weight = self.weight(z, a)
        components = jax.vmap(self.component, in_axes=(None, None, 0))(
            z, a, jnp.diag(jnp.ones_like(weight.logits))
        )
        return GaussianMixture(weight, components)

    def weight(self, z: Array, a: Array) -> Categorical:
        """Compute the mixture weight p(y'|z,a) of the next state distribution.

        Args:
            z (`Array`): The latent state z.
            a (`Array`): The action a.

        Returns:
            The mixture weight p(y'|z,a).
        """
        return self.cat(jnp.concat([z, a]))

    def component(self, z: Array, a: Array, y: Array) -> Gaussian:
        """Compute a component distribution p(z'|y',z,a) of the next state
        distribution, where the component is specified by y'.

        Args:
            z (`Array`): The latent state z.
            a (`Array`): The action a.
            y (`Array`): An one-hot vector

        Returns:
            The Gaussian component distribution p(z'|y',z,a).
        """
        return self.gauss(jnp.concat([z, a, y]))


Tr = GaussTr | MixtureTr
