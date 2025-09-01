import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray


class Categorical(eqx.Module):
    """Categorical Distribution."""

    log_p: Array

    def sample(self, tau: float = 1e-3, *, key: PRNGKeyArray) -> Array:
        """Sample from the distribution using the reparametrization trick.

        Args:
            tau (`float`, optional): Gumbel-softmax temperature. Default to `1e-3`.
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A sample drawn from the Categorical distribution.
        """
        gumbel = jr.gumbel(key, self.log_p.shape)
        return jnp.exp(jax.nn.log_softmax((self.log_p + gumbel) / tau))

    def to(self) -> distrax.Categorical:
        """Cast to a `distrax.Categorical`.

        Returns:
            A `distrax.Categorical` distribution.
        """
        return distrax.Categorical(self.log_p)


class Gaussian(eqx.Module):
    """Gaussian Distribution (Diagonal)."""

    mean: Array
    std: Array

    def sample(self, *, key: PRNGKeyArray) -> Array:
        """Sample from the distribution using the reparametrization trick.

        Args:
            key (`PRNGKeyArray`): JAX random key.

        Returns:
            A sample drawn from the Gaussian distribution.
        """
        return self.mean + self.std * jr.normal(key, self.std.shape)

    def to(self) -> distrax.MultivariateNormalDiag:
        """Cast to a `distrax.MultivariateNormalDiag`.

        Returns:
            A `distrax.MultivariateNormalDiag` distribution.
        """
        return distrax.MultivariateNormalDiag(self.mean, self.std)
