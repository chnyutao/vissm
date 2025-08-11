import equinox as eqx
import jax.numpy as jnp
from distrax import Categorical, MixtureSameFamily, Normal
from distrax import MultivariateNormalDiag as MvNormal
from jaxtyping import Array


class Gaussian(eqx.Module):
    "Gaussian distribution parameters."

    mean: Array
    std: Array

    def density(self, lo: Array, hi: Array) -> tuple[Array, ...]:
        """Compute the probability density over a given range.

        Args:
            lo (`Array`): Lower bound of the range.
            hi (`Array`): Upper bound of the range.

        Returns:
            A `(n+1)`-tuple containing the x1, x2, ..., xn coordinates of the grid, and
            the probability density values at each point on the grid.
        """
        xy = jnp.unstack(jnp.linspace(lo, hi, num=100), axis=-1)
        z = self.to().prob(jnp.dstack(jnp.meshgrid(*xy)))
        assert isinstance(z, Array)
        return (*xy, z)

    def marginal(self, lo: float, hi: float, *, dim: int) -> tuple[Array, Array]:
        """Compute the marginal density at a given dimension over a given range.

        Args:
            lo (`float`): The lower bound of the range.
            hi (`float`): The upper bound of the range.
            dim (`int`): The dimension of the marginal.

        Returns:
            A 2-tuple containing the marginal coordinates `x`
            and the corresponding probability density values `y`.
        """
        x = jnp.arange(lo, hi, min((hi - lo) / 100, 1e-2))
        y = Normal(self.mean[dim], self.std[dim]).prob(x)
        assert isinstance(y, Array)
        return x, y

    def to(self) -> MvNormal:
        """Cast to a `distrax.MultivariateNormalDiag`.

        Returns:
            A `distrax.MultivariateNormalDiag` distribution.
        """
        return MvNormal(self.mean, self.std)


class GaussianMixture(eqx.Module):
    """Mixture of Gaussians distribution parameters."""

    logits: Array
    means: Array
    stds: Array

    def density(self, lo: Array, hi: Array) -> tuple[Array, ...]:
        """Compute the probability density over a given range.

        Args:
            lo (`Array`): Lower bound of the range.
            hi (`Array`): Upper bound of the range.

        Returns:
            A `(n+1)`-tuple containing the x1, x2, ..., xn coordinates of the grid, and
            the probability density values at each point on the grid.
        """
        xy = jnp.unstack(jnp.linspace(lo, hi, num=100), axis=-1)
        z = self.to().prob(jnp.dstack(jnp.meshgrid(*xy)))
        assert isinstance(z, Array)
        return (*xy, z)

    def marginal(self, lo: float, hi: float, *, dim: int) -> tuple[Array, Array]:
        """Compute the marginal density at a given dimension over a given range.

        Args:
            lo (`float`): The lower bound of the range.
            hi (`float`): The upper bound of the range.
            dim (`int`): The dimension of the marginal.

        Returns:
            A 2-tuple containing the marginal coordinates `x`
            and the corresponding probability density values `y`.
        """
        x = jnp.arange(lo, hi, min((hi - lo) / 100, 1e-2))
        y = MixtureSameFamily(
            mixture_distribution=Categorical(self.logits),
            components_distribution=Normal(self.means[:, dim], self.stds[:, dim]),
        ).prob(x)
        assert isinstance(y, Array)
        return x, y

    def to(self) -> MixtureSameFamily:
        """Cast to a `distrax.MixtureSameFamily`.

        Returns:
            A `distrax.MixtureSameFamily` distribution.
        """
        return MixtureSameFamily(
            mixture_distribution=Categorical(self.logits),
            components_distribution=MvNormal(self.means, self.stds),
        )


Distribution = Gaussian | GaussianMixture
