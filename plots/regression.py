from collections.abc import Callable, Sequence
from typing import Any, Self

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from models.distributions import Distribution, Gaussian, GaussianMixture


class Regression:
    """Regression curve fitting plot."""

    fig: Figure
    ax: Axes
    domain: Array

    def __init__(self, *, figsize: tuple[int, int] = (4, 3)) -> None:
        """Initialize a regression plot.

        Args:
            figsize (`tuple[int, int]`, optional): Figure width and height.
        """
        self.fig = plt.figure(figsize=figsize)
        self.fig.set_layout_engine('constrained')
        self.ax = self.fig.gca()
        self.ax.set_xlabel('$x$')
        self.ax.set_ylabel('$y$')

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns:
            The current plot instance.
        """
        return self

    def __exit__(self, *_: Any) -> None:
        """Exit the context manager."""
        pass

    def show(
        self,
        model: Callable[[Array], Distribution],
        data: tuple[Array, Array],
        options: Sequence[dict[str, Any]],
    ) -> Self:
        """Display the ground truth data points and predictive distributions.

        Args:
            model (`Callable[[Array], Distribution]`): The trained predictive model.
            data (`tuple[Array, Array]`): The grouth-truth data points `(x, y)`.
            options (`Sequence[dict[str, Any]]`): Options for predictive distributions.

        Returns:
            The current instance `self`, allowing chaining methods.
        """
        x, y = data
        # ground truth
        self.ax.scatter(x, y, alpha=0.2, c='gray', s=1)
        # predictive distribution
        self.domain = jnp.arange(x.min(), x.max(), 1e-2)
        distributions = jax.vmap(model)(self.domain)
        match distributions:
            case Gaussian(mean, std):
                self._interval(mean, std, **options[0])
            case GaussianMixture(_, Gaussian(mean, std)):
                k = mean.shape[1]
                for i in range(k):
                    option = options[i % len(options)]
                    self._interval(mean[:, i], std[:, i], **option)
            case _:
                raise NotImplementedError
        return self

    def _interval(self, mean: Array, std: Array, **kwds: Any) -> None:
        """Display the 1-sigma interval of Gaussian distributions.

        Args:
            mean (`Array`): Mean of the normal distribution.
            std (`Array): Standard deviation of the normal distribution.
            **kwds (`Any`): Extra keyword arguments fo `ax.plot` and `ax.fill_between`.
        """
        self.ax.plot(self.domain, mean, **kwds)
        kwds.setdefault('alpha', 0.2)
        self.ax.fill_between(
            self.domain,
            (mean - std).ravel(),
            (mean + std).ravel(),
            **kwds,
        )
