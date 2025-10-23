from collections.abc import Callable, Sequence
from typing import Any, Self

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import dataset
from models.distributions import Distribution, Gaussian, GaussianMixture

x = jnp.arange(*dataset.sinusoid.lims, step=1e-2)


class Sinusoid:
    """Sinusoid fitting curve plot."""

    fig: Figure
    axes: Axes

    def __init__(self) -> None:
        """Initialize a sinusoid plot."""
        self.fig = plt.figure(figsize=(4, 3))
        self.fig.set_layout_engine('constrained')
        self.axes = self.fig.gca()
        self.axes.set_xlabel('$x$')
        self.axes.set_xlim(*dataset.sinusoid.lims)
        self.axes.set_xticks(jnp.linspace(*dataset.sinusoid.lims, 5))
        self.axes.set_xticklabels(['$0$', '$\\pi$', '$2\\pi$', '$3\\pi$', '$4\\pi$'])
        self.axes.set_ylabel('$y$')
        self.axes.set_ylim(-1.25, 1.25)
        self.axes.set_yticks(jnp.linspace(-1, 1, 5))
        self.axes.set_yticklabels(['$-1$', '', '$0$', '', '$1$'])

    def show(
        self,
        model: Callable[[Array], Distribution],
        options: Sequence[dict[str, Any]],
    ) -> Self:
        """Display the target sinusoid waves and predictive distributions.

        Args:
            model (`Callable[[Array], Distribution]`): The trained predictive model.
            options (`Sequence[dict[str, Any]]`): Options for predictive distributions.

        Returns:
            The current instance `self`, allowing chaining methods.
        """
        y1 = dataset.sinusoid.f[0](x)
        y2 = dataset.sinusoid.f[1](x)
        # predicted curves
        dists = jax.vmap(model)(x)
        self._predictions(dists, options)
        # target sinusoid waves
        self._targets(y1, y2)
        return self

    def _predictions(
        self,
        dists: Distribution,
        options: Sequence[dict[str, Any]],
    ) -> None:
        """Display predictive distributions with confidence intervals.

        Args:
            dists (`Distribution`): The given distribution.
            **kwds (`Any`): Extra keyword arguments for `ax.plot` and `ax.fill_between`.
        """
        match dists:
            case Gaussian(mean, std):
                self._interval(mean, std, **options[0])
            case GaussianMixture(_, Gaussian(mean, std)):
                _, k, _ = mean.shape
                for i in range(k):
                    self._interval(mean[:, i], std[:, i], **options[i % len(options)])
            case _:
                raise NotImplementedError

    def _interval(self, mean: Array, std: Array, **kwds: Any) -> None:
        """Display the 1-sigma interval of a normal distribution.

        Args:
            mean (`Array`): Mean of the normal distribution.
            std (`Array): Standard deviation of the normal distribution.
            **kwds (`Any`): Extra keyword arguments fo `ax.plot` and `ax.fill_between`.
        """
        self.axes.plot(x, mean, **kwds)
        kwds.setdefault('alpha', 0.2)
        self.axes.fill_between(x, (mean - std).ravel(), (mean + std).ravel(), **kwds)

    def _targets(self, y1: Array, y2: Array) -> None:
        """Display the target sinusoid waves.

        Args:
            y1 (`Array`): Values of the first sinusoid wave.
            y2 (`Array`): Values of the second sinusoid wave.
        """
        kwds = {'alpha': 0.2, 'c': 'gray', 'linewidth': 1}
        self.axes.plot(x, y1, **kwds)
        self.axes.plot(x, y2, **kwds)
