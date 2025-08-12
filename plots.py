from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from models.distributions import Distribution, Gaussian, GaussianMixture

# matplotlib global config
plt.rcParams.update(
    {
        'figure.dpi': 300,
        'text.usetex': True,
    }
)


class Heatmap:
    """Distribution heatmap plot (for continous rv)."""

    fig: Figure
    axes: dict[str, Axes]

    def __init__(self) -> None:
        """Initialize a heatmap plot."""
        scale = 1.0
        self.fig, self.axes = plt.subplot_mosaic(
            [
                ['.', 'marginalx', '.'],
                ['marginaly', 'main', 'bar'],
            ],
            figsize=(4 * scale, 3 * scale),
            width_ratios=(2, 16, 6),
            height_ratios=(2, 16),
        )
        self.fig.tight_layout()
        # magic parameters to ensure subplots alignment
        # need to adjust `bottom` according to `scale`
        self.fig.subplots_adjust(left=0, bottom=0.045, top=1, wspace=0, hspace=0)

    def density(self, dist: Distribution) -> None:
        """Plot the density heatmap of a given distribution.

        Args:
            dist (`Distribution`): The given distribution.
                See `models.distributions.Distribution`.
        """
        # data
        if isinstance(dist, Gaussian):
            mean, std = dist.mean, dist.std
            lo, hi = mean - 2 * std.max(), mean + 2 * std.max()
            x, y, z = dist.density(lo, hi)
        if isinstance(dist, GaussianMixture):
            means, stds = dist.means, dist.stds
            lo = (means - 2 * stds.max(axis=1)[jnp.newaxis]).min(axis=0)
            hi = (means + 2 * stds.max(axis=1)[jnp.newaxis]).max(axis=0)
            d = (hi - lo).max() - (hi - lo)
            lo, hi = lo - d / 2, hi + d / 2
            x, y, z = dist.density(lo, hi)
        # heatmap
        ax = self.axes['main']
        im = ax.pcolormesh(x, y, z, cmap='RdBu', shading='gouraud')
        im.set_clim(0, float(z.max()))
        ax.axis('scaled')
        ax.set_xticks(jnp.linspace(x.min(), x.max(), 5))
        ax.set_yticks(jnp.linspace(y.min(), y.max(), 5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel(r'$\mathbf{z}_1$')
        ax.set_ylabel(r'$\mathbf{z}_2$')
        ax.xaxis.set_label_position('top')
        ax.yaxis.tick_right()
        # colorbar
        ax = self.axes['bar']
        ax.axis('off')
        ax = ax.inset_axes((0.4, 0, 0.2, 1))
        ticks = jnp.linspace(0, z.max(), num=5)
        bar = self.fig.colorbar(im, cax=ax)
        bar.set_ticks(list(ticks))
        bar.set_ticklabels([f'{tick:.2f}' for tick in ticks])

    def marginal(self, dist: Distribution, **kwds: Any):
        """Plot the marginals of a given distribution.

        Args:
            dist (`Distribution`): The given distribution.
                See `models.distributions.Distribution`.
            **kwds (`Any`): Extra keyword arguments for `ax.fill_between`.
        """
        # data
        x1, y1 = dist.marginal(*self.axes['main'].get_xlim(), dim=0)
        x2, y2 = dist.marginal(*self.axes['main'].get_ylim(), dim=1)
        # marginal x
        ax = self.axes['marginalx']
        ax.sharex(self.axes['main'])
        ax.fill_between(x1.ravel(), y1.ravel(), **kwds)
        ax.axis('off')
        # marginal y
        ax = self.axes['marginaly']
        ax.sharey(self.axes['main'])
        ax.fill_betweenx(x2.ravel(), y2.ravel(), **kwds)
        ax.invert_xaxis()
        ax.axis('off')
        # legends
        ax = self.axes['main']
        ax.fill_between([], [], **kwds)
        ax.legend(fontsize='small')

    def mean(self, dist: Distribution, **kwds: Any) -> None:
        """Plot the mean (points) of a given distribution.

        Args:
            dist (`Distribution`): The given distribution.
                See `models.distributions.Distribution`.
            **kwds (`Any`): Extra keyword arguments for `ax.scatter`.
        """
        # data
        mean = dist.to().mean()
        assert isinstance(mean, Array)
        x, y = mean
        # connection patch
        ax = self.axes['main']
        configs = {'alpha': 0.5, 'color': 'w', 'linestyle': (0, (1, 3))}
        art = ConnectionPatch((x, y), (x, ax.get_ylim()[1]), 'data', **configs)
        ax.add_artist(art)
        art = ConnectionPatch((x, y), (ax.get_xlim()[0], y), 'data', **configs)
        ax.add_artist(art)
        # mean
        ax.scatter(
            *mean,
            edgecolors='k',
            linewidths=0.5,
            marker='*',
            s=self.fig.get_figwidth() * 10,
            **kwds,
        )


class Bars:
    """Distribution bar plot (for discrete rv)."""

    fig: Figure
    axes: Axes

    def __init__(self) -> None:
        """Initialize a distribution bar plot."""
        scale = 1.0
        self.fig, self.axes = plt.subplots(
            figsize=(4 * scale, 3 * scale),
            layout='constrained',
        )

    def barh(self, dist: Distribution, **kwds: Any) -> None:
        """Plot the logits of a given distribution as bars.

        Args:
            dist (`Distribution`): The given distribution.
                See `models.distributions.Distribution`.
            **kwds (`Any`): Extra keyword arguments for `ax.barh`.
        """
        # data
        if isinstance(dist, Gaussian):
            return
        if isinstance(dist, GaussianMixture):
            probs = jnp.exp(dist.logits)
        # bars
        if self.axes.get_yticks().dtype == float:
            self.axes.set_yticks([])  # clear default ticks
        ys = jnp.arange(len(probs)) + len(self.axes.get_yticks())
        yticks = [*self.axes.get_yticks(), *ys]
        self.axes.barh(ys, probs, align='center', **kwds)
        self.axes.set_xticks([0, 0.5, 1])
        self.axes.set_yticks(yticks)
        self.axes.set_yticklabels([f'${i % len(probs)}$' for i in range(len(yticks))])
        self.axes.set_xlabel('$\\mathrm{Pr}(y)$')
        self.axes.set_ylabel('$y$')
        self.axes.legend(fontsize='small')
