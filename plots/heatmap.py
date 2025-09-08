from collections.abc import Sequence
from typing import Any, Self

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from models.distributions import Distribution, Gaussian, GaussianMixture


class Heatmap:
    """Distribution heatmap plot (for continous rv).

    ONLY the first two dimensions of the distributions will be visualized.
    """

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

    def show(
        self,
        prior: Distribution,
        posteriors: Sequence[Distribution],
        cfgs: Sequence[dict[str, Any]],
    ) -> Self:
        """Display a distribution heatmap plot for the priors and posteriors.

        Args:
            prior (`Distribution`): Prior distribution.
            posteriors (`Sequence[Distribution]`): Posterior distributions.
            cfgs (`Sequence[dict[str, Any]]`): Configs for posterior distributions.

        Returns:
            The current instance `self`, allowing chaining methods.
        """
        self._heatmap(prior)
        for posterior, cfg in zip(posteriors, cfgs):
            self._marginal(posterior, **cfg)
            self._mean(posterior, color=cfg.get('color') or cfg.get('c'))
        self._marginal(prior, alpha=0.2, color='k', hatch='///', label='prior')
        return self

    def _heatmap(self, dist: Distribution) -> None:
        """Display the heatmap and the colorbar.

        Args:
            dist (`Distribution`): The given distribution.
        """
        # data
        match dist:
            case Gaussian(mean, std):
                mean, std = mean[..., :2], std[..., :2]
                lo, hi = mean - 2 * std.max(), mean + 2 * std.max()
            case GaussianMixture(_, Gaussian(means, stds)):
                means, stds = means[..., :2], stds[..., :2]
                lo = (means - 2 * stds).min(axis=0)
                hi = (means + 2 * stds).max(axis=0)
                d = (hi - lo).max() - (hi - lo)
                lo, hi = lo - d / 2, hi + d / 2
            case dist:
                raise TypeError(f'Unsupported distribution {type(dist)}')
        x, y = jnp.unstack(jnp.linspace(lo, hi, num=100), axis=-1)
        z = dist.to().prob(jnp.stack(jnp.meshgrid(x, y), axis=-1))
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

    def _marginal(self, dist: Distribution, **kwds: Any) -> None:
        """Display the marginal distributions.

        Args:
            dist (`Distribution`): The given distribution.
            **kwds (`Any`): Extra keyword arguments for `ax.fill_between`.
        """
        # data
        xy = []
        xlim = self.axes['main'].get_xlim()
        ylim = self.axes['main'].get_ylim()
        for idx, (lo, hi) in enumerate([xlim, ylim]):
            idx = slice(idx, idx + 1)
            x = jnp.arange(lo, hi, min((hi - lo) / 100, 1e-2))[:, jnp.newaxis]
            match dist:
                case Gaussian(mean, std):
                    marginal = Gaussian(mean[..., idx], std[..., idx]).to()
                case GaussianMixture(weight, Gaussian(means, stds)):
                    means, stds = means[..., idx], stds[..., idx]
                    marginal = GaussianMixture(weight, Gaussian(means, stds)).to()
                case dist:
                    raise TypeError(f'Unsupported distribution {type(dist)}')
            xy.append((x, marginal.prob(x)))
        (x1, y1), (x2, y2) = xy
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

    def _mean(self, dist: Distribution, **kwds: Any) -> None:
        """Display the mean of the distribution.

        Args:
            dist (`Distribution`): The given distribution.
            **kwds (`Any`): Extra keyword arguments for `ax.scatter`.
        """
        # data
        mean = jnp.array(dist.to().mean())[..., :2]
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
