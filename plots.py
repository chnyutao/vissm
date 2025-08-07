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


def make_distribution_map() -> tuple[Figure, dict[str, Axes]]:
    """Initialize a distribution map plot.

    Returns:
        A 2-tuple containing the figure and a dictionary of axes, where the dictionary
        keys include `'marginalx'`, `'marginaly'`, `'main'`, and `'bar'`.
    """
    scale = 1.0
    fig, axes = plt.subplot_mosaic(
        [
            ['.', 'marginalx', '.'],
            ['marginaly', 'main', 'bar'],
        ],
        figsize=(4 * scale, 3 * scale),
        width_ratios=(2, 16, 6),
        height_ratios=(2, 16),
    )
    fig.tight_layout()
    # magic parameters to ensure subplots alignment
    # need to adjust `bottom` according to `scale`
    fig.subplots_adjust(left=0, bottom=0.045, top=1, wspace=0, hspace=0)
    return fig, axes


def heatmap(fig: Figure, axes: dict[str, Axes], dist: Distribution) -> None:
    """Plot the density heatmap of a given distribution.

    Args:
        fig (`Figure`): A matplotlib figure.
        axes (`dict[str, Axes]`): A dictionary of matplotlib axes.
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
    ax = axes['main']
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
    ax = axes['bar']
    ax.axis('off')
    ax = ax.inset_axes((0.4, 0, 0.2, 1))
    ticks = jnp.linspace(0, z.max(), num=5)
    bar = fig.colorbar(im, cax=ax)
    bar.set_ticks(list(ticks))
    bar.set_ticklabels([f'{tick:.2f}' for tick in ticks])


def marginal(
    fig: Figure,
    axes: dict[str, Axes],
    dist: Distribution,
    **kwds: Any,
):
    """Plot the marginal density of a given distribution.

    Args:
        fig (`Figure`): A matplotlib figure.
        axes (`dict[str, Axes]`): A dictionary of matplotlib axes.
        dist (`Distribution`): The given distribution.
            See `models.distributions.Distribution`.
        **kwds (`Any`): Extra keyword arguments for `ax.fill_between`.
    """
    # data
    x1, y1 = dist.marginal(*axes['main'].get_xlim(), dim=0)
    x2, y2 = dist.marginal(*axes['main'].get_ylim(), dim=1)
    # marginal x
    ax = axes['marginalx']
    ax.sharex(axes['main'])
    ax.fill_between(x1.ravel(), y1.ravel(), **kwds)
    ax.axis('off')
    # marginal y
    ax = axes['marginaly']
    ax.sharey(axes['main'])
    ax.fill_betweenx(x2.ravel(), y2.ravel(), **kwds)
    ax.invert_xaxis()
    ax.axis('off')
    # legends
    ax = axes['main']
    ax.fill_between([], [], **kwds)
    ax.legend(fontsize='small')


def mean(
    fig: Figure,
    axes: dict[str, Axes],
    dist: Distribution,
    **kwds: Any,
) -> None:
    """Plot the mean of a given distribution.

    Args:
        fig (`Figure`): A matplotlib figure.
        axes (`dict[str, Axes]`): A dictionary of matplotlib axes.
        dist (`Distribution`): The given distribution.
            See `models.distributions.Distribution`.
        **kwds (`Any`): Extra keyword arguments for `ax.scatter`.
    """
    # data
    mean = dist.dist().mean()
    assert isinstance(mean, Array)
    x, y = mean
    # connection patch
    ax = axes['main']
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
        s=fig.get_figwidth() * 10,
        **kwds,
    )
