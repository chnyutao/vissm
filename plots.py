import jax.numpy as jnp
import matplotlib.pyplot as plt
from distrax import MultivariateNormalDiag as MvNormal
from distrax import Normal
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch

from models.distributions import Distribution, Gaussian

# matplotlib global config
plt.rcParams.update(
    {
        'figure.dpi': 300,
        'text.usetex': True,
    }
)


def make_distribution_map(scale: float = 2.0) -> tuple[Figure, dict[str, Axes]]:
    """Initialize a distribution map plot.

    Args:
        scale (`float`, optional): Scaling factor for the figure size. Defaults to 2.0.

    Returns:
        A 2-tuple containing the figure and a dictionary of axes, where the dictionary
        keys include `'marginalx'`, `'marginaly'`, `'main'`, and `'bar'`.
    """
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
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig, axes


def heatmap(fig: Figure, axes: dict[str, Axes], dist: Distribution) -> None:
    # data
    if isinstance(dist, Gaussian):
        mean, std = dist.mean, dist.std
        x, y = jnp.unstack(
            # use `std.max()` to ensure square heatmaps
            jnp.linspace(mean - 2 * std.max(), mean + 2 * std.max()),
            axis=-1,
        )
        z = MvNormal(mean, std).prob(jnp.dstack(jnp.meshgrid(x, y)))
    # heatmap
    ax = axes['main']
    image = ax.pcolormesh(x, y, z, cmap='RdBu', shading='gouraud')
    image.set_clim(0, float(z.max()))
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
    bar = fig.colorbar(image, cax=ax)
    bar.set_ticks(list(ticks))
    bar.set_ticklabels([f'${tick:.2f}$' for tick in ticks])


def marginal(
    fig: Figure,
    axes: dict[str, Axes],
    dist: Distribution,
    *,
    color: str,
):
    # data
    if isinstance(dist, Gaussian):
        mean, std = dist.mean, dist.std
        x1 = jnp.arange(*axes['main'].get_xlim(), 0.01)
        y1 = Normal(mean[0], std[0]).prob(x1)
        x2 = jnp.arange(*axes['main'].get_ylim(), 0.01)
        y2 = Normal(mean[1], std[1]).prob(x2)
    # marginal x
    ax = axes['marginalx']
    ax.sharex(axes['main'])
    ax.fill_between(x1.ravel(), y1.ravel(), alpha=0.3, color=color)
    ax.axis('off')
    # marginal y
    ax = axes['marginaly']
    ax.sharey(axes['main'])
    ax.fill_betweenx(x2.ravel(), y2.ravel(), alpha=0.3, color=color)
    ax.invert_xaxis()
    ax.axis('off')


def contour(
    fig: Figure,
    axes: dict[str, Axes],
    dist: Distribution,
    *,
    color: str,
    label: str,
) -> None:
    # data
    if isinstance(dist, Gaussian):
        mean, std = dist.mean, dist.std
    # connection patch
    ax = axes['main']
    x, y = tuple(mean)
    kwargs = {'color': 'w', 'linestyle': (0, (1, 3)), 'linewidth': 0.8}
    art = ConnectionPatch((x, y), (x, ax.get_ylim()[1]), 'data', **kwargs)
    ax.add_artist(art)
    art = ConnectionPatch((x, y), (ax.get_xlim()[0], y), 'data', **kwargs)
    ax.add_artist(art)
    # center point
    ax.scatter(
        *mean,
        c=color,
        edgecolors='black',
        label=label,
        linewidths=0.2,
        marker='*',
        s=36,
    )
    # contour
    x = jnp.arange(*ax.get_xlim(), 0.01)
    y = jnp.arange(*ax.get_ylim(), 0.01)
    z = MvNormal(mean, std).prob(jnp.dstack(jnp.meshgrid(x, y)))
    levels = [MvNormal(mean, std).prob(mean + n * std) for n in (2, 1)]
    ax.contour(x, y, z, colors='w', levels=levels, linewidths=0.8)
    ax.legend(fontsize='small')
