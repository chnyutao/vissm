from itertools import cycle

import jax.numpy as jnp
import matplotlib.pyplot as plt
from distrax import MultivariateNormalDiag as MvNormal
from jaxtyping import Array

from models import gmvae, vae

plt.rcParams.update(
    {
        'figure.dpi': 300,
        'text.usetex': True,
    }
)

COLORS = cycle(iter(['C0', 'C1']))
MARKERS = cycle(iter(['*', 'p']))


# def heatmapp(distribution: gmvae.Distribution | vae.Distribution) -> None:


def heatmap(mean: Array, std: Array) -> None:
    """Plot the density heatmap for a bivariate Gaussian distribution.

    Args:
        mean (`Array`): Mean of the distribution.
        std (`Array`): Standard deviation of the distribution.
    """
    if not mean.shape == std.shape == (2,):
        return
    # heatmap
    x, y = jnp.unstack(
        # use `std.max()` to ensure square heatmaps
        jnp.linspace(mean - 2 * std.max(), mean + 2 * std.max()),
        axis=-1,
    )
    z = MvNormal(mean, std).prob(jnp.dstack(jnp.meshgrid(x, y)))
    plt.pcolormesh(x, y, z, cmap='RdBu', shading='gouraud')
    # colorbar
    plt.clim(0, float(z.max()))
    plt.colorbar()
    # axis appearance
    plt.axis('scaled')
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$')


def contour(mean: Array, std: Array, *, label: str) -> None:
    """Plot the density contour for a bivariate Gaussian distribution.

    Args:
        mean (`Array`): Mean of the distribution.
        std (`Array`): Standard deviation of the distribution.
        label (`str`): Label for the plot.
    """
    if not mean.shape == std.shape == (2,):
        return
    # mean point
    plt.scatter(
        *mean,
        c=next(COLORS),
        edgecolors='black',
        label=label,
        linewidths=0.2,
        marker=next(MARKERS),
        s=36,
    )
    # contour
    x, y = jnp.unstack(jnp.linspace(mean - 3 * std, mean + 3 * std), axis=-1)
    z = MvNormal(mean, std).prob(jnp.dstack(jnp.meshgrid(x, y)))
    levels = [MvNormal(mean, std).prob(mean + n * std) for n in (2, 1)]
    plt.contour(x, y, z, colors='w', levels=levels, linewidths=1)
    plt.legend(fontsize='small')
