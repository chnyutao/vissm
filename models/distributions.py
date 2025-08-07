import equinox as eqx
from jaxtyping import Array


class Gaussian(eqx.Module):
    "Gaussian distribution parameters."

    mean: Array
    std: Array


class GaussianMixture(eqx.Module):
    """Mixture of Gaussians distribution parameters."""

    logits: Array
    means: Array
    stds: Array


Distribution = Gaussian | GaussianMixture
