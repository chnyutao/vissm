from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class Config:
    batch_size: int = 64
    """Batch size."""

    epochs: int = 10
    """Epochs."""

    k: int = 2
    """Number of components in Gaussianx mixture."""

    latent_size: int = 2
    """Latent Gaussian dimensionality."""

    length: int = 100
    """Length of each trajectory."""

    lr: float = 1e-4
    """Learning rate."""

    n: int = 1000
    """Number of trajectories."""

    seed: int = 42
    """Random seed."""

    shuffle: bool = True
    """Dataset random shuffling."""

    tau: float = 0.1
    """Temperature for Gumbel-softmax sampling."""

    vae: Literal['gmvae', 'vae'] = 'vae'
    """Type of variational auto-encoder to use for state estimation."""

    def asdict(self) -> dict:
        return asdict(self)
