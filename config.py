from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class DataConfig:
    batch_size: int = 64
    """Batch size."""

    n: int = 1000
    """Number of trajectories."""

    shuffle: bool = True
    """Dataset random shuffling."""

    length: int = 100
    """Length of each trajectory."""


@dataclass
class ModelConfig:
    act: str = 'relu'
    """The activation function (in `jax.nn.*`)."""

    k: int = 2
    """Number of mixture components."""

    latent_size: int = 2
    """Latent distribution dimensionality."""

    posterior: Literal['gaussian'] = 'gaussian'
    """Filtering posterior distribution."""

    tau: float = 1e-5
    """Temperature for Gumbel-softmax sampling."""

    tr: Literal['gaussian'] = 'gaussian'
    """Transition distribution."""


@dataclass
class Config:
    data: DataConfig
    """Dataset configuration."""

    model: ModelConfig
    """Model configuration."""

    epochs: int = 100
    """Epochs."""

    lr: float = 1e-4
    """Learning rate."""

    seed: int = 42
    """Random seed."""

    def asdict(self) -> dict:
        return asdict(self)
