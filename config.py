from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class DatasetConfig:
    batch_size: int = 64
    """Batch size."""

    length: int = 100
    """Length of each trajectory (in random walk)."""

    n: int = 1000
    """
    Number of data points (in bimodal / sinusoid), or
    number of trajectories (in random walk).
    """

    name: Literal['bimodal', 'random_walk', 'sinusoid'] = 'bimodal'
    """Name of the dataset."""

    shuffle: bool = True
    """Dataset random shuffling."""


@dataclass
class ModelConfig:
    act: str = 'relu'
    """The activation function in `jax.nn`."""

    density: Literal['gaussian', 'mixture'] = 'mixture'
    """Output distribution family."""

    k: int = 2
    """Number of mixture components."""

    loss: Literal['mle', 'ngem', 'sgem'] = 'ngem'
    """
    Loss functions:
    - `mle` Maximum likelihood,
    - `ngem` Natural gradient expectation maximization,
    - `sgem` Stochastic gradient expectation maximization.
    """

    n: int = 2
    """Dimensionality of each mixture component."""


@dataclass
class OptConfig:
    name: Literal['adam', 'sgd'] = 'sgd'
    """Name of the `optax` optimizer."""

    lr: float = 1e-4
    """Learning rate."""


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Dataset configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model configuration."""

    opt: OptConfig = field(default_factory=OptConfig)
    """Optimizer configuration."""

    epochs: int = 100
    """Epochs."""

    seed: int = 42
    """Random seed."""

    def asdict(self) -> dict:
        return asdict(self)


default_configs = {
    'bimodal': (
        'Run bimodal experiments.',
        Config(
            dataset=DatasetConfig(batch_size=1, n=100, name='bimodal'),
            epochs=10,
        ),
    )
}
