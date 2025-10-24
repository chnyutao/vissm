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
    """Activation function in `jax.nn`."""

    density: Literal['gaussian', 'mixture'] = 'mixture'
    """Model output density family."""

    hidden_sizes: tuple[int, ...] = ()
    """MLP hidden layer sizes."""

    k: int = 2
    """Number of mixture components."""

    loss: Literal['ngem', 'nll', 'sgem'] = 'ngem'
    """
    Loss functions:
    - `nll` Negative log-likelihood,
    - `ngem` Natural gradient expectation maximization,
    - `sgem` Stochastic gradient expectation maximization.
    """

    n: int = 2
    """Dimensionality of each mixture component."""


@dataclass
class OptConfig:
    lr: float = 1e-4
    """Learning rate."""

    name: Literal['adam', 'rmsprop', 'sgd'] = 'sgd'
    """Name of the `optax` optimizer."""


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Dataset configuration."""

    epochs: int = 100
    """Epochs."""

    log_every: int = 1
    """Log evaluation metrics every `log_every` epochs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model configuration."""

    opt: OptConfig = field(default_factory=OptConfig)
    """Optimizer configuration."""

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
            log_every=1,
        ),
    ),
    'sinusoid': (
        'Run sinusoid experiments.',
        Config(
            dataset=DatasetConfig(batch_size=10, n=1000, name='sinusoid'),
            model=ModelConfig(act='sigmoid', hidden_sizes=(100, 100, 100), n=1),
            epochs=1000,
            log_every=100,
        ),
    ),
}
