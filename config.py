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

    name: Literal['bimodal', 'canonical', 'sinusoid'] = 'bimodal'
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

    def __repr__(self) -> str:
        return (
            f'{self.model.loss.upper()}/{self.dataset.name}'
            f'-{self.opt.name}({self.opt.lr:.0e})'
            f'-seed={self.seed}'
            f'-act={self.model.act}'
            f'-k={self.model.k}'
        )

    def asdict(self) -> dict:
        return asdict(self)


default_configs = {
    'bimodal': (
        'Run bimodal experiments.',
        Config(
            dataset=DatasetConfig(
                batch_size=1,
                n=100,
                name='bimodal',
            ),
            epochs=10,
            log_every=1,
        ),
    ),
    'canonical': (
        'Run canonical experiments.',
        Config(
            dataset=DatasetConfig(
                batch_size=32,
                n=1000,
                name='canonical',
            ),
            epochs=100,
            log_every=10,
            model=ModelConfig(
                act='gelu',
                hidden_sizes=(128, 128),
                k=3,
                n=1,
            ),
        ),
    ),
    'sinusoid': (
        'Run sinusoid experiments.',
        Config(
            dataset=DatasetConfig(
                batch_size=32,
                n=1000,
                name='sinusoid',
            ),
            epochs=1000,
            log_every=100,
            model=ModelConfig(
                act='gelu',
                hidden_sizes=(128, 128, 128, 128),
                n=1,
            ),
        ),
    ),
}
