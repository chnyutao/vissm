from dataclasses import asdict, dataclass


@dataclass
class Config:
    batch_size: int = 64
    """Batch size."""

    epochs: int = 10
    """Epochs."""

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

    def asdict(self) -> dict:
        return asdict(self)
