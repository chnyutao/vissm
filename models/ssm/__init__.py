from .gauss import GaussSSM
from .mixture import MixtureSSM

SSM = GaussSSM | MixtureSSM

__all__ = ['SSM', 'GaussSSM', 'MixtureSSM']
