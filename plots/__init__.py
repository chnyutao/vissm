import matplotlib.pyplot as plt

from .heatmap import Heatmap
from .sinusoid import Sinusoid

# matplotlib global config
plt.rcParams.update(
    {
        'figure.dpi': 300,
        'text.usetex': True,
    }
)

__all__ = ['Heatmap', 'Sinusoid']
