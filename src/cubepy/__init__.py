__all__ = [
    "integrate",
    "integration",
    "points",
    "genz_malik",
    "gauss_kronrod",
    "region",
]

from . import gauss_kronrod, genz_malik, integration, points, region
from .integration import integrate
