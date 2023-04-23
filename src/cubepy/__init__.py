__all__ = [
    "integrate",
    "input",
    "integration",
    "points",
    "genz_malik",
    "gauss_kronrod",
    "region",
    "type_aliases",
]

from . import (
    gauss_kronrod,
    genz_malik,
    input,
    integration,
    points,
    region,
    type_aliases,
)
from .integration import integrate
