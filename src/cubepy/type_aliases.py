from __future__ import annotations

import typing

import numpy as np
from numpy.typing import NDArray

__all__ = ["NPF", "NPI", "NPB", "InputBoundsT", "ValidBoundsT"]

NPT = NDArray[typing.Union[np.floating, np.integer]]
NPF = NDArray[np.floating]
NPI = NDArray[np.integer]
NPB = NDArray[np.bool_]

# ValidBoundsT = typing.Tuple[typing.Union[int, float, NPT], ...]
ValidBoundsT = typing.Tuple[NPT, ...]

InputBoundsT = typing.Union[
    int, float, NPT, typing.Sequence[typing.Union[int, float, NPT]], ValidBoundsT
]
# InputBoundsT = typing.Sequence[NPF]
