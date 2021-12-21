from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["NPF", "NPI", "NPB"]

NPF = NDArray[np.floating]
NPI = NDArray[np.integer]
NPB = NDArray[np.bool_]
