from __future__ import annotations

import numpy as np

from .type_aliases import NPB, NPF, NPT


def converged(
    result: NPF, local_error: NPF, parent_result: NPT, rtol: float, atol: float
) -> NPB:
    """Determine wether error values are below threshod for convergence."""

    r = result.shape[0] // 2
    E = local_error
    if r >= 1:
        E2 = np.abs(parent_result - (result[:r] + result[r:]))  # [ r/2, events ]
        ES = E[:r] + E[r:]
        inv_error = np.reciprocal(
            np.where(ES <= 2.0 * np.finfo(ES.dtype).eps, 1.0, ES)
        )  # [ r/2, events ]
        E[:r] += E2 * (0.25 + 0.5 * inv_error * E[:r])
        E[r:] += E2 * (0.25 + 0.5 * inv_error * E[r:])

    # {cmask}       [ regions, events ]
    return E <= (atol + rtol * np.abs(result))
