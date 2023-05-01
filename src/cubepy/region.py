from __future__ import annotations

from functools import reduce
from operator import mul

import numpy as np

# from .type_aliases import NPT  # , NPF, NPI

# from typing import

__all__ = ["region", "split"]


def region(low, high):
    """Compute the hyper-rectangular region parameters from given limits of integration."""

    # :::::::::::::::: Shapes ::::::::::::::::::
    # {low, high} domain_dim * [ { 1 | nevts } ]
    # center      domain_dim * [ regions, { 1 | nevts } ]
    # halfwidth   domain_dim * [ regions, { 1 | nevts } ]
    # vol.shape         [ nevts ]

    center = [np.expand_dims(0.5 * (hi + lo), 0) for lo, hi in zip(low, high)]
    halfwidth = [np.expand_dims(0.5 * (hi - lo), 0) for lo, hi in zip(low, high)]
    vol = reduce(mul, [2.0 * h for h in halfwidth]).ravel()
    return center, halfwidth, vol


def split(center, halfwidth, vol, split_dim):
    """
    Split the input arrays along the specified dimension according to split_dim
    unless every event in which that region is active is converged. In that case, drop
    the region.
    """
    if np.amin(split_dim) < 0 or np.amax(split_dim) >= len(center):
        raise IndexError("split dimension invalid")

    # {center, halfwidth}  [ regions, { 1 | nevts } ]

    dim = len(center)
    nreg = center[0].shape[0]

    # split_dim [ regions ]
    split_mask = split_dim == np.arange(dim)[:, None]

    for d in range(dim):
        halfwidth[d][split_mask[d]] *= 0.5
        center[d] = np.tile(center[d], (2, 1))
        m1 = split_mask[d]
        m2 = split_mask[d]
        center[d][:nreg][m1] -= halfwidth[d][m1]
        center[d][nreg:][m2] += halfwidth[d][m2]
        halfwidth[d] = np.tile(halfwidth[d], (2, 1))

    # vol [ events ]
    vol *= 0.5

    return center, halfwidth, vol
