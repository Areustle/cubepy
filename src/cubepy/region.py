# BSD 3-Clause License
#
# Copyright (c) 2021, Alex Reustle
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import numpy as np

from .type_aliases import NPT  # , NPF, NPI

# from functools import reduce
# from operator import mul

# from typing import

__all__ = ["region", "split"]


def region(low: NPT, high: NPT):
    # ) -> Tuple[ValidBoundsT, ValidBoundsT, ValidBoundsT]:
    """Compute the hyper-rectangular region parameters from given limits of integration."""

    # :::::::::::::::: Shapes ::::::::::::::::::
    # {low, high}.shape [ domain_dim ]
    # centers.shape     [ domain_dim, regions ]
    # halfwidth.shape   [ domain_dim, regions ]
    # vol.shape         [ regions ]

    centers = 0.5 * (high + low)[:, None]
    halfwidth = 0.5 * (high - low)[:, None]
    vol = np.prod(2.0 * halfwidth, axis=0)
    return centers, halfwidth, vol


def split(center, halfwidth, vol, split_dim, active_region_mask):
    """
    Split the input arrays along the specified dimension according to split_dim
    unless every event in which that region is active is converged. In that case, drop
    the region.
    """
    if np.amin(split_dim) < 0 or np.amax(split_dim) >= (center.shape[0]):
        raise IndexError("split dimension invalid")

    # {center, halfwidth}  [ domain_dim, regions ]
    center = center[:, active_region_mask]
    halfwidth = halfwidth[:, active_region_mask]

    # split_dim [ regions ]
    split_dim = split_dim[active_region_mask]
    halfwidth[split_dim, :] *= 0.5
    rnum = center.shape[1]
    center = np.tile(center, (1, 2))
    center[:, :rnum][split_dim] -= halfwidth[split_dim]
    center[:, rnum:][split_dim] += halfwidth[split_dim]

    halfwidth = np.tile(halfwidth, (1, 2))

    # vol [ regions ]
    vol = np.tile(vol[active_region_mask] * 0.5, 2)

    return center, halfwidth, vol
