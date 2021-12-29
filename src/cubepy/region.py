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

from .type_aliases import NPF, NPI

__all__ = ["region", "split"]


def region(low: NPF, high: NPF) -> tuple[NPF, ...]:
    """Compute the hyper-rectangular region parameters from given limits of integration."""

    # :::::::::::::::: Shapes ::::::::::::::::::
    # {low, high}.shape [ domain_dim, events ]
    # centers.shape     [ domain_dim, regions_events ]
    # halfwidth.shape   [ domain_dim, regions_events ]
    # vol.shape         [ regions_events ]

    if low.shape != high.shape:
        raise RuntimeError(
            "Vector limits of integration must be equivalent.", low.shape, high.shape
        )

    if low.ndim == 1:
        low = np.expand_dims(low, 0)
        high = np.expand_dims(high, 0)

    if low.ndim != 2:
        raise RuntimeError("Input limits shape not supported.")

    centers = (high + low) * 0.5
    halfwidth = (high - low) * 0.5
    vol = np.prod(2 * halfwidth, axis=0)

    return centers, halfwidth, vol


def split(centers: NPF, halfwidth: NPF, volumes: NPF, split_dim: NPI):

    # centers.shape   [ domain_dim, regions_events ]
    # split_dim.shape [ 1, regions_events ]

    if np.amin(split_dim) < 0 or np.amax(split_dim) >= (centers.shape[0]):
        raise IndexError("split dimension invalid")

    if split_dim.ndim < centers.ndim:
        split_dim = np.expand_dims(split_dim, 0)

    ## {center, hwidth}  [ domain_dim, (regions, events) ]

    mask = np.zeros_like(centers, dtype=np.bool_)
    np.put_along_axis(mask, split_dim, True, 0)

    h = np.copy(halfwidth)
    h[mask] *= 0.5

    v = np.copy(volumes)
    v *= 0.5

    c1 = np.copy(centers)
    c2 = np.copy(centers)
    c1[mask] -= h[mask]
    c2[mask] += h[mask]

    c = np.concatenate((c1, c2), axis=1)
    h = np.concatenate((h, h), axis=1)
    v = np.concatenate((v, v), axis=0)

    return c, h, v
