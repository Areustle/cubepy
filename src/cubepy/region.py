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

__all__ = ["region"]


def volume(h: NPF) -> NPF:
    return np.asarray(np.prod(2 * h, axis=0))


def region(low: NPF, high: NPF) -> tuple[NPF, ...]:
    """Compute the hyper-rectangular region parameters from given limits of integration."""

    # {low, high}.shape [ domain_dim, events ]
    # centers.shape     [ domain_dim, events, 1(regions) ]
    # halfwidth.shape   [ domain_dim, events, 1(regions) ]
    # vol.shape         [ domain_dim, events ]

    # if low.ndim <= 1:
    #     low.reshape(1, *low.shape)
    # if high.ndim <= 1:
    #     high.reshape(1, *high.shape)

    if low.shape != high.shape:
        raise RuntimeError("Vector limits of integration must be equivalent.")

    centers = (high + low) / 2.0
    halfwidth = (high - low) / 2.0
    vol = volume(halfwidth)

    return np.expand_dims(centers, -1), np.expand_dims(halfwidth, -1), vol


def split(centers: NPF, halfwidth: NPF, volumes: NPF, split_dim: NPI, axis=-1):

    # centers.shape   [ domain_dim, events, regions ]
    # split_dim.shape [ events, regions ]

    if np.amin(split_dim) < 0 or np.amax(split_dim) > (centers.ndim - 1):
        IndexError("split dimension invalid")

    halfwidth[split_dim] /= 2.0
    volumes /= 2.0

    c2 = np.copy(centers)
    centers[split_dim] -= halfwidth[split_dim]
    c2[split_dim] += halfwidth[split_dim]

    centers = np.stack((centers, c2), axis=axis)
    halfwidth = np.stack((halfwidth, halfwidth), axis=axis)
    volumes = np.stack((volumes, volumes), axis=axis)

    return centers, halfwidth, volumes
