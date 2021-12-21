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

from typing import Any, Callable

import numpy as np

from . import converged, region
from .gauss_kronrod import gauss_kronrod
from .genz_malik import genz_malik
from .type_aliases import NPF

__all__ = ["integrate"]


def integrate(
    f: Callable,
    low: NPF | float,
    high: NPF | float,
    args: tuple[Any, ...] = tuple([]),
    abstol: float = 1e-5,
    reltol: float = 1e-5,
    itermax: int | float = 50,
    **kwargs
) -> tuple[NPF, NPF]:

    # Prepare parameters
    low = np.asarray(low)
    high = np.asarray(high)

    norm = "inf" if "norm" not in kwargs else kwargs["norm"]

    def _f(x: NPF):
        return f(x, *args)

    # :::::::::::::::: Shapes ::::::::::::::::::
    # {low, high}       [ domain_dim, events ]
    # {center, hwidth}  [ domain_dim, 1(regions), events ]
    # {volume}          [ 1(regions), events ]
    # ---------------- Results -----------------
    # {value}       [ range_dim, regions, events ]
    # {error}       [ range_dim, regions, events ]
    # {split_dim}   [ regions, events ]
    #
    # {cmask}   [ regions, events ]

    # perform initial rule application
    center, hwidth, vol = region.region(low, high)

    def rule1d(c: NPF, h: NPF, _):
        return gauss_kronrod(_f, c, h)

    def rulend(c: NPF, h: NPF, v: NPF):
        return genz_malik(_f, c, h, v)

    rule = rule1d if center.shape[0] == 1 else rulend

    value, error, split_dim = rule(center, hwidth, vol)

    # prepare results
    if value.shape != error.shape:
        # expect [range_dim, regions, events]
        raise RuntimeError("Value/Error shape mismatch after rule application")

    # [range_dim, events]
    result_value = np.zeros(value.shape[:2])
    result_error = np.zeros(value.shape[:2])
    # converge_msk = np.ones(value.shape[:2], dtype=bool)

    iter = 1
    while np.any(iter < int(itermax)):

        # cmask.shape [ regions, events ]
        cmask = converged.converged(value, error, abstol, reltol, norm)

        # shape [range_dim, regions, events]
        result_value += np.sum(value[..., cmask], axis=-1)
        result_error += np.sum(error[..., cmask], axis=-1)

        # nmask.shape [ regions, events ]
        nmask = ~cmask

        if not np.any(nmask):
            break

        # {center, hwidth}  [ domain_dim, regions, events ]
        center, hwidth, vol = center[:, nmask], hwidth[:, nmask], vol[nmask]
        split_dim = split_dim[:, nmask]

        center, hwidth, vol = region.split(center, hwidth, vol, split_dim)
        value, error, split_dim = rule(center, hwidth, vol)

        iter += 1

    if iter == int(itermax):
        raise RuntimeError("Failed to converge within the iteration limit.")

    return np.sum(result_value, axis=0), np.sum(result_error, axis=0)
