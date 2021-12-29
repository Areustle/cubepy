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
    low: NPF,
    high: NPF,
    args: tuple[Any, ...] = tuple([]),
    abstol: float = 1e-6,
    reltol: float = 1e-6,
    itermax: int | float = 100,
) -> tuple[NPF, NPF]:

    ### Prepare parameters
    # low = np.asarray(low)
    # high = np.asarray(high)
    if low.ndim == 1:
        low = np.expand_dims(low, 0)
    if high.ndim == 1:
        high = np.expand_dims(high, 0)

    num_evts = low.shape[-1]
    evtidx = np.arange(num_evts)

    def _f(x: NPF):
        return f(x, *args)

    # :::::::::::::::: Shapes ::::::::::::::::::
    # {low, high}       [ domain_dim, events ]
    # {center, hwidth}  [ domain_dim, regions_events ]
    # {volume}          [ regions_events ]

    # ---------------- Results -----------------
    # {value}       [ range_dim, regions_events ]
    # {error}       [ range_dim, regions_events ]
    # {split_dim}   [ regions_events ]

    # ------------ Working vectors -------------
    # {cmask}   [ regions_events ]

    # Create initial region
    center, halfwidth, vol = region.region(low, high)

    if center.shape[0] == 1:

        def rule(c: NPF, h: NPF, v: NPF):
            return gauss_kronrod(_f, c, h, v)

    else:

        def rule(c: NPF, h: NPF, v: NPF):
            return genz_malik(_f, c, h, v)

    # perform initial rule application
    value, error, split_dim = rule(center, halfwidth, vol)

    # prepare results
    if value.shape != error.shape:
        # expect [range_dim, regions_events]
        raise RuntimeError("Value/Error shape mismatch after rule application")

    range_dim = value.shape[0]

    # [range_dim, events]
    result_value = np.zeros((range_dim, num_evts))
    result_error = np.zeros((range_dim, num_evts))

    iter = 1
    while np.any(iter < int(itermax)):

        # cmask.shape [ regions_events ]
        # Determine which regions are converged
        cmask = converged.converged(value, error, abstol, reltol)

        # shape [ range_dim, regions_events ]
        # Accumulate converged region results into correct event
        for i in range(range_dim):
            result_value[i, :] += np.bincount(evtidx[cmask], value[i, cmask], num_evts)
            result_error[i, :] += np.bincount(evtidx[cmask], error[i, cmask], num_evts)

        if np.all(cmask):
            break

        # nmask.shape [ regions_events ]
        nmask = ~cmask

        center, halfwidth, vol = region.split(
            center[:, nmask], halfwidth[:, nmask], vol[nmask], split_dim[nmask]
        )

        evtidx = np.tile(evtidx[nmask], 2)

        # # Buffered iteration over region_events
        # it = np.nditer(
        #     [center, halfwidth, vol, None, None, None],
        #     flags=["external_loop", "buffered"],
        #     op_flags=[
        #         ["readonly"],
        #         ["readonly"],
        #         ["readonly"],
        #         ["writeonly", "allocate", "no_broadcast"],
        #         ["writeonly", "allocate", "no_broadcast"],
        #         ["writeonly", "allocate", "no_broadcast"],
        #     ],
        # )
        # with it:
        #     for ci, hi, vi, rv, re, rs in it:
        #         print("nditer:", ci.shape, hi.shape, vi.shape)
        #         rv[...], re[...], rs[...] = rule(ci, hi, vi)
        #     value, error, split_dim = it.operands[3:]

        value, error, split_dim = rule(center, halfwidth, vol)

        iter += 1

    if iter == int(itermax):
        raise RuntimeError("Failed to converge within the iteration limit: ", itermax)

    # return np.sum(result_value, axis=0), np.sum(result_error, axis=0)
    return result_value, result_error
