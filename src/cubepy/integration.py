# BSD 3-Clause License
#
# Copyright (c) 2021, Alexander Reustle
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

from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from operator import mul
from typing import Any, Callable, Sequence

import numpy as np

from . import converged, input, region
from .gauss_kronrod import gauss_kronrod
from .genz_malik import genz_malik
from .type_aliases import NPF, InputBoundsT

# import multiprocessing as mp

__all__ = ["integrate"]


# @profile
def integrate(
    f: Callable,
    low: InputBoundsT,
    high: InputBoundsT,
    args: Sequence[Any] = tuple([]),
    *,  # kwonly arguments
    nint: None | int = None,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    itermax: int | float = 1000,
    # tile_byte_limit: int | float = 2**25,
    # parallel: bool = False,
) -> tuple[NPF, NPF]:
    """Numerical Cubature in multiple dimensions.

    Functions with 1D domain use an adaptive Gauss Kronrod Quadrature scheme.
    Functions with 2+D domain use an adaptive Genz Malik Cubature scheme.

    The adaptive regional subdivision is performed independently for un-converged
    regions.

    Local convergence is obtained when the global tolerance values exceed a local
    region's error estimate.

    Global convergence is obtained when all regions are locally converged.

    Parameters
    ==========
        f: Callable
            Integrand function: expected function signature is
            `f(x: NDArray, *args) -> NDArray`
            where `x` is the expected vector input, with domain in the leading dimension
            (`x.shape[0]`)
        low: Scalar, Iterable, or NDArray
            The finite integral lower bound. If a 1D vector the default behavior is to
            treat the length (`shape[0]`) as the domain of integration of 1 event.
            If a 2D ndarray, the default behavior is to treat the leading dimension
            (`shape[0]`) as the domain of integration, and the trailing dimension as the
            set of independent events over which to integrate. This default behavior
            is controlled by the `event_axis` parameter.
        high: Scalar, Iterable, or NDArray
            The finite integral high bound. Behavior is identical to `low` parameter.
        args: tuple, Optional
            The function arguments to pass into `f`.
        reltol: float, Optional
            Relative local error tolerance. Default 1e-5
        abstol: float, Optional
            Absolute local error tolerance. Default 1e-8
        itermax: int or float, Optional
            Maximum number of subdivision iterations to perform on the integrand.
            Default is 100
    """

    # :::::::::::::::: Shapes ::::::::::::::::::
    # {low, high}       [ domain_dim ]
    # {center, hwidth}  [ domain_dim, regions ]
    # {volume}          [ regions ]

    # ---------------- Results -----------------
    # {value}       [ events ]
    # {error}       [ events ]
    # {split_dim}   [ events ]

    # ------------ Working vectors -------------
    # {cmask}   [ regions, events ]

    low, high, event_shape = input.parse_input(f, low, high, args, nint)
    nint = low.shape[0]
    num_evts = reduce(mul, event_shape, 1)
    # evtidx = np.expand_dims(np.arange(num_evts), 0)  # [ regions, events ]
    global_active_mask = np.ones((1, num_evts), dtype=bool)  # [ regions, events ]
    global_active_event_idxs = np.arange(num_evts)
    aemsk = input.get_arg_evt_mask(args, event_shape)

    # [ events ]
    result_value = np.zeros(num_evts)
    result_error = np.zeros(num_evts)
    # [ regions, events ]
    parent_value = np.zeros((1, num_evts))

    # Prepare the integrand including applying the event mask to the maskable elements
    # in the args array.
    def _f(ev):
        if nint == 1:
            return lambda x: f(x, *(a[ev] if b else a for a, b in zip(args, aemsk)))
        else:
            return lambda x: f(*x, *(a[ev] if b else a for a, b in zip(args, aemsk)))

    # Create initial region
    center, halfwidth, vol = region.region(low, high)
    # max_tile_len = input.get_max_tile_length(center, tile_byte_limit)

    # prepare the integral rule
    rule_ = gauss_kronrod if len(center) == 1 else genz_malik

    def tiled_rule(c, h, v, e):
        return rule_(_f(e), c, h, v)

    # tiled_rule = tiled_rule_generator(_f, rule, max_tile_len, parallel)

    # Perform initial rule application. [regions, events]
    value, error, split_dim = tiled_rule(
        center, halfwidth, vol, global_active_event_idxs
    )

    # Prepare results
    if value.shape != error.shape:
        # expect [regions, events]
        raise RuntimeError("Value/Error shape mismatch after rule application")

    iter: int = 1
    while iter < int(itermax):
        # print(iter, "====================================================")
        # print("value", value, np.sum(value))
        # print("error", error)
        # Determine which regions are converged [ regions, events ]
        local_cmask = converged.converged(value, error, parent_value, rtol, atol)
        # print("local_cmask")
        # print(local_cmask)

        # Some of the converged regions were converged in a previous iteration, but the
        # event was kept for subsequent iterations so remaining regions could converge.
        # To avoid double counting the converged subregions mask them out. The mask
        # should shrink as events become fully converged.
        cmask = global_active_mask & local_cmask  # [R,E]
        # print("cmask")
        # print(cmask)

        # global event indices of locally converged regions, events [ regions, events ]
        cevtidx = np.broadcast_to(global_active_event_idxs, cmask.shape)[cmask]
        # print("cevtidx")
        # print(cevtidx)

        # Accumulate converged region results into correct global events.
        # bincount is most efficient accumulator when multiple regions in the same
        # event converge.
        # print(np.bincount(cevtidx, value[cmask], num_evts).shape)
        # print(np.bincount(cevtidx, value[cmask], num_evts))
        result_value[global_active_event_idxs] += np.bincount(
            cevtidx, value[cmask], num_evts
        )
        result_error[global_active_event_idxs] += np.bincount(
            cevtidx, error[cmask], num_evts
        )

        if np.all(cmask):
            break

        # Globally active, locally unconverged regions.
        umask = global_active_mask & ~local_cmask  # [R,E]
        # print("umask", umask.shape, umask)
        active_region_mask = np.any(umask, axis=1)  # [ regions ]
        active_event_mask = np.any(umask, axis=0)  # [ events ]

        # global indices of unconverged events # [ kept_events ]
        global_active_event_idxs = global_active_event_idxs[active_event_mask]
        # update parent_values with active values
        # [ kept_regions, kept_events ]
        parent_value = value[np.ix_(active_region_mask, active_event_mask)]

        # subdivide the un-converged regions
        center, halfwidth, vol = region.split(
            center, halfwidth, vol, split_dim, active_region_mask
        )
        # [ kept_regions, events ]

        # perform the rule on un-converged regions.
        value, error, split_dim = tiled_rule(
            center, halfwidth, vol, global_active_event_idxs
        )

        iter += 1

    if iter == int(itermax):
        raise RuntimeWarning(
            "Failed to converge within the iteration limit: ",
            itermax,
            "Maximum un-converged error estimate: ",
            np.amax(error),
        )

    result_value = np.reshape(result_value, event_shape)
    result_error = np.reshape(result_error, event_shape)

    return np.squeeze(result_value), np.squeeze(result_error)


# # prepare tiled version of the rule.
# def tiled_rule_generator(f, rule, max_tile_len, parallel):
#     def tiled_rule(center, halfwidth, vol, evtidx):
#         revt_len = evtidx.shape[0]
#         numtiles = int(np.ceil(revt_len / max_tile_len))
#
#         value = np.empty(revt_len, dtype=center.dtype)
#         error = np.empty(revt_len, dtype=center.dtype)
#         split_dim = np.empty((revt_len), dtype=np.intp)
#
#         c_sp = np.array_split(center, numtiles, -1)
#         h_sp = np.array_split(halfwidth, numtiles, -1)
#         v_sp = np.array_split(vol, numtiles, -1)
#         e_sp = np.array_split(evtidx, numtiles, -1)
#
#         lens = np.roll(np.cumsum(np.array(list(map(lambda x: x.shape[1], c_sp))), 0), 1)
#         lens[0] = 0
#
#         def rule_worker(iter, c, h, v, e):
#             end = iter + c.shape[1]
#             # val, err, sub = rule(f(e), c, h, v)
#             # value[:, iter:end] = val
#             # error[:, iter:end] = err
#             # split_dim[iter:end] = sub
#             value[:, iter:end], error[:, iter:end], split_dim[iter:end] = rule(
#                 f(e), c, h, v
#             )
#
#         if parallel:
#             with ThreadPoolExecutor(max_workers=None) as exec:
#                 exec.map(rule_worker, lens, c_sp, h_sp, v_sp, e_sp)
#         else:
#             for i, c, h, v, e in zip(lens, c_sp, h_sp, v_sp, e_sp):
#                 rule_worker(i, c, h, v, e)
#
#         return value, error, split_dim
#
#     return tiled_rule
