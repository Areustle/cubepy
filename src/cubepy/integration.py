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

# import threading
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Pool, Process

# from multiprocessing import shared_memory as sm
from typing import Any, Callable

import numpy as np

from . import converged, points, region
from .input_output import BufferedVectors
from .rule import Rule
from .type_aliases import NPF

__all__ = ["integrate"]


def integrate(
    f: Callable,
    low,
    high,
    args: tuple[Any, ...] = tuple([]),
    abstol: float = 1e-6,
    reltol: float = 1e-6,
    is_1d: bool = False,
    evt_idx_arg: bool = False,
    itermax: int | float = 1000,
    tile_byte_limit: int | float = 2 ** 30,
    range_dim: int = 0,
    parallel: bool = False,
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
        low: float or NDArray
            The finite integral lower bound. If a 1D vector the default behavior is to
            treat the length (`shape[0]`) as the domain of integration of 1 event.
            If a 2D ndarray, the default behavior is to treat the leading dimension
            (`shape[0]`) as the domain of integration, and the trailing dimension as the
            set of independent events over which to integrate. This default behavior
            is controlled by the `event_axis` parameter.
        high: float or NDArray
            The finite integral high bound. Behavior is identical to `low` parameter.
        args: tuple, Optional
            The function arguments to pass into `f`.
        abstol: float, Optional
            Absolute local error tolerance. Default 1e-6
        reltol: float, Optional
            Relative local error tolerance. Default 1e-6
        itermax: int or float, Optional
            Maximum number of subdivision iterations to perform on the integrand.
            Default is 100
    """

    ### Prepare parameters
    low = np.asarray(low)
    high = np.asarray(high)

    if low.shape == ():
        low = np.expand_dims(low, 0)
    if high.shape == ():
        high = np.expand_dims(high, 0)

    if low.shape != high.shape:
        raise RuntimeError(
            "Limits of integration must be the same shape", low.shape, high.shape
        )

    input_shape = low.shape

    ## Reshape the limits of integration if necessary. Domain_dim must be along axis 0
    ## and the final shape for low and high must both be 2D. Ravel trailing dimensions
    ## and reshape results once complete.
    if low.ndim == 1:
        low = np.expand_dims(low, 0 if is_1d else -1)
        high = np.expand_dims(high, 0 if is_1d else -1)
        event_shape = low.shape[1:]
    elif low.ndim > 1:
        if is_1d:
            event_shape = input_shape
            low = np.ravel(low).reshape(1, np.prod(input_shape))
            high = np.ravel(high).reshape(1, np.prod(input_shape))
        else:
            event_shape = low.shape[1:]
            low = np.ravel(low).reshape(input_shape[0], np.prod(event_shape))
            high = np.ravel(high).reshape(input_shape[0], np.prod(event_shape))
    else:
        raise RuntimeError("Unsupported shape for limits of integration", low.shape)

    num_evts = low.shape[-1]
    evtidx = np.arange(num_evts)

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

    domain_dim = center.shape[0]
    pts = points.num_points(domain_dim) if domain_dim > 1 else 15
    event_size = center.itemsize * domain_dim * pts
    max_tile_len = tile_byte_limit / event_size
    max_tile_len = np.maximum(max_tile_len, 1)

    if range_dim == 0:
        rule = Rule(f, domain_dim, evt_idx_arg, *args)
        value, error, _ = rule(center[:, 0:1], halfwidth[:, 0:1], vol[0:1], evtidx[0:1])

        # prepare results
        if value.shape != error.shape:
            # expect [range_dim, regions_events]
            raise RuntimeError("Value/Error shape mismatch after rule application")
        range_dim = value.shape[0]

    bufvec = BufferedVectors(range_dim, center.dtype, parallel)
    bufvec.update_inputs(center, halfwidth, vol, evtidx)
    bufvec.allocate_results(num_evts)

    rule = Rule(f, domain_dim, evt_idx_arg, parallel=parallel, *args)

    tiled_rule = tiled_rule_generator(rule, max_tile_len, parallel)

    # perform initial rule application
    value, error, split_dim = tiled_rule(bufvec)

    # expect [range_dim, regions_events]
    if value.shape != error.shape:
        raise RuntimeError("Value/Error shape mismatch after rule application")
    if range_dim != value.shape[0]:
        raise RuntimeError("Function return value does not match input range_dim.")

    # [range_dim, events]
    result_value = np.zeros((range_dim, num_evts))
    result_error = np.zeros((range_dim, num_evts))

    iter = 1
    while iter < int(itermax):

        # cmask.shape [ regions_events ]
        # Determine which regions are converged
        cmask = converged.converged(value, error, abstol, reltol)

        # Accumulate converged region results into correct event
        for i in range(range_dim):
            result_value[i, :] += np.bincount(evtidx[cmask], value[i, cmask], num_evts)
            result_error[i, :] += np.bincount(evtidx[cmask], error[i, cmask], num_evts)

        if np.all(cmask) or iter == int(itermax):
            break

        # nmask.shape [ regions_events ]
        nmask = ~cmask

        # subdivide the un-converged regions
        center, halfwidth, vol = region.split(
            center[:, nmask], halfwidth[:, nmask], vol[nmask], split_dim[nmask]
        )
        evtidx = np.tile(evtidx[nmask], 2)

        bufvec.update_inputs(center, halfwidth, vol, evtidx)
        bufvec.allocate_results(evtidx.shape[0])

        value, error, split_dim = tiled_rule(bufvec)

        iter += 1

    if iter == int(itermax):
        raise RuntimeError(
            "Failed to converge within the iteration limit: ",
            itermax,
            "Maximum un-converged error estimate: ",
            np.amax(error),
        )

    result_value = np.reshape(result_value, (range_dim, *event_shape))
    result_error = np.reshape(result_error, (range_dim, *event_shape))

    # return np.sum(result_value, axis=0), np.sum(result_error, axis=0)
    return result_value, result_error


def tile_indices(numtotal: int, max_tile_len: int):
    numtiles = int(np.ceil(numtotal / max_tile_len))
    if numtiles <= 0:
        raise ValueError("Tile number must be non-negative.") from None
    each, extras = divmod(numtotal, numtiles)
    section_sizes = extras * [each + 1] + (numtiles - extras) * [each]
    ends = np.array(section_sizes, dtype=np.intp).cumsum()
    starts = np.roll(ends, 1)
    starts[0] = 0
    return starts, ends


def tile_rule_worker(start, end, buffered_vectors, rule):

    c = buffered_vectors.get_np("center")
    h = buffered_vectors.get_np("halfwidth")
    v = buffered_vectors.get_np("vol")
    e = buffered_vectors.get_np("evtidx")

    val, err, sub = rule(c[:, start:end], h[:, start:end], v[start:end], e[start:end])

    value = buffered_vectors.get_np("value")
    error = buffered_vectors.get_np("error")
    split = buffered_vectors.get_np("split")

    value[:, start:end] = val
    error[:, start:end] = err
    split[start:end] = sub


# prepare tiled iteration of the rule.
def tiled_rule_generator(rule, max_tile_len, _):
    def tiled_rule(buffs):

        revt_len = buffs.vecs["vol"].shape[0]
        starts, ends = tile_indices(revt_len, max_tile_len)

        # if isinstance(parallel, bool):
        #     max_workers = None if parallel else 1
        # else:
        #     max_workers = parallel

        tile_rule_worker(0, -1, buffs, rule)
        # threads = []
        # for s, e in zip(starts, ends):
        #     x = Process(
        #         target=tile_rule_worker,
        #         args=(s, e, buffs, rule),
        #     )
        #     threads.append(x)
        #     x.start()

        # for t in threads:
        #     t.join()

        # with Pool(
        #     1,
        #     initializer=init_tile_rule_worker,
        #     initargs=(rule, domain_dim, range_dim, revt_len, domain_dtype),
        # ) as p:
        #     p.map(tile_rule_worker, iterable=zip(starts, ends))

        # return value, error, split_dim
        return buffs.result_arrays()

    return tiled_rule
