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

import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Pool, Process
from multiprocessing import shared_memory as sm
from typing import Any, Callable

import numpy as np

from . import converged, points, region
from .gauss_kronrod import gauss_kronrod
from .genz_malik import genz_malik
from .type_aliases import NPF

__all__ = ["integrate"]


class Rule:
    def __init__(self, f, domain_dim, evt_idx_arg, *args):
        self.f = f
        # prepare the integral rule
        self.rule = gauss_kronrod if domain_dim == 1 else genz_malik
        self.evt_idx_arg = evt_idx_arg
        self.args = args

        # cen_sh = sm.SharedMemory("cen", create=True, size=center.nbytes)
        # center_ = np.ndarray(center.shape, dtype=center.dtype, buffer=cen_sh.buf)
        # np.copyto(center_, center)

    def __call__(self, center, halfwidth, vol, evi):
        if self.evt_idx_arg:
            return self.rule(
                lambda x: self.f(x, evi, *self.args), center, halfwidth, vol
            )
        else:
            return self.rule(lambda x: self.f(x, *self.args), center, halfwidth, vol)


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

    # prepare the integrand
    def func(evi):
        return (lambda x: f(x, evi, *args)) if evt_idx_arg else lambda x: f(x, *args)

    # prepare the integral rule
    rule = gauss_kronrod if center.shape[0] == 1 else genz_malik

    if range_dim == 0:
        value, error, _ = rule(
            func(evtidx[0:1]), center[:, 0:1], halfwidth[:, 0:1], vol[0:1]
        )

        # prepare results
        if value.shape != error.shape:
            # expect [range_dim, regions_events]
            raise RuntimeError("Value/Error shape mismatch after rule application")
        range_dim = value.shape[0]

    tiled_rule = tiled_rule_generator(
        max_tile_len, parallel, range_dim, Rule(f, domain_dim, evt_idx_arg, *args)
    )

    # perform initial rule application
    value, error, split_dim = tiled_rule(center, halfwidth, vol, evtidx)

    # prepare results
    if value.shape != error.shape:
        # expect [range_dim, regions_events]
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

        if np.all(cmask):
            break

        # nmask.shape [ regions_events ]
        nmask = ~cmask
        # subdivide the un-converged regions
        center, halfwidth, vol = region.split(
            center[:, nmask], halfwidth[:, nmask], vol[nmask], split_dim[nmask]
        )
        evtidx = np.tile(evtidx[nmask], 2)

        value, error, split_dim = tiled_rule(center, halfwidth, vol, evtidx)

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


_rule = None
_domain_dim = None
_range_dim = None
_revt_len = None
_domain_dtype = None


def init_tile_rule_worker(rule, domain_dim, range_dim, revt_len, domain_dtype):
    global _rule
    global _domain_dim
    global _range_dim
    global _revt_len
    global _domain_dtype
    _rule = rule
    _domain_dim = domain_dim
    _range_dim = range_dim
    _revt_len = revt_len
    _domain_dtype = domain_dtype


def tile_rule_worker(strtend):

    strt, end = strtend

    global _rule
    global _domain_dim
    global _range_dim
    global _revt_len
    global _domain_dtype

    c_sh = sm.SharedMemory("cen")
    h_sh = sm.SharedMemory("hal")
    v_sh = sm.SharedMemory("vol")
    e_sh = sm.SharedMemory("evi")

    c = np.ndarray((_domain_dim, _revt_len), dtype=_domain_dtype, buffer=c_sh.buf)
    h = np.ndarray((_domain_dim, _revt_len), dtype=_domain_dtype, buffer=h_sh.buf)
    v = np.ndarray((_revt_len), dtype=_domain_dtype, buffer=v_sh.buf)
    e = np.ndarray((_revt_len), dtype=np.intp, buffer=e_sh.buf)

    val, err, sub = _rule(c[:, strt:end], h[:, strt:end], v[strt:end], e[strt:end])

    c_sh.close()
    h_sh.close()
    v_sh.close()
    e_sh.close()

    val_sh = sm.SharedMemory("val")
    value = np.ndarray((_range_dim, _revt_len), dtype=_domain_dtype, buffer=val_sh.buf)
    value[:, strt:end] = val
    val_sh.close()

    err_sh = sm.SharedMemory("err")
    error = np.ndarray((_range_dim, _revt_len), dtype=_domain_dtype, buffer=err_sh.buf)
    error[:, strt:end] = err
    err_sh.close()

    sub_sh = sm.SharedMemory("sub")
    split_dim = np.ndarray((_revt_len), dtype=np.intp, buffer=sub_sh.buf)
    split_dim[strt:end] = sub
    sub_sh.close()


# prepare tiled iteration of the rule.
def tiled_rule_generator(max_tile_len, parallel, range_dim, rule):
    def tiled_rule(center, halfwidth, vol, evtidx):

        print("Entering tiled_rule")

        domain_dim = center.shape[0]
        domain_dtype = center.dtype

        revt_len = evtidx.shape[0]
        numtiles = int(np.ceil(revt_len / max_tile_len))

        print("Allocating result memory")

        val_err_size = range_dim * revt_len * center.itemsize
        val_sh = sm.SharedMemory("val", create=True, size=val_err_size)
        err_sh = sm.SharedMemory("err", create=True, size=val_err_size)
        sub_sh = sm.SharedMemory("sub", create=True, size=revt_len * np.intp(0).nbytes)

        print("Allocating input memory")
        # cen_half_size = domain_dim * revt_len * center.itemsize
        cen_sh = sm.SharedMemory("cen", create=True, size=center.nbytes)
        hal_sh = sm.SharedMemory("hal", create=True, size=halfwidth.nbytes)
        vol_sh = sm.SharedMemory("vol", create=True, size=vol.nbytes)
        evi_sh = sm.SharedMemory("evi", create=True, size=evtidx.nbytes)

        print("Allocating input memory as numpy arrays")
        center_ = np.ndarray(center.shape, dtype=center.dtype, buffer=cen_sh.buf)
        hwidth_ = np.ndarray(halfwidth.shape, dtype=halfwidth.dtype, buffer=hal_sh.buf)
        vol_ = np.ndarray(vol.shape, dtype=vol.dtype, buffer=vol_sh.buf)
        evi_ = np.ndarray(evtidx.shape, dtype=evtidx.dtype, buffer=evi_sh.buf)

        print("filling input memory with copyto")
        np.copyto(center_, center)
        np.copyto(hwidth_, halfwidth)
        np.copyto(vol_, vol)
        np.copyto(evi_, evtidx)

        c_sp = np.array_split(center, numtiles, -1)
        lengths = np.array(list(map(lambda x: x.shape[1], c_sp)))
        ends = np.cumsum(lengths, 0)
        starts = np.roll(ends, 1)
        starts[0] = 0

        # if isinstance(parallel, bool):
        #     max_workers = None if parallel else 1
        # else:
        #     max_workers = parallel

        # with ProcessPoolExecutor(max_workers=None) as exec:
        #     exec.map(tile_rule_worker, starts, ends)

        # with ThreadPoolExecutor(max_workers=max_workers) as exec:
        #     exec.map(tile_rule_worker, starts, ends)

        # threads = []
        # for s, e in zip(starts, ends):
        #     x = Process(
        #         target=tile_rule_worker,
        #         args=(s, e, domain_dim, range_dim, revt_len, domain_dtype, rule, _f),
        #     )
        #     threads.append(x)
        #     x.start()

        # for t in threads:
        #     t.join()
        print("before Pool")
        with Pool(
            1,
            initializer=init_tile_rule_worker,
            initargs=(rule, domain_dim, range_dim, revt_len, domain_dtype),
        ) as p:
            print("launching jobs")
            p.map(tile_rule_worker, iterable=zip(starts, ends))

        print("cleaning up")
        cen_sh.close()
        hal_sh.close()
        vol_sh.close()
        evi_sh.close()
        cen_sh.unlink()
        hal_sh.unlink()
        vol_sh.unlink()
        evi_sh.unlink()

        value_ = np.ndarray(
            (range_dim, revt_len), dtype=center.dtype, buffer=val_sh.buf
        )
        error_ = np.ndarray(
            (range_dim, revt_len), dtype=center.dtype, buffer=err_sh.buf
        )
        split_dim_ = np.ndarray((revt_len), dtype=np.intp, buffer=sub_sh.buf)

        value = np.copy(value_)
        error = np.copy(error_)
        split_dim = np.copy(split_dim_)

        val_sh.close()
        err_sh.close()
        sub_sh.close()
        val_sh.unlink()
        err_sh.unlink()
        sub_sh.unlink()

        print("leaving tiled_rule")

        return value, error, split_dim

    return tiled_rule
