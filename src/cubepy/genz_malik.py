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

from functools import cache
from typing import Callable

import numpy as np

from . import points
from .type_aliases import *


@cache
def genz_malik_weights(dim: int) -> NPF:
    return np.array(
        [
            (12824.0 - 9120.0 * dim + 400.0 * dim ** 2) / 19683.0,
            980.0 / 6561.0,
            (1820.0 - 400.0 * dim) / 19683.0,
            200.0 / 19683.0,
            (6859.0 / 19683.0) / 2 ** dim,
        ]
    )


@cache
def genz_malik_err_weights(dim: int) -> NPF:
    return np.array(
        [
            (729.0 - 950.0 * dim + 50.0 * dim ** 2) / 729.0,
            245.0 / 486.0,
            (265.0 - 100.0 * dim) / 1458.0,
            25.0 / 729.0,
        ]
    )


def genz_malik(
    f: Callable, centers: NPF, halfwidths: NPF, volumes: NPF
) -> tuple[NPF, NPF, NPI]:

    # if centers.ndim == 1:
    #     raise ValueError("Invalid Centers ndim. Expected more than 1")
    # if halfwidths.ndim == 1:
    #     raise ValueError("Invalid halfwidths ndim. Expected more than 1")
    # if centers.shape != halfwidths.shape:
    #     raise ValueError(
    #         "Invalid Region NDArray shapes, expected centers and "
    #         "halfwidths to be idendically shaped, but got ",
    #         centers.shape,
    #         halfwidths.shape,
    #     )
    # if centers.shape[1:] != volumes.shape:
    #     raise ValueError(
    #         "Invalid Region NDArray shapes, expected centers and "
    #         "vol to share lower 2 dimension shapes, but got ",
    #         centers.shape[1:],
    #         volumes.shape,
    #     )

    ### [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
    ### integration Over an N-dimensional rectangular region"
    ### alpha2 = sqrt(9/70), alpha4 = sqrt(9/10), alpha5 = sqrt(9/19)
    ### ratio = (alpha2 ** 2) / (alpha4 ** 2)
    alpha2 = 0.35856858280031809199064515390793749545406372969943071
    alpha4 = 0.94868329805051379959966806332981556011586654179756505
    alpha5 = 0.68824720161168529772162873429362352512689535661564885
    ratio = 0.14285714285714285714285714285714285714285714285714281

    width_alpha2 = halfwidths * alpha2
    width_alpha4 = halfwidths * alpha4
    width_alpha5 = halfwidths * alpha5

    # p shape [ domain_dim, points, ... ]
    p = points.fullsym(centers, width_alpha2, width_alpha4, width_alpha5)
    dim = p.shape[0]
    d1 = points.num_k0k1(dim)
    d2 = points.num_k2(dim)
    d3 = d1 + d2

    # vals shape [ range_dim, points, ... ]
    vals = f(p)

    if vals.ndim <= 2:
        vals = np.expand_dims(vals, 0)

    print(vals.shape)

    vc = vals[:, 0:1, ...]  # center integrand value. shape = [ rdim, 1, ... ]

    # [ range_dim, domain_dim, ... ]
    v01 = vals[:, 1:d1:4, ...] + vals[:, 2:d1:4, ...]
    v23 = vals[:, 3:d1:4, ...] + vals[:, 4:d1:4, ...]

    # fdiff = np.abs(v0 + v1 - 2 * vc - ratio * (v2 + v3 - 2 * vc))
    fdiff = np.abs(v01 - 2 * vc - ratio * (v23 - 2 * vc))
    diff = np.sum(fdiff, axis=0)  # [ domain_dim, ... ]

    vc = np.squeeze(vc, 1)
    s2 = np.sum(v01, axis=1)  # [ range_dim, ... ]
    s3 = np.sum(v23, axis=1)  # [ range_dim, ... ]
    s4 = np.sum(vals[:, d1:d3, ...], axis=1)  # [ range_dim, ... ]
    s5 = np.sum(vals[:, d3:, ...], axis=1)  # [ range_dim, ... ]

    w = genz_malik_weights(dim)  # [5]
    wE = genz_malik_err_weights(dim)  # [4]

    # [5] . [5,range_dim, ... ] = [range_dim, ... ]
    result = volumes * np.tensordot(w, (vc, s2, s3, s4, s5), (0, 0))

    # [4] . [4,range_dim, ... ] = [range_dim, ... ]
    res5th = volumes * np.tensordot(wE, (vc, s2, s3, s4), (0, 0))

    err = np.abs(res5th - result)  # [range_dim, ... ]

    # determine split dimension
    split_dim = np.argmax(diff, axis=0)  # [ ... ]
    split_i = np.zeros_like(diff, dtype=np.bool_)
    np.put_along_axis(split_i, np.expand_dims(split_dim, 0), True, axis=0)

    widest_dim = np.argmax(halfwidths, axis=0)
    widest_i = np.zeros_like(halfwidths, dtype=np.bool_)
    np.put_along_axis(widest_i, np.expand_dims(widest_dim, 0), True, axis=0)

    delta = np.reshape(diff[split_i] - diff[widest_i], diff.shape[1:])  # [ ... ]
    df = np.sum(err, axis=0) / (volumes * 10 ** dim)  # [ ... ]
    too_close = delta <= df
    split_dim[too_close] = widest_dim[too_close]

    return result, err, split_dim
