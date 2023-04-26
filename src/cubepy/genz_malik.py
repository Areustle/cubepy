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
from .type_aliases import NPF, NPI


# def genz_malik_weights(dim: int) -> NPF:
@cache
def rule_weights(dim: int) -> NPF:
    return np.array(
        [
            (12824.0 - 9120.0 * dim + 400.0 * dim**2) / 19683.0,
            0.149367474470355128791342783112330437433318091754305746075293,  # 980/6561
            (1820.0 - 400.0 * dim) / 19683.0,
            0.010161052685058172026621958034852410709749530051313316059543,  # 200/19683
            (6859.0 / 19683.0) / 2**dim,
        ]
    )


# def genz_malik_err_weights(dim: int) -> NPF:
@cache
def error_weights(dim: int) -> NPF:
    return np.array(
        [
            (729.0 - 950.0 * dim + 50.0 * dim**2) / 729.0,
            0.50411522633744855967078189300411522633744855967078189300411522,  # 245/486
            (265.0 - 100.0 * dim) / 1458.0,
            0.034293552812071330589849108367626886145404663923182441700960219,  # 25/729
        ]
    )


@cache
def div_diff_weights(dim: int) -> NPF:
    ratio = 0.14285714285714285714285714285714285714285714285714281  # ⍺₂² / ⍺₄²
    a = np.zeros((dim, points.num_k0k1(dim)), dtype=float)
    a[:, 0] = -2 + 2 * ratio
    # for i in range(dim):
    #     start = 1 + (i * 4)
    #     stop = start + 4
    #     a[i, start:stop] = [1.0, 1.0, -ratio, -ratio]

    k1T0 = np.tile(np.arange(dim)[:, None], (1, 4))
    k1T1 = np.arange(1, points.num_k0k1(dim)).reshape((dim, 4))
    # M1 = np.array([-alpha2, alpha2, -alpha4, alpha4], dtype=dtype)
    a[k1T0, k1T1] = [1.0, 1.0, -ratio, -ratio]

    return a


def rule_split_dim(vals, err, halfwidth, volume):
    # vals [ points, regions, events ]
    # err [ regions, events ]
    # halfwidth [ domain_dim, regions ]
    # volume [ regions ]

    dim = halfwidth.shape[0]
    npts, nreg, nevt = vals.shape
    d1 = points.num_k0k1(dim)

    # Compute the 4th divided difference to determine dimension on which to split.
    s1 = (npts, nreg * nevt)
    s2 = (dim, nreg, nevt)

    # [ domain_dim, regions, events ] = [ domain_dim, d1 ] @ [ d1, regions, events ]
    # fdiff = div_diff_weights(dim) @ vals.reshape(s1)[:d1, ...]
    # [ domain_dim, regions ]
    diff = np.linalg.norm(
        (div_diff_weights(dim) @ vals.reshape(s1)[:d1, ...]).reshape(s2), ord=1, axis=-1
    )

    split_dim = np.argmax(diff, axis=0)  # [ regions ]
    widest_dim = np.argmax(halfwidth, axis=0)  # [ regions ]

    # [ domain_dim, regions ]
    delta = np.abs(diff[split_dim, np.arange(nreg)] - diff[widest_dim, np.arange(nreg)])
    df = np.sum(err, axis=1) * (volume * 10 ** (-dim))  # [ regions ]

    too_close = delta <= df
    split_dim[too_close] = widest_dim[too_close]
    return split_dim


@cache
def error_weights_vec(dim: int) -> NPF:
    d1 = points.num_k0k1(dim)
    d2 = points.num_k2(dim)
    d3 = d1 + d2
    a = np.zeros(d3)
    wE = error_weights(dim)
    a[0] = wE[0]
    a[1:d1:4] = wE[1]
    a[2:d1:4] = wE[1]
    a[3:d1:4] = wE[2]
    a[4:d1:4] = wE[2]
    a[d1:d3] = wE[3]
    return a


@cache
def rule_weights_vec(dim: int) -> NPF:
    d1 = points.num_k0k1(dim)
    d2 = points.num_k2(dim)
    d3 = d1 + d2
    a = np.zeros(points.num_points(dim))
    w = rule_weights(dim)
    a[0] = w[0]
    a[1:d1:4] = w[1]
    a[2:d1:4] = w[1]
    a[3:d1:4] = w[2]
    a[4:d1:4] = w[2]
    a[d1:d3] = w[3]
    a[d3:] = w[4]
    return a


@cache
def rule_error_weights(dim: int) -> NPF:
    """
    The necessary weights for computing the 7th and 5th order genz malik rule values
    laid out in matrix form.
    """
    d1 = points.num_k0k1(dim)
    d2 = points.num_k2(dim)
    d3 = d1 + d2
    a = np.zeros((2, points.num_points(dim)))
    w = rule_weights(dim)
    wE = error_weights(dim)
    a[:, 0:1] = np.array([w[0], wE[0]])[:, None]
    a[:, 1:d1:4] = np.array([w[1], wE[1]])[:, None]
    a[:, 2:d1:4] = np.array([w[1], wE[1]])[:, None]
    a[:, 3:d1:4] = np.array([w[2], wE[2]])[:, None]
    a[:, 4:d1:4] = np.array([w[2], wE[2]])[:, None]
    a[:, d1:d3] = np.array([w[3], wE[3]])[:, None]
    a[:, d3:] = np.array([w[4], 0])[:, None]
    return a


def genz_malik(f: Callable, center, halfwidth, volume) -> tuple[NPF, NPF, NPI]:
    # [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
    # integration Over an N-dimensional rectangular region", updated by Bernstein,
    # Espelid, Genz in "An Adaptive Algorithm for the Approximate Calculation of
    # Multiple Integrals"
    # alpha2 = √(9/70)
    # alpha4 = √(9/10)
    # alpha5 = √(9/19)
    # ratio = 0.14285714285714285714285714285714285714285714285714281  # ⍺₂² / ⍺₄²

    # p shape [ domain_dim, points, regions ]
    p = points.gm_pts(center, halfwidth)
    dim = p.shape[0]
    npts = p.shape[1]
    nreg = p.shape[2]

    # save shape then reshape to [domain_dim, (points*regions), 1] before passing to f
    p = np.reshape(p, (dim, nreg * npts, 1))
    vals = f(p)
    # vals shape [ points * regions, events  ] ==> [ points, regions, events ]
    nevt = vals.size // (nreg * npts)
    vals = np.reshape(vals, (npts, nreg, nevt))
    # any eventwise operations in the integrand will automatically be a (N, 1) * (M)
    # operation, so should return events in the trailing dimension.

    s1 = (npts, nreg * nevt)
    s2 = (2, nreg, nevt)

    result, res5th = (rule_error_weights(dim) @ vals.reshape(s1)).reshape(s2) * volume[
        None, :, None
    ]
    err = np.abs(res5th - result)  # [ regions, events ]
    split_dim = rule_split_dim(vals, err, halfwidth, volume)  # [ regions ]

    # [regions, events] [ regions, events ] [ regions ]
    return result, err, split_dim
