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

from typing import Callable

import numpy as np

from . import points
from .type_aliases import NPF, NPI


def genz_malik_weights(dim: int) -> NPF:
    return np.array(
        [
            (12824.0 - 9120.0 * dim + 400.0 * dim**2) / 19683.0,
            0.149367474470355128791342783112330437433318091754305746075293,  # 980/6561
            (1820.0 - 400.0 * dim) / 19683.0,
            0.010161052685058172026621958034852410709749530051313316059543,  # 200/19683
            (6859.0 / 19683.0) / 2**dim,
        ]
    )


def genz_malik_err_weights(dim: int) -> NPF:
    return np.array(
        [
            (729.0 - 950.0 * dim + 50.0 * dim**2) / 729.0,
            0.50411522633744855967078189300411522633744855967078189300411522,  # 245/486
            (265.0 - 100.0 * dim) / 1458.0,
            0.034293552812071330589849108367626886145404663923182441700960219,  # 25/729
        ]
    )


# @profile


def genz_malik(f: Callable, center, halfwidth, volume) -> tuple[NPF, NPF, NPI]:
    # [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
    # integration Over an N-dimensional rectangular region", updated by Bernstein,
    # Espelid, Genz in "An Adaptive Algorithm for the Approximate Calculation of
    # Multiple Integrals"
    # alpha2 = √(9/70)
    # alpha4 = √(9/10)
    # alpha5 = √(9/19)
    ratio = 0.14285714285714285714285714285714285714285714285714281  # ⍺₂² / ⍺₄²

    # p shape [ domain_dim, points, regions ]
    # p shape [ domain_dim, regions, points ]
    p = points.gm_pts(center, halfwidth)
    ndim = p.shape[0]
    nreg = p.shape[2]
    d1 = points.num_k0k1(ndim)
    d2 = points.num_k2(ndim)
    d3 = d1 + d2

    # vals shape [ points, regions, events ]
    # vals shape [ events, regions, points  ]
    vals = f(p)

    if vals.ndim == 2:
        vals = np.expand_dims(vals, 2)

    # print("vals shape", vals.shape)

    vc = vals[0:1]  # center integrand value. shape = [ 1, regions, events ]
    # N.B. [0:1] is a load-bearing slice to keep the leading dimension from getting
    # squeezed out. Same as the more verbose, less efficient vals[0][None, ...]

    # [ domain_dim, regions, events ]
    v01 = vals[1:d1:4] + vals[2:d1:4]
    v23 = vals[3:d1:4] + vals[4:d1:4]

    # Compute the 4th divided difference to determine dimension on which to split.
    # [ domain_dim, regions ]
    diff = np.linalg.norm(v01 - 2 * vc - ratio * (v23 - 2 * vc), ord=1, axis=-1)

    vc = np.squeeze(vc, 0)  # [ regions, events ]
    s2 = np.sum(v01, axis=0)  # [ regions, events ]
    s3 = np.sum(v23, axis=0)  # [ regions, events ]
    s4 = np.sum(vals[d1:d3], axis=0)  # [ regions, events ]
    s5 = np.sum(vals[d3:], axis=0)  # [ regions, events ]

    w = genz_malik_weights(ndim)  # [5]
    wE = genz_malik_err_weights(ndim)  # [4]

    # [ regions, events ] = [5] . [ 5, regions, events ]
    result = volume[:, None] * np.tensordot(w, (vc, s2, s3, s4, s5), (0, 0))
    # print("volume shape", volume.shape)
    # print("result shape", result.shape)

    # [ regions, events ] = [4] . [ 4, regions, events ]
    res5th = volume[:, None] * np.tensordot(wE, (vc, s2, s3, s4), (0, 0))

    err = np.abs(res5th - result)  # [ regions, events ]

    # determine split dimension
    split_dim = np.argmax(diff, axis=0)  # [ regions ]
    widest_dim = np.argmax(halfwidth, axis=0)  # [ regions ]

    # [ domain_dim, regions ]
    delta = diff[split_dim, np.arange(nreg)] - diff[widest_dim, np.arange(nreg)]
    df = np.sum(err, axis=1) * (volume * 10 ** (-ndim))  # [ regions ]
    too_close = delta <= df
    split_dim[too_close] = widest_dim[too_close]

    # print("vals", vals.shape)
    # print("result", result.shape)
    # print("err", err.shape)
    # print("split_dim", split_dim.shape)

    # [regions, events] [ regions, events ] [ regions ]
    return result, err, split_dim
