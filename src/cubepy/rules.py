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

from functools import cache
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from . import points


@cache
def weights(dim: int):
    return np.array(
        [
            (12824.0 - 9120.0 * dim + 400.0 * np.sqrt(dim)) / 19683.0,
            980.0 / 6561.0,
            (1820.0 - 400.0 * dim) / 19683.0,
            200.0 / 19683.0,
            6859.0 / 19683.0 / 2 ** dim,
        ]
    )


@cache
def err_weights(dim: int):
    return np.array(
        [
            729.0 - 950.0 * dim + 50.0 * np.sqrt(dim) / 729.0,
            245.0 / 486.0,
            265.0 - 100.0 * dim / 1458.0,
            25.0 / 729.0,
        ]
    )


def genz_malik(
    f: Callable,
    centers: NDArray[np.floating],
    halfwidths: NDArray[np.floating],
    vol: NDArray[np.floating],
):

    if centers.ndim != 3:
        raise ValueError("Invalid Centers Order. Expected 3, got ", centers.ndim)
    if halfwidths.ndim != 3:
        raise ValueError("Invalid widths Order. Expected 3, got ", halfwidths.ndim)
    if vol.ndim != 2:
        raise ValueError("Invalid volume Order. Expected 2, got ", vol.ndim)

    # lambda2 = sqrt(9/70), lambda4 = sqrt(9/10), lambda5 = sqrt(9/19)
    # ratio = (lambda2 ** 2) / (lambda4 ** 2)
    lambda2 = 0.35856858280031809199064515390793749545406372969943071
    lambda4 = 0.94868329805051379959966806332981556011586654179756505
    lambda5 = 0.68824720161168529772162873429362352512689535661564885
    ratio = 0.14285714285714285714285714285714285714285714285714281

    width_lambda2 = halfwidths * lambda2
    width_lambda4 = halfwidths * lambda4
    width_lambda5 = halfwidths * lambda5

    # p shape [ domain_dim, points, regions, events ]
    p = points.fullsym(centers, width_lambda2, width_lambda4, width_lambda5)
    dim = p.shape[0]
    d1 = points.num_k0k1(dim)
    d2 = points.num_k2(dim)
    d3 = d1 + d2

    # vals shape [range_dim, points, regions, events]
    vals = f(p)

    if vals.ndim == 3:
        vals = np.expand_dims(vals, 0)

    vc = vals[:, 0:1, ...]  # center integrand value. shape = [rdim, 1, reg, evt]
    v0 = vals[:, 1:d1:4, ...]  # [range_dim, domain_dim, regions, events]
    v1 = vals[:, 2:d1:4, ...]  # [range_dim, domain_dim, regions, events]
    v2 = vals[:, 3:d1:4, ...]  # [range_dim, domain_dim, regions, events]
    v3 = vals[:, 4:d1:4, ...]  # [range_dim, domain_dim, regions, events]

    fdiff = np.abs(v0 + v1 - 2 * vc - ratio * (v2 + v3 - 2 * vc))
    diff = np.sum(fdiff, axis=0)  # [domain_dim, regions, events]

    s2 = np.sum(v0 + v1, axis=1)  # [range_dim, regions, events]
    s3 = np.sum(v2 + v3, axis=1)  # [range_dim, regions, events]
    s4 = np.sum(vals[:, d1:d3, ...], axis=1)  # [range_dim, regions, events]
    s5 = np.sum(vals[:, d3:, ...], axis=1)  # [range_dim, regions, events]

    vc = np.squeeze(vc, 1)

    w = weights(dim)  # [5]
    wE = err_weights(dim)  # [4]

    result = vol * np.tensordot(
        w, (vc, s2, s3, s4, s5), (0, 0)
    )  # [5].[5,rd,r,e] = [rd,r,e]
    res5th = vol * np.tensordot(
        wE, (vc, s2, s3, s4), (0, 0)
    )  # [4].[4,rd,r,e] = [rd,r,e]

    err = np.abs(res5th - result)  # [range_dim, regions, events]

    # determine split dimension
    # df_scale = 10**dim
    # df = np.sum(err, axis=0) / (vol * df_scale) # [regions]

    split_dim = np.argmax(diff, axis=0)

    return result, err, split_dim


def gauss_kronrod(
    f: Callable,
    centers: NDArray[np.floating],
    halfwidths: NDArray[np.floating],
    vol: NDArray[np.floating],
):
    # # abscissae of the 15-point kronrod rule
    # xgk = np.array(
    #     [
    #         0.991455371120812639206854697526329,
    #         0.949107912342758524526189684047851,
    #         0.864864423359769072789712788640926,
    #         0.741531185599394439863864773280788,
    #         0.586087235467691130294144838258730,
    #         0.405845151377397166906606412076961,
    #         0.207784955007898467600689403773245,
    #         0.000000000000000000000000000000000,
    #     ]
    # )
    # # /* xgk[1], xgk[3], ... abscissae of the 7-point gauss rule.
    # #    xgk[0], xgk[2], ... to optimally extend the 7-point gauss rule */

    # # /* weights of the 7-point gauss rule */
    # wg = np.array(
    #     [
    #         0.129484966168869693270611432679082,
    #         0.279705391489276667901467771423780,
    #         0.381830050505118944950369775488975,
    #         0.417959183673469387755102040816327,
    #     ]
    # )
    # # /* weights of the 15-point kronrod rule */
    # wgk = np.array(
    #     [
    #         0.022935322010529224963732008058970,
    #         0.063092092629978553290700663189204,
    #         0.104790010322250183839876322541518,
    #         0.140653259715525918745189590510238,
    #         0.169004726639267902826583426598550,
    #         0.190350578064785409913256402421014,
    #         0.204432940075298892414161999234649,
    #         0.209482141084727828012999174891714,
    #     ]
    # )

    pass
