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


def gauss_kronrod(f: Callable, centers: NPF, halfwidths: NPF) -> tuple[NPF, NPF, NPI]:

    # weights of the 7-point gauss rule
    wg = np.array(
        [
            0.129484966168869693270611432679082,
            0.279705391489276667901467771423780,
            0.381830050505118944950369775488975,
            0.417959183673469387755102040816327,
        ]
    )
    # weights of the 15-point kronrod rule
    wgk = np.array(
        [
            0.022935322010529224963732008058970,  # 0
            0.063092092629978553290700663189204,  # 1
            0.104790010322250183839876322541518,  # 2
            0.140653259715525918745189590510238,  # 3
            0.169004726639267902826583426598550,  # 4
            0.190350578064785409913256402421014,  # 5
            0.204432940075298892414161999234649,  # 6
            0.209482141084727828012999174891714,  # 7
        ]
    )

    # weights of the 15-point kronrod rule
    wgk_ro = np.array(
        [
            0.063092092629978553290700663189204,  # 1
            0.140653259715525918745189590510238,  # 3
            0.190350578064785409913256402421014,  # 5
            0.022935322010529224963732008058970,  # 0
            0.104790010322250183839876322541518,  # 2
            0.169004726639267902826583426598550,  # 4
            0.204432940075298892414161999234649,  # 6
            0.209482141084727828012999174891714,  # 7
        ]
    )

    # p shape [ points, regions, events ]
    # p shape [ 15,     regions, events ]
    p = points.gk_pts(centers, halfwidths)

    # # vals shape [range_dim, points, regions, events]
    vals = f(p)

    rg = vals[:, 0, ...] * wg[3]
    rk = vals[:, 0, ...] * wgk[7]
    ra = np.abs(rk)

    v = vals[:, 1:13:2, ...] + vals[:, 2::2, ...]
    va = np.abs(vals[:, 1:13:2, ...]) + np.abs(vals[:, 2::2, ...])

    # rg += np.sum(wg[0:3] * v[:, 0:3, ...], axis=1)
    rg += np.tensordot(wg[:3], v[:, :3, ...], (0, 1))

    # rk += np.sum(wgk[1:6:2] * v[:, 0:3, ...], axis=1)
    # ra += np.sum(wgk[1:6:2] * va[:, 0:3, ...], axis=1)
    # rk += np.sum(wgk[:7:2] * v[:, 3:, ...], axis=1)
    # ra += np.sum(wgk[:7:2] * va[:, 3:, ...], axis=1)

    rk += np.tensordot(wgk_ro, v, (0, 1))
    ra += np.tensordot(wgk_ro, va, (0, 1)) * halfwidths

    # error
    err = np.abs(rk - rg) * halfwidths

    mean = rk * 0.5

    rasc = wgk[7] * np.abs(vals[:, 0, ...] - mean)
    va = np.abs(vals[:, 1:13:2, ...] - mean) + np.abs(vals[:, 2::2, ...] - mean)
    rasc += np.tensordot(wgk_ro, va) * halfwidths

    msk = (rasc != 0) & (err != 0)
    scale = (200.0 * err[msk] / rasc[msk]) ** 1.5
    scale[scale < 1] = 1.0
    err[msk] = rasc[msk] * scale

    min_err = 50.0 * np.finfo(rk.dtype).eps
    msk = ra > (np.finfo(rk.dtype).min / min_err)
    err[msk & (min_err > err)] = min_err

    return rk * halfwidths, err, np.zeros_like(err, dtype=int)
