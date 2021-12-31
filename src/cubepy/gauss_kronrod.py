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


def gauss_kronrod(
    f: Callable, centers: NPF, halfwidths: NPF, v: NPF
) -> tuple[NPF, NPF, NPI]:

    # GK [7, 15] weights from
    # https://www.advanpix.com/2011/11/07/gauss-kronrod-quadrature-nodes-weights/
    gk_weights = np.array(
        [
            0.022935322010529224963732008058970,  # 0
            0.063092092629978553290700663189204,  # 1
            0.104790010322250183839876322541518,  # 2
            0.140653259715525918745189590510238,  # 3
            0.169004726639267902826583426598550,  # 4
            0.190350578064785409913256402421014,  # 5
            0.204432940075298892414161999234649,  # 6
            0.209482141084727828012999174891714,  # 7
            0.204432940075298892414161999234649,  # 6
            0.190350578064785409913256402421014,  # 5
            0.169004726639267902826583426598550,  # 4
            0.140653259715525918745189590510238,  # 3
            0.104790010322250183839876322541518,  # 2
            0.063092092629978553290700663189204,  # 1
            0.022935322010529224963732008058970,  # 0
        ]
    )

    gl_weights = np.array(
        [
            0.129484966168869693270611432679082,  # 0
            0.279705391489276667901467771423780,  # 1
            0.381830050505118944950369775488975,  # 2
            0.417959183673469387755102040816327,  # 3
            0.381830050505118944950369775488975,  # 2
            0.279705391489276667901467771423780,  # 1
            0.129484966168869693270611432679082,  # 0
        ]
    )

    # p.shape [ 15(points), ... ]
    p = points.gk_pts(centers, halfwidths)

    # vals.shape [ range_dim, points, ... ]
    vals = f(p)

    # r_.shape [ range_dim, ... ]
    rg: NPF = np.tensordot(gl_weights, vals[:, 1::2, ...], (0, 1))
    rk: NPF = np.tensordot(gk_weights, vals, (0, 1))

    # error
    err = halfwidths * np.abs(rk - rg)
    mean = 0.5 * rk
    I_tilde = halfwidths * np.tensordot(gk_weights, np.abs(vals - mean), (0, 1))

    mask = np.abs(I_tilde) > 1.0e-15
    scale = (200.0 * err[mask] / I_tilde[mask]) ** 1.5
    scale[scale > 1.0] = 1.0
    err[mask] = I_tilde[mask] * scale

    min_err = 50.0 * np.finfo(rk.dtype).eps
    rabs = halfwidths * np.tensordot(gk_weights, np.abs(vals), (0, 1))
    err[(rabs > (np.finfo(rk.dtype).tiny / min_err)) & (min_err > err)] = min_err

    return (rk * halfwidths), err, np.zeros(err.shape[-1], dtype=int)
