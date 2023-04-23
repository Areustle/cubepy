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
from itertools import combinations, product
from typing import List, Tuple

import numpy as np

from .type_aliases import NPF, NPI, NPT


def num_k0k1(dim: int) -> int:
    return 1 + 4 * dim


def num_k2(dim: int) -> int:
    return 2 * dim * (dim - 1)


def num_k6(dim: int) -> int:
    return 1 << dim


def num_points(dim: int) -> int:
    return num_k0k1(dim) + num_k2(dim) + num_k6(dim)


def num_points_full(dim: int) -> tuple[int, ...]:
    return (
        num_k0k1(dim) + num_k2(dim) + num_k6(dim),
        num_k0k1(dim),
        num_k0k1(dim) + num_k2(dim),
    )


@cache
def k2indexes(ndim: int) -> Tuple[List[NPI], List[NPI]]:
    # fmt: off
    offset = num_k0k1(ndim)
    A: List[List[int]] = [[] for _ in range(ndim)]
    B: List[List[int]] = [[] for _ in range(ndim)]
    # fmt: on
    for i, (a, b) in enumerate(combinations(range(ndim), 2)):
        A[a].append(i)
        B[b].append(i)

    return (
        [4 * np.array(x, dtype=int)[:, None] + np.arange(4) + offset for x in A],
        [4 * np.array(x, dtype=int)[:, None] + np.arange(4) + offset for x in B],
    )


def gm_pts(
    center: NPF,
    halfwidth: NPF,
    alpha2: float = 0.35856858280031809199064515390793749545406372969943071,
    alpha4: float = 0.94868329805051379959966806332981556011586654179756505,
    alpha5: float = 0.68824720161168529772162873429362352512689535661564885,
):
    # [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
    # integration Over an N-dimensional rectangular region", updated by Bernstein,
    # Espelid, Genz in "An Adaptive Algorithm for the Approximate Calculation of
    # Multiple Integrals"
    # alpha2 = 0.35856858280031809199064515390793749545406372969943071  # √(9/70)
    # alpha4 = 0.94868329805051379959966806332981556011586654179756505  # √(9/10)
    # alpha5 = 0.68824720161168529772162873429362352512689535661564885  # √(9/19)
    ndim = center.shape[0]
    npts = num_points(ndim)
    dtype = center.dtype

    #  [ domain_dim, points, regions ]
    p: NPF = np.tile(center[:, None, :], (1, npts, 1))

    # k1
    # Center points in fullsym(lambda2, 0, ... ,0) & fullsym(a4, 0, ..., 0)
    k1T0 = np.tile(np.arange(ndim)[:, None], (1, 4))
    k1T1 = np.arange(1, num_k0k1(ndim)).reshape((ndim, 4))
    M1 = np.array([-alpha2, alpha2, -alpha4, alpha4], dtype=dtype)
    p[k1T0, k1T1] += M1[None, :, None] * halfwidth[:, None]

    # k2
    # Center points in full sym(lambda4, lambda4, ... ,0)
    k2A, k2B = k2indexes(ndim)
    M3 = np.array([-alpha4, alpha4, -alpha4, alpha4], dtype=dtype)[:, None]
    M4 = np.array([-alpha4, -alpha4, alpha4, alpha4], dtype=dtype)[:, None]
    for d in range(ndim):
        p[d, k2A[d]] += M3 * halfwidth[d]
        p[d, k2B[d]] += M4 * halfwidth[d]

    # k6
    # Center points in full sym(lambda5, lambda5, ... ,lambda5)
    off6 = num_k0k1(ndim) + num_k2(ndim)
    M5 = np.array([*product([-alpha5, alpha5], repeat=ndim)], dtype=dtype).T[..., None]
    p[:, off6:] += M5 * halfwidth[:, None]

    return p


def gk_pts(c: NPF, h: NPF) -> NPF:
    # GK [7, 15] node points from
    # https://www.advanpix.com/2011/11/07/gauss-kronrod-quadrature-nodes-weights/
    nodes = np.array(
        [
            -0.991455371120812639206854697526329,  # 0
            -0.949107912342758524526189684047851,  # 1
            -0.864864423359769072789712788640926,  # 2
            -0.741531185599394439863864773280788,  # 3
            -0.586087235467691130294144838258730,  # 4
            -0.405845151377397166906606412076961,  # 5
            -0.207784955007898467600689403773245,  # 6
            0.000000000000000000000000000000000,  # 7
            0.207784955007898467600689403773245,  # 8
            0.405845151377397166906606412076961,  # 9
            0.586087235467691130294144838258730,  # 10
            0.741531185599394439863864773280788,  # 11
            0.864864423359769072789712788640926,  # 12
            0.949107912342758524526189684047851,  # 13
            0.991455371120812639206854697526329,  # 14
        ]
    )

    # {c, h}  [ regions, events ]
    c = np.squeeze(c, 0)
    h = np.squeeze(h, 0)

    # {p}  [ points, regions, events ]
    p = c + np.multiply.outer(nodes, h)

    # {p}  [ 1(domain_dim), points, regions, events ]
    return p
