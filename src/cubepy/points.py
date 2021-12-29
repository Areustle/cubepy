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

from itertools import product
from typing import Callable

import numpy as np

from .type_aliases import NPF


def num_k0k1(dim: int) -> int:
    return 1 + 4 * dim


def num_k2(dim: int) -> int:
    return 2 * dim * (dim - 1)


def num_k6(dim: int) -> int:
    return 1 << dim


def num_points(dim: int) -> int:
    return num_k0k1(dim) + num_k2(dim) + num_k6(dim)


def num_points_full(dim: int) -> tuple[int, ...]:
    return tuple(
        [
            num_k0k1(dim) + num_k2(dim) + num_k6(dim),
            num_k0k1(dim),
            num_k0k1(dim) + num_k2(dim),
        ]
    )


# p shape [ domain_dim, points, ...]
def full_kn(c: NPF, numf: Callable) -> NPF:
    dim = c.shape[0]
    s = [dim, numf(c.shape[0]), *c.shape[1:]]
    _shape: tuple = tuple(filter(None, s))
    return np.full(_shape, c[:, None, ...])


# Center points in full sym(lambda2, 0, ... ,0) & full sym(lambda3=lambda4, 0, ..., 0)
def pts_k0k1(c: NPF, r1: NPF, r2: NPF, p: NPF | None = None) -> NPF:

    p = full_kn(c, num_k0k1) if p is None else p

    for i in range(p.shape[0]):
        j = 4 * i
        p[i, j + 1] -= r1[i]
        p[i, j + 2] += r1[i]
        p[i, j + 3] -= r2[i]
        p[i, j + 4] += r2[i]

    return p


# Center points for full sym(lambda4, lambda4, 0, ...,0)
def pts_k2(c: NPF, r: NPF, p: NPF | None = None) -> NPF:

    p = full_kn(c, num_k2) if p is None else p
    dim = p.shape[0]
    k = 0

    for i in range(dim - 1):
        for j in range(i + 1, dim):
            p[i, k] -= r[i]
            p[j, k] -= r[j]
            k += 1
            p[i, k] += r[i]
            p[j, k] -= r[j]
            k += 1
            p[i, k] -= r[i]
            p[j, k] += r[j]
            k += 1
            p[i, k] += r[i]
            p[j, k] += r[j]
            k += 1

    return p


# Center points for full sym(lambda5, ...,  lambda5)
def pts_k6(c: NPF, r: NPF, p: NPF | None = None) -> NPF:

    p = full_kn(c, num_k6) if p is None else p
    t = np.array(list(product([-1, 1], repeat=p.shape[0]))).T
    # # p += r[:, None, ...] * t[..., None, None]
    for i in range(p.shape[0]):
        p[i] += np.multiply.outer(t[i], r[i])
    return p


def fullsym(c: NPF, l2: NPF, l4: NPF, l5: NPF) -> NPF:

    p: NPF = full_kn(c, num_points)
    _, d1, d2 = num_points_full(p.shape[0])

    pts_k0k1(c, l2, l4, p=p[:, 0:d1, ...])
    pts_k2(c, l4, p=p[:, d1:d2, ...])
    pts_k6(c, l5, p=p[:, d2:, ...])

    return p


def gk_pts(c: NPF, h: NPF, p: NPF | None = None) -> NPF:

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
    return np.expand_dims(p, 0)


def gm_pts(ndim):
    pass
