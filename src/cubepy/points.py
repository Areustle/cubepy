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


@cache
def gm_weights(
    dim,
    alpha2: float = 0.35856858280031809199064515390793749545406372969943071,
    alpha4: float = 0.94868329805051379959966806332981556011586654179756505,
    alpha5: float = 0.68824720161168529772162873429362352512689535661564885,
    dtype=np.float64,
):
    # [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
    # integration Over an N-dimensional rectangular region", updated by Bernstein,
    # Espelid, Genz in "An Adaptive Algorithm for the Approximate Calculation of
    # Multiple Integrals"
    # alpha2 = 0.35856858280031809199064515390793749545406372969943071  # √(9/70)
    # alpha4 = 0.94868329805051379959966806332981556011586654179756505  # √(9/10)
    # alpha5 = 0.68824720161168529772162873429362352512689535661564885  # √(9/19)
    npts = num_points(dim)

    M = [np.zeros(npts) for _ in range(dim)]
    M1 = np.array([-alpha2, alpha2, -alpha4, alpha4], dtype=dtype)
    M2a = np.array([-alpha4, alpha4, -alpha4, alpha4], dtype=dtype)
    M2b = np.array([-alpha4, -alpha4, alpha4, alpha4], dtype=dtype)
    M6 = np.array([*product([-alpha5, alpha5], repeat=dim)], dtype=dtype).T
    k2A, k2B = k2indexes(dim)
    off6 = num_k0k1(dim) + num_k2(dim)
    for d in range(dim):
        # k1: Center points in fullsym(lambda2, 0, ... ,0) & fullsym(a4, 0, ..., 0)
        j = 4 * d
        M[d][1 + j : 5 + j] = M1
        # k2: Center points in full sym(lambda4, lambda4, ... ,0)
        M[d][k2A[d]] = M2a
        M[d][k2B[d]] = M2b
        # k6: Center points in full sym(lambda5, lambda5, ... ,lambda5)
        M[d][off6:] = M6[d]

    return M


def gm_pts(
    center,
    halfwidth,
    alpha2: float = 0.35856858280031809199064515390793749545406372969943071,
    alpha4: float = 0.94868329805051379959966806332981556011586654179756505,
    alpha5: float = 0.68824720161168529772162873429362352512689535661564885,
    pts=None,
):
    # [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
    # integration Over an N-dimensional rectangular region", updated by Bernstein,
    # Espelid, Genz in "An Adaptive Algorithm for the Approximate Calculation of
    # Multiple Integrals"
    # alpha2 = 0.35856858280031809199064515390793749545406372969943071  # √(9/70)
    # alpha4 = 0.94868329805051379959966806332981556011586654179756505  # √(9/10)
    # alpha5 = 0.68824720161168529772162873429362352512689535661564885  # √(9/19)
    #
    # {center, halfwidth}  domain_dim * [ regions, { 1 | nevts } ]

    dim = len(center)
    nreg = center[0].shape[0]
    npts = num_points(dim)
    dtype = center[0].dtype

    # p: domain_dim * [ points, regions, { 1 | nevts } ]
    if not pts:
        pts = [
            np.empty((npts, nreg * center[d].shape[1]), dtype=dtype) for d in range(dim)
        ]

    M = gm_weights(dim, alpha2, alpha4, alpha5)

    for d in range(dim):
        pts[d] = np.empty((npts, nreg * center[d].shape[1]), dtype=dtype)
        np.multiply.outer(
            M[d], halfwidth[d].reshape((nreg * halfwidth[d].shape[1])), out=pts[d]
        )
        np.add(pts[d], center[d].reshape((1, nreg * center[d].shape[1])), out=pts[d])
        pts[d] = np.reshape(pts[d], (npts, nreg, center[d].shape[1]))

    return pts


def gk_pts(center, halfwidth, pts=None):
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

    # {p}  [ points, regions, events ]
    p = center[0] + np.multiply.outer(nodes, halfwidth[0])

    # {p}  [ 1(domain_dim), points, regions, events ]
    # return [p]
    return p[None, ...]
