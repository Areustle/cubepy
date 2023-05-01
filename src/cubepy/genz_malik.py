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
    k1T0 = np.tile(np.arange(dim)[:, None], (1, 4))
    k1T1 = np.arange(1, points.num_k0k1(dim)).reshape((dim, 4))
    a[k1T0, k1T1] = [1.0, 1.0, -ratio, -ratio]

    return a


def rule_split_dim(vals, err, halfwidth, volume):
    # :::::::::::::: Shapes ::::::::::::::::
    # vals [ points, regions, events ]
    # err [ regions, events ]
    # halfwidth domain_dim * [ regions, { 1 | nevts } ]
    # volume [ events ]

    dim = len(halfwidth)
    npts, nreg, nevt = vals.shape
    d1 = points.num_k0k1(dim)

    # Compute the 4th divided difference to determine the dimension on which to split.
    s1 = (npts, nreg * nevt)
    s3 = (dim, nreg, nevt)

    # [ domain_dim, regions, events ] = [ domain_dim, d1 ] @ [ d1, regions, events ]
    # fdiff = div_diff_weights(dim) @ vals.reshape(s1)[:d1, ...]
    # [ domain_dim, regions ]
    diff = np.linalg.norm(
        (div_diff_weights(dim) @ vals.reshape(s1)[:d1, ...]).reshape(s3), ord=1, axis=-1
    )

    # [ regions ]
    split_dim = np.argmax(diff, axis=0)
    widest_dim = np.argmax(np.asarray([np.amax(h, axis=1) for h in halfwidth]), axis=0)

    # [ domain_dim, regions ]
    delta = np.abs(diff[split_dim, np.arange(nreg)] - diff[widest_dim, np.arange(nreg)])
    df = np.sum(err * (volume[None, :] * 10 ** (-dim)), axis=1)  # [ regions ]

    too_close = delta <= df
    split_dim[too_close] = widest_dim[too_close]
    return split_dim


@cache
def error_weights_vec(dim: int) -> NPF:
    d1 = points.num_k0k1(dim)
    d2 = points.num_k2(dim)
    d3 = d1 + d2
    wE = error_weights(dim)
    a = np.zeros(d3)
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
    w = rule_weights(dim)
    a = np.zeros(points.num_points(dim))
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


def genz_malik(f: Callable, pts, halfwidth, volume) -> tuple[NPF, NPF, NPI]:
    # [7, 5] FS rule weights from Genz, Malik: "An adaptive algorithm for numerical
    # integration Over an N-dimensional rectangular region", updated by Bernstein,
    # Espelid, Genz in "An Adaptive Algorithm for the Approximate Calculation of
    # Multiple Integrals"

    # p shape domain_dim * [ points, regions, { 1 | nevts } ]
    dim = len(pts)
    npts = points.num_points(dim)
    nreg = halfwidth[0].shape[0]

    # Save shape then reshape to [domain_dim, (points*regions), 1] before passing to f
    pts = [np.reshape(p, (npts * nreg, p.shape[-1])) for p in pts]
    # vals shape [ points * regions, events  ] ==> [ points, regions, events ]
    vals = f(pts)
    # any eventwise operations in the integrand will automatically be (N, 1) * (M)
    # or (N, M) * (M) operations, so should return events in the trailing dimension.
    nevt = vals.size // (nreg * npts)

    # Reshape shapes to conform to matmul shape requirements.
    s0 = (npts, nreg, nevt)
    s1 = (npts, nreg * nevt)
    s2 = (2, nreg, nevt)

    vals = np.reshape(vals, s0)
    # vals = np.reshape(vals, s1)
    w = rule_error_weights(dim)

    rpack = (w @ vals.reshape(s1)).reshape(s2) * volume
    rpack[1] = np.abs(np.diff(rpack, axis=0))  # [ regions, events ]
    result, err = rpack
    split_dim = rule_split_dim(vals.reshape(s0), err, halfwidth, volume)  # [ regions ]

    # [regions, events] [ regions, events ] [ regions ]
    return result, err, split_dim
