import numpy as np
from functools import cache

from . import point_gen

@cache
def weights(dim):
    return np.array([
            (12824. - 9120. * dim + 400. * np.sqrt(dim)) / 19683.0,
            980. / 6561.,
            (1820. - 400. * dim) / 19683.,
            200. / 19683.,
            6859. / 19683. / 2**dim
        ])

@cache
def err_weights(dim):
    return np.array([
            729. - 950. * dim + 50. * np.sqrt(dim) / 729.,
            245.0 / 486.0,
            265. - 100. * dim / 1458.,
            25.0 / 729.0,
        ])


def genz_malik(f, centers, halfwidths, vol):

    if centers.ndim != 3:
        raise ValueError("Invalid Centers Order. Expected 3, got", centers.ndim)
    if halfwidths.ndim != 3:
        raise ValueError("Invalid widths Order. Expected 3, got", halfwidths.ndim)
    if vol.ndim != 2:
        raise ValueError("Invalid volume Order. Expected 2, got", vol.ndim)

    # lambda2 = sqrt(9/70), lambda4 = sqrt(9/10), lambda5 = sqrt(9/19)
    # ratio = (lambda2 ** 2) / (lambda4 ** 2)
    lambda2 = 0.35856858280031809199064515390793749545406372969943071
    lambda4 = 0.94868329805051379959966806332981556011586654179756505
    lambda5 = 0.68824720161168529772162873429362352512689535661564885
    ratio   = 0.14285714285714285714285714285714285714285714285714281

    width_lambda2 = halfwidths * lambda2
    width_lambda4 = halfwidths * lambda4
    width_lambda5 = halfwidths * lambda5

    # p shape [ domain_dim, points, regions, events ]
    p = point_gen.fullsym(centers, width_lambda2, width_lambda4, width_lambda5)
    dim = p.shape[0]
    d1 = point_gen.num_k0k1(dim)
    d2 = point_gen.num_k2(dim)
    d3 = d1 + d2

    # vals shape [range_dim, points, regions, events]
    vals = f(p)

    if vals.ndim == 3:
        vals = np.expand_dims(vals, 0)

    vc = vals[:, 0:1   , ...] # center integrand value. shape = [rdim, 1, reg, evt]
    v0 = vals[:, 1:d1:4, ...] # [range_dim, domain_dim, regions, events]
    v1 = vals[:, 2:d1:4, ...] # [range_dim, domain_dim, regions, events]
    v2 = vals[:, 3:d1:4, ...] # [range_dim, domain_dim, regions, events]
    v3 = vals[:, 4:d1:4, ...] # [range_dim, domain_dim, regions, events]

    fdiff = np.abs(v0 + v1 - 2 * vc - ratio * (v2 + v3 - 2 * vc))
    diff = np.sum(fdiff, axis=0) # [domain_dim, regions, events]

    s2 = np.sum(v0 + v1, axis=1) # [range_dim, regions, events]
    s3 = np.sum(v2 + v3, axis=1) # [range_dim, regions, events]
    s4 = np.sum(vals[:, d1:d3, ...], axis=1) # [range_dim, regions, events]
    s5 = np.sum(vals[:, d3:, ...], axis=1) # [range_dim, regions, events]

    vc = np.squeeze(vc, 1)

    w = weights(dim) # [5]
    wE = err_weights(dim) # [4]

    result = vol * np.tensordot(w, (vc, s2, s3, s4, s5), (0,0)) # [5].[5,rd,r,e] = [rd,r,e]
    res5th = vol * np.tensordot(wE, (vc, s2, s3, s4), (0,0))    # [4].[4,rd,r,e] = [rd,r,e]

    err = np.abs(res5th - result) # [range_dim, regions, events]

    # determine split dimension
    # df_scale = 10**dim
    # df = np.sum(err, axis=0) / (vol * df_scale) # [regions]

    split_dim = np.argmax(diff, axis=0)

    return result, err, split_dim
