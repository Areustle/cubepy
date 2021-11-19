import itertools
import numpy as np


def num_k0k1(dim):
    return 1 + 4 * dim


def num_k2(dim):
    return 2 * dim * (dim - 1)


def num_k6(dim):
    return 1 << dim


def num_points(dim, full=False):
    if full:
        return (
            num_k0k1(dim) + num_k2(dim) + num_k6(dim),
            num_k0k1(dim),
            num_k0k1(dim) + num_k2(dim),
        )
    else:
        return num_k0k1(dim) + num_k2(dim) + num_k6(dim)


# p shape [ domain_dim, regions, points ]
def full_kn(c, numf):
    dim = c.shape[0]
    _shape = tuple(filter(None, (dim, numf(c.shape[0]), *c.shape[1:])))
    return np.full(_shape, c[:, None, ...])


# Center points in full sym(lambda2, 0, ... ,0) & full sym(lambda3=lambda4, 0, ..., 0)
def pts_k0k1(c, r1, r2, p=None):

    p = full_kn(c, num_k0k1) if p is None else p

    for i in range(p.shape[0]):
        j = 4 * i
        p[i, j+1] -= r1[i]
        p[i, j+2] += r1[i]
        p[i, j+3] -= r2[i]
        p[i, j+4] += r2[i]

    return p


# Center points for full sym(lambda4, lambda4, 0, ...,0)
def pts_k2(c, r, p=None):

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
def pts_k6(c, r, p=None):

    p = full_kn(c, num_k6) if p is None else p
    t = np.array(list(itertools.product([-1, 1], repeat=p.shape[0]))).T
    # # p += r[:, None, ...] * t[..., None, None]
    for i in range(p.shape[0]):
        p[i] += np.multiply.outer(t[i], r[i])
    return p


def fullsym(c, l2, l4, l5):

    p = full_kn(c, num_points)
    _, d1, d2 = num_points(p.shape[0], full=True)

    pts_k0k1(c, l2, l4, p=p[:, 0:d1, ...])
    pts_k2(c, l4, p=p[:, d1:d2, ...])
    pts_k6(c, l5, p=p[:, d2:, ...])

    return p
