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


def shape_k0k1(x):
    dim = x.shape[-1]
    return tuple(filter(None, (x.shape[0:-1], num_k0k1(dim), dim)))


def shape_k2(x):
    dim = x.shape[-1]
    return tuple(filter(None, (x.shape[0:-1], num_k2(dim), dim)))


def shape_k6(x):
    dim = x.shape[-1]
    return tuple(filter(None, (x.shape[0:-1], num_k6(dim), dim)))


# Center points in (lambda2,0,...,0) and (lambda3=lambda4, 0, ..., 0).
def pts_k0k1(c, r1, r2, p=None):
    if p is None:
        _shape = shape_k0k1(c)
        p = np.full(_shape, c)

    dim = p.shape[-1]
    for i in range(dim):
        p[..., 1 + 4 * i : 5 + 4 * i, i] += [-r1, +r1, -r2, +r2]

    return p


# Center points for (lambda4, lambda4, 0, ...,0)
def pts_k2(c, r, p=None):

    if p is None:
        _shape = shape_k2(c)
        p = np.full(_shape, c)

    dim = p.shape[-1]

    k = 0
    for i in range(dim - 1):
        for j in range(i + 1, dim):
            p[..., k, i] -= r
            p[..., k, j] -= r
            k += 1
            p[..., k, i] += r
            p[..., k, j] -= r
            k += 1
            p[..., k, i] -= r
            p[..., k, j] += r
            k += 1
            p[..., k, i] += r
            p[..., k, j] += r
            k += 1

    return p


# Center points for (lambda5, ...,  lambda5)
def pts_k6(c, r, p=None):

    if p is None:
        _shape = shape_k6(c)
        p = np.full(_shape, c)

    dim = p.shape[-1]

    p += r * np.array(list(itertools.product([-1, 1], repeat=dim)))
    return p


def fullsym(c, l1=1, l2=1, l4=1, l5=1):
    dim = c.shape[-1]
    npts, k1, k2 = num_points(dim, full=True)
    _shape = tuple(filter(None, (c.shape[0:-1], npts, dim)))

    p = np.full(_shape, c)
    pts_k0k1(c, l1, l2, p=p[..., 0:k1, :])
    pts_k2(c, l4, p=p[..., k1:k2, :])
    pts_k6(c, l5, p=p[..., k2:, :])

    return p


if __name__ == "__main__":

    print(pts_k0k1(np.zeros(4), 1, 2))
    print(pts_k2(np.zeros(4), 1))
    print(pts_k6(np.zeros(4), 1))
    print("======================")
    print(fullsym(np.zeros(4)))
    v = np.concatenate(
        (pts_k0k1(np.zeros(4), 1, 2), pts_k2(np.zeros(4), 1), pts_k6(np.zeros(4), 1))
    )
    print(v.shape)
    print(np.allclose(fullsym(np.zeros(4), 1, 2, 1, 1), v))
