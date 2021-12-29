from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np

from cubepy import points
from cubepy.type_aliases import NPF


def test_nums():
    n = np.arange(1, 10)

    assert np.all(points.num_k0k1(n) == [5, 9, 13, 17, 21, 25, 29, 33, 37])
    assert np.all(points.num_k2(n) == [0, 4, 12, 24, 40, 60, 84, 112, 144])
    assert np.all(points.num_k6(n) == [2, 4, 8, 16, 32, 64, 128, 256, 512])
    assert np.all(points.num_points(n) == [7, 17, 33, 57, 93, 149, 241, 401, 693])


def test_full_kn():

    for i in range(1, 10):

        c = np.ones((i))

        n01 = points.num_k0k1(i)
        n2 = points.num_k2(i)
        n6 = points.num_k6(i)

        assert np.all(points.full_kn(c, points.num_k0k1) == np.ones((i, n01)))
        assert np.all(points.full_kn(c, points.num_k2) == np.ones((i, n2)))
        assert np.all(points.full_kn(c, points.num_k6) == np.ones((i, n6)))


def test_fulls():

    c = np.zeros((4))
    h = np.ones((4))

    # fmt: off
    x01 = [
        [0, -1, 1, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, -1, 1, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, -2, 2, 0, 0, 0, 0, ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, -2, 2, ],
    ]
    # fmt: on

    assert np.all(x01 == points.pts_k0k1(c, h, 2.0 * h))

    # fmt: off
    x2 = [
        [-2, 2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-2, -2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, -2, 2, -2, 2, -2, 2, -2, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, -2, -2, 2, 2, 0, 0, 0, 0, -2, -2, 2, 2, 0, 0, 0, 0, -2, 2, -2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, -2, -2, 2, 2, 0, 0, 0, 0, -2, -2, 2, 2, -2, -2, 2, 2],
    ]
    # fmt: on

    assert np.all(x2 == points.pts_k2(c, 2.0 * h))

    # fmt: off
    x6 = [
        [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, ],
        [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, ],
        [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, ],
        [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, ],
    ]
    # fmt: on

    assert np.all(x6 == points.pts_k6(c, h))


def test_pts_k0k1():

    for i in range(1, 10):

        n = points.num_k0k1(i)

        c = np.zeros((i))
        h = np.ones((i))
        assert points.pts_k0k1(c, h, 2.0 * h).shape == (i, n)

        c = np.zeros((i, 5))
        h = np.ones((i, 5))
        assert points.pts_k0k1(c, h, 2.0 * h).shape == (i, n, 5)

        c = np.zeros((i, 5, 6))
        h = np.ones((i, 5, 6))
        assert points.pts_k0k1(c, h, 2.0 * h).shape == (i, n, 5, 6)

        c = np.zeros((i, 5, 6, 7))
        h = np.ones((i, 5, 6, 7))
        assert points.pts_k0k1(c, h, 2.0 * h).shape == (i, n, 5, 6, 7)


def test_pts_k2():

    for i in range(1, 10):

        n = points.num_k2(i)

        # [dim]
        c = np.zeros((i))
        h = np.ones((i))
        assert points.pts_k2(c, h).shape == tuple(filter(None, (i, n)))

        # [dim, regions]
        c = np.zeros((i, 5))
        h = np.ones((i, 5))
        assert points.pts_k2(c, h).shape == tuple(filter(None, (i, n, 5)))

        # [dim, regions, events]
        c = np.zeros((i, 5, 6))
        h = np.ones((i, 5, 6))
        assert points.pts_k2(c, h).shape == tuple(filter(None, (i, n, 5, 6)))

        # [dim, regions, events, xxx]
        c = np.zeros((i, 5, 6, 7))
        h = np.ones((i, 5, 6, 7))
        assert points.pts_k2(c, h).shape == tuple(filter(None, (i, n, 5, 6, 7)))


def test_pts_k6():

    for i in range(1, 10):

        n = points.num_k6(i)

        c = np.zeros((i))
        h = np.ones((i))
        assert points.pts_k6(c, h).shape == (i, n)

        c = np.zeros((i, 5))
        h = np.ones((i, 5))
        assert points.pts_k6(c, h).shape == (i, n, 5)

        c = np.zeros((i, 5, 6))
        h = np.ones((i, 5, 6))
        assert points.pts_k6(c, h).shape == (i, n, 5, 6)

        c = np.zeros((i, 5, 6, 7))
        h = np.ones((i, 5, 6, 7))
        assert points.pts_k6(c, h).shape == (i, n, 5, 6, 7)


def test_fullsym():

    for i in range(1, 10):

        n = points.num_points(i)

        c = np.zeros((i))
        h = np.ones((i))
        assert points.fullsym(c, h, 2.0 * h, h).shape == (i, n)

        c = np.zeros((i, 5))
        h = np.ones((i, 5))
        assert points.fullsym(c, h, 2.0 * h, h).shape == (i, n, 5)

        c = np.zeros((i, 5, 6))
        h = np.ones((i, 5, 6))
        assert points.fullsym(c, h, 2.0 * h, h).shape == (i, n, 5, 6)

        c = np.zeros((i, 5, 6, 7))
        h = np.ones((i, 5, 6, 7))
        assert points.fullsym(c, h, 2.0 * h, h).shape == (i, n, 5, 6, 7)


def test_gk_pts():

    c = np.zeros((1, 1, 1))
    h = np.ones((1, 1, 1)) * 0.5

    gk = points.gk_pts(c, h)

    assert gk.ndim == 4
    assert gk.shape[0] == 1
    assert gk.shape[1] == 15
    assert gk.shape[2] == 1
    assert gk.shape[3] == 1
    assert not np.all(gk == 0)
    assert np.all(gk[1::2] == -gk[2::2])

    xgk = np.array(
        [
            0.991455371120812639206854697526329,
            0.949107912342758524526189684047851,
            0.864864423359769072789712788640926,
            0.741531185599394439863864773280788,
            0.586087235467691130294144838258730,
            0.405845151377397166906606412076961,
            0.207784955007898467600689403773245,
        ]
    )

    w = np.multiply.outer(xgk, np.squeeze(h, 0))

    assert gk[:, 0, ...] == c - w[0, ...]
    assert gk[:, 1, ...] == c - w[1, ...]
    assert gk[:, 2, ...] == c - w[2, ...]
    assert gk[:, 3, ...] == c - w[3, ...]
    assert gk[:, 4, ...] == c - w[4, ...]
    assert gk[:, 5, ...] == c - w[5, ...]
    assert gk[:, 6, ...] == c - w[6, ...]
    assert gk[:, 7, ...] == c
    assert gk[:, 8, ...] == c + w[6, ...]
    assert gk[:, 9, ...] == c + w[5, ...]
    assert gk[:, 10, ...] == c + w[4, ...]
    assert gk[:, 11, ...] == c + w[3, ...]
    assert gk[:, 12, ...] == c + w[2, ...]
    assert gk[:, 13, ...] == c + w[1, ...]
    assert gk[:, 14, ...] == c + w[0, ...]


# Center points in full sym(lambda2, 0, ... ,0) & full sym(lambda3=lambda4, 0, ..., 0)
def pts_k0k1(p: NPF, r1: float, r2: float) -> NPF:

    for i in range(p.shape[0]):
        j = 4 * i
        p[i, j + 1 : j + 5] = [-r1, r1, -r2, r2]

    return p


# Center points for full sym(lambda4, lambda4, 0, ...,0)
def pts_k2(p: NPF, r: float) -> NPF:

    dim = p.shape[0]
    k = 0

    for i in range(dim - 1):
        for j in range(i + 1, dim):
            p[i, k] -= r
            p[j, k] -= r
            k += 1
            p[i, k] += r
            p[j, k] -= r
            k += 1
            p[i, k] -= r
            p[j, k] += r
            k += 1
            p[i, k] += r
            p[j, k] += r
            k += 1

    return p


# Center points for full sym(lambda5, ...,  lambda5)
def pts_k6(p: NPF, r: float) -> NPF:
    t = np.array(list(product([-1, 1], repeat=p.shape[0]))).T
    p += r * t
    return p


def fullsym(domain_dim: int, l2: float, l4: float, l5: float):
    p = np.zeros((domain_dim, points.num_points(domain_dim)))
    _, d1, d2 = points.num_points_full(domain_dim)
    pts_k0k1(p[:, :d1], l2, l4)
    pts_k2(p[:, d1:d2], l4)
    pts_k6(p[:, d2:], l5)

    return p


def new_points(c, h, l2, l4, l5):
    p = fullsym(c.shape[0], l2, l4, l5)
    hp = p[..., None] * h[:, None]
    chp = c[:, None, :] + hp
    return chp


def old_points(c, h, l2, l4, l5):
    p = points.fullsym(c, h * l2, h * l4, h * l5)
    return p


def run():
    alpha2 = 0.35856858280031809199064515390793749545406372969943071  # √(9/70)
    alpha4 = 0.94868329805051379959966806332981556011586654179756505  # √(9/10)
    alpha5 = 0.68824720161168529772162873429362352512689535661564885  # √(9/19)
    c = np.zeros((8, 100000))
    h = 9.0 * np.ones((8, 100000))
    new_points(c, h, alpha2, alpha4, alpha5)
    old_points(c, h, alpha2, alpha4, alpha5)


if __name__ == "__main__":
    np.set_printoptions(linewidth=512)
    run()

    # x = 2.0
    # y = 6.0
    # z = 8.0

    # p = fullsym(c.shape[0], x, y, z)
    # print(p)

    # hp = np.einsum("dp,dr->dpr", p, h)
    # chp = c[:, None, :] + hp
    # print(chp)
