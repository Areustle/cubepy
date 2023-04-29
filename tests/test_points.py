from __future__ import annotations

import numpy as np

from cubepy import points


def test_nums():
    n = np.arange(1, 10)

    assert np.all(points.num_k0k1(n) == [5, 9, 13, 17, 21, 25, 29, 33, 37])
    assert np.all(points.num_k2(n) == [0, 4, 12, 24, 40, 60, 84, 112, 144])
    assert np.all(points.num_k6(n) == [2, 4, 8, 16, 32, 64, 128, 256, 512])
    assert np.all(points.num_points(n) == [7, 17, 33, 57, 93, 149, 241, 401, 693])


def test_gm_pts():
    #############################################################
    c = np.zeros((2, 1, 1))
    h = np.ones((2, 1, 1))
    p = points.gm_pts(c, h, 2, 4, 5)
    assert len(p) == 2
    assert len(p[0]) == 17
    assert len(p[1]) == 17
    assert p[0].shape == (17, 1, 1)
    assert p[1].shape == (17, 1, 1)
    assert np.all(p[0][0].flat == [0])
    assert np.all(p[1][0].flat == [0])
    assert np.all(p[0][1:9].flat == [-2, 2, -4, 4, 0, 0, 0, 0])
    assert np.all(p[1][1:9].flat == [0, 0, 0, 0, -2, 2, -4, 4])
    assert np.all(p[0][9:13].flat == [-4, 4, -4, 4])
    assert np.all(p[1][9:13].flat == [-4, -4, 4, 4])
    assert np.all(p[0][13:].flat == [-5, -5, 5, 5])
    assert np.all(p[1][13:].flat == [-5, 5, -5, 5])

    c = np.zeros((3, 1, 1))
    h = np.ones((3, 1, 1))
    p = points.gm_pts(c, h, 2, 4, 5)
    assert len(p) == 3
    assert len(p[0]) == 33
    assert len(p[1]) == 33
    assert len(p[2]) == 33
    assert p[0].shape == (33, 1, 1)
    assert p[1].shape == (33, 1, 1)
    assert p[2].shape == (33, 1, 1)
    assert np.all(p[0][0].flat == [0])
    assert np.all(p[1][0].flat == [0])
    assert np.all(p[2][0].flat == [0])
    assert np.all(p[0][1:13].flat == [-2, 2, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.all(p[1][1:13].flat == [0, 0, 0, 0, -2, 2, -4, 4, 0, 0, 0, 0])
    assert np.all(p[2][1:13].flat == [0, 0, 0, 0, 0, 0, 0, 0, -2, 2, -4, 4])
    assert np.all(p[0][13:25].flat == [-4, 4, -4, 4, -4, 4, -4, 4, 0, 0, 0, 0])
    assert np.all(p[1][13:25].flat == [-4, -4, 4, 4, 0, 0, 0, 0, -4, 4, -4, 4])
    assert np.all(p[2][13:25].flat == [0, 0, 0, 0, -4, -4, 4, 4, -4, -4, 4, 4])
    assert np.all(p[0][25:].flat == [-5, -5, -5, -5, 5, 5, 5, 5])
    assert np.all(p[1][25:].flat == [-5, -5, 5, 5, -5, -5, 5, 5])
    assert np.all(p[2][25:].flat == [-5, 5, -5, 5, -5, 5, -5, 5])

    c = np.zeros((4, 1, 1))
    h = np.ones((4, 1, 1))
    p = points.gm_pts(c, h, 2, 4, 5)
    assert len(p) == 4
    assert len(p[0]) == 57
    assert len(p[1]) == 57
    assert len(p[2]) == 57
    assert len(p[3]) == 57
    assert p[0].shape == (57, 1, 1)
    assert p[1].shape == (57, 1, 1)
    assert p[2].shape == (57, 1, 1)
    assert p[3].shape == (57, 1, 1)
    assert np.all(
        p[0][:17].flat == [0, -2, 2, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    assert np.all(
        p[1][:17].flat == [0, 0, 0, 0, 0, -2, 2, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    assert np.all(
        p[2][:17].flat == [0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 2, -4, 4, 0, 0, 0, 0]
    )
    assert np.all(
        p[3][:17].flat == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 2, -4, 4]
    )

    # fmt: off
    assert np.all(p[0][17:41].flat == [
        -4, 4, -4, 4, -4, 4, -4, 4, -4, 4, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ])
    assert np.all(p[1][17:41].flat == [
        -4, -4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, -4, 4, -4, 4, -4, 4, -4, 4, 0, 0, 0, 0,
    ])
    assert np.all(p[2][17:41].flat == [
        0, 0, 0, 0, -4, -4, 4, 4, 0, 0, 0, 0, -4, -4, 4, 4, 0, 0, 0, 0, -4, 4, -4, 4,
    ])
    assert np.all(p[3][17:41].flat == [
        0, 0, 0, 0, 0, 0, 0, 0, -4, -4, 4, 4, 0, 0, 0, 0, -4, -4, 4, 4, -4, -4, 4, 4,
    ])
    # fmt: on

    assert np.all(
        p[0][41:].flat == [-5, -5, -5, -5, -5, -5, -5, -5, 5, 5, 5, 5, 5, 5, 5, 5]
    )
    assert np.all(
        p[1][41:].flat == [-5, -5, -5, -5, 5, 5, 5, 5, -5, -5, -5, -5, 5, 5, 5, 5]
    )
    assert np.all(
        p[2][41:].flat == [-5, -5, 5, 5, -5, -5, 5, 5, -5, -5, 5, 5, -5, -5, 5, 5]
    )
    assert np.all(
        p[3][41:].flat == [-5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5, -5, 5]
    )


def test_gm_evts():
    c = [np.zeros((1, 1)), np.zeros((1, 10))]
    # h = np.ones((2, 1, 1))
    h = [np.ones((1, 1)), np.ones((1, 10))]
    p = points.gm_pts(c, h, 2, 4, 5)
    assert len(p) == 2
    assert len(p[0]) == 17
    assert len(p[1]) == 17
    assert p[0].shape == (17, 1, 1)
    assert p[1].shape == (17, 1, 10)
    assert np.all(p[0][0].flat == [0])
    assert np.all(p[1][0].flat == np.tile([0], 10))
    assert np.all(p[0][1:9].flat == [-2, 2, -4, 4, 0, 0, 0, 0])
    for i in range(10):
        assert np.all(p[1][1:9, 0, i] == [0, 0, 0, 0, -2, 2, -4, 4])
    assert np.all(p[0][9:13].flat == [-4, 4, -4, 4])
    for i in range(10):
        assert np.all(p[1][9:13, 0, i] == [-4, -4, 4, 4])
    # assert np.all(p[1][9:13].flatten() == np.tile([-4, -4, 4, 4], (1, 10)).flatten())
    assert np.all(p[0][13:].flat == [-5, -5, 5, 5])
    # assert np.all(p[1][13:].flatten() == np.tile([-5, 5, -5, 5], (1, 10)).flatten())
    for i in range(10):
        assert np.all(p[1][13:, 0, i] == [-5, 5, -5, 5])


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
    assert np.all(gk[:7] == -gk[-1:7:-1])

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

    w = np.multiply.outer(xgk, h.ravel())

    assert gk[0, 0, ...] == c - w[0]
    assert gk[0, 1, ...] == c - w[1]
    assert gk[0, 2, ...] == c - w[2]
    assert gk[0, 3, ...] == c - w[3]
    assert gk[0, 4, ...] == c - w[4]
    assert gk[0, 5, ...] == c - w[5]
    assert gk[0, 6, ...] == c - w[6]
    assert gk[0, 7, ...] == c
    assert gk[0, 8, ...] == c + w[6]
    assert gk[0, 9, ...] == c + w[5]
    assert gk[0, 10, ...] == c + w[4]
    assert gk[0, 11, ...] == c + w[3]
    assert gk[0, 12, ...] == c + w[2]
    assert gk[0, 13, ...] == c + w[1]
    assert gk[0, 14, ...] == c + w[0]
