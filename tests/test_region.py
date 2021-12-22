import numpy as np

from cubepy import region


def test_region_simple():
    def tr(lo, hi):
        c, h, v = region.region(lo, hi)
        assert c.ndim == 3
        assert h.ndim == 3
        assert v.ndim == 2
        assert np.all(c == 0.5)
        assert np.all(h == 0.5)
        assert np.all(v == 1.0)

    tr(np.zeros(1), np.ones(1))
    tr(np.zeros((1)), np.ones((1)))
    tr(np.zeros((1, 1)), np.ones((1, 1)))
    tr(np.zeros((1, 1024)), np.ones((1, 1024)))
    tr(np.zeros((2, 1024)), np.ones((2, 1024)))
    tr(np.zeros((3, 1024)), np.ones((3, 1024)))
    tr(np.zeros((4, 1024)), np.ones((4, 1024)))
    tr(np.zeros((5, 1024)), np.ones((5, 1024)))
    tr(np.zeros((6, 1024)), np.ones((6, 1024)))
    tr(np.zeros((7, 1024)), np.ones((7, 1024)))
    tr(np.zeros((8, 1024)), np.ones((8, 1024)))


def test_region_complex_1d():

    lo = np.expand_dims(np.arange(1.0, 1e5), 0)
    lo -= 0.5e5
    hi = lo + 1.0

    c_true = lo + 0.5
    c_true = np.expand_dims(c_true, 1)
    h_true = 0.5
    v_true = 1.0

    c, h, v = region.region(lo, hi)

    assert c.shape == c_true.shape

    assert np.all(c == c_true)
    assert np.all(h == h_true)
    assert np.all(v == v_true)


def test_region_complex_2d():

    lo = np.arange(1.0, 1e5)
    lo -= 0.5e5
    lo = np.stack((lo, lo), 0)

    assert lo.ndim == 2
    assert lo.shape[0] == 2

    hi = lo + 1.0

    c_true = lo + 0.5
    c_true = np.expand_dims(c_true, 1)
    h_true = 0.5
    v_true = 1.0

    c, h, v = region.region(lo, hi)

    assert c.shape == c_true.shape

    assert np.all(c == c_true)
    assert np.all(h == h_true)
    assert np.all(v == v_true)


def test_region_random_4d():

    lo = np.random.uniform(low=-1e4, high=1e4, size=(4, int(1e5)))

    assert lo.ndim == 2
    assert lo.shape[0] == 4

    hi = lo + 1.0

    c_true = lo + 0.5
    c_true = np.expand_dims(c_true, 1)
    h_true = 0.5
    v_true = 1.0

    c, h, v = region.region(lo, hi)

    assert c.shape == c_true.shape

    assert np.all(c == c_true)
    assert np.all(h == h_true)
    assert np.all(v == v_true)


def test_split():
    c0 = np.ones((1, 1, 10)) * 0.5
    h0 = np.ones((1, 1, 10)) * 0.5
    v0 = np.ones((1, 1, 10))
    sd = np.zeros((1, 10), dtype=int)

    c1, h1, v1 = region.split(c0, h0, v0, sd)

    assert c1.ndim == c0.ndim
    assert h1.ndim == h0.ndim
    assert v1.ndim == v0.ndim

    assert np.all(c1[:, : (c1.shape[1] // 2), ...] == 0.25), f"{c1}"
    assert np.all(c1[:, (c1.shape[1] // 2) : -1, ...] == 0.75)
    assert np.all(h1 == 0.25)
    assert np.all(v1 == 0.5)


def test_split_complex():

    lo = np.arange(1.0, 5)
    lo -= 2.5
    lo = np.stack((lo, lo), 0)

    assert lo.ndim == 2
    assert lo.shape[0] == 2

    hi = lo + 1.0

    c0, h0, v0 = region.region(lo, hi)

    assert np.all(c0.shape == (2, 1, 4))

    c_ = np.copy(c0)
    h_ = np.copy(h0)
    v_ = np.copy(v0)

    sd = np.zeros((1, 4), dtype=int)
    sd[:, 1::2] = 1

    c1, h1, v1 = region.split(c0, h0, v0, sd)

    assert c1.ndim == c_.ndim
    assert h1.ndim == h0.ndim
    assert v1.ndim == v0.ndim

    assert c1.shape[0] == c_.shape[0]
    assert h1.shape[0] == h_.shape[0]
    assert v1.shape[1] == v_.shape[1]

    assert c1.shape[2] == c_.shape[2]
    assert h1.shape[2] == h_.shape[2]

    assert c1.shape[1] == c_.shape[1] * 2
    assert h1.shape[1] == h_.shape[1] * 2
    assert v1.shape[0] == v_.shape[0] * 2

    for i in range(c_.shape[1]):

        j = i + c_.shape[1]

        assert np.all(
            c_[0, i, 0] - 0.25 == c1[0, i, 0]
        ), f"{i} {j} \n {c_.squeeze()} \n {c1.squeeze()}"

        assert np.all(
            c_[0, i, 0] + 0.25 == c1[0, j, 0]
        ), f"{i} {j} {c_.squeeze()} {c1.squeeze()}"

        assert np.all(
            c_[1, i, 0] == c1[1, i, 0]
        ), f"{i} {j} {c_.squeeze()} {c1.squeeze()}"

        assert np.all(
            c_[1, i, 0] == c1[1, j, 0]
        ), f"{i} {j} {c_.squeeze()} {c1.squeeze()}"

        # assert c_[1, :, i] - 0.25 == c1[1, :, i]
        # assert c_[k, :, i] - 0.25 == c1[k, :, i]
        # assert c_[1, :, i] + 0.25 == c1[1, :, j]

    # assert np.all(c1[:, : (c1.shape[1] // 2), ...] == 0.25)
    # assert np.all(c1[:, (c1.shape[1] // 2) : -1, ...] == 0.75)
    # assert np.all(h1 == 0.25)
    # assert np.all(v1 == 0.5)
