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
    sd = np.zeros((1, 1), dtype=int)

    c1, h1, v1 = region.split(c0, h0, v0, sd)

    assert c1.ndim == c0.ndim
    assert h1.ndim == h0.ndim
    assert v1.ndim == v0.ndim

    assert np.all(c1[:, : (c1.shape[1] // 2), ...] == 0.25)
    assert np.all(c1[:, (c1.shape[1] // 2) : -1, ...] == 0.75)
    assert np.all(h1 == 0.25)
    assert np.all(v1 == 0.5)
