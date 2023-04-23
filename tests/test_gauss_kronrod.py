import numpy as np

import cubepy as cp


def test_quadratic():
    def quadratic(x):
        return x**2

    def exact_quadratic(a, b):
        def exact(x):
            return x**3 / 3.0

        return exact(b) - exact(a)

    center, hwidth, vol = cp.region.region(np.asarray([-1.0]), np.asarray([1.0]))
    assert center.shape == (1, 1)
    assert hwidth.shape == (1, 1)
    p = cp.points.gk_pts(center, hwidth)
    assert p.shape == (15, 1)
    vals = quadratic(p)
    assert vals.shape == (15, 1)

    value, error, split_dim = cp.gauss_kronrod.gauss_kronrod(
        quadratic, center, hwidth, vol
    )
    #
    assert np.allclose(value, exact_quadratic(-1.0, 1.0))
    assert np.all(error < 1e-13)
    assert np.all(split_dim == 0)

    center, hwidth, vol = cp.region.region(np.asarray([-np.pi]), np.asarray([np.pi]))
    value, error, split_dim = cp.gauss_kronrod.gauss_kronrod(
        quadratic, center, hwidth, vol
    )

    assert np.allclose(value, exact_quadratic(-np.pi, np.pi))
    assert np.all(error < 1e-13)
    assert np.all(split_dim == 0)

    # low = -1.0  # np.random.uniform(-100, 100, size=(1, 100))
    # high = 1.0  # low + np.random.uniform(0, 100, size=(1, 100))
    # def vector_quadratic(x, off, scale):
    #     return (x + off)**2
    #
    # low, high, _ = cp.input.parse_input(
    #
    # c, h, v = cp.region.region(low, high)
    #
    # val, err, spd = cp.gauss_kronrod.gauss_kronrod(quadratic, c, h, v)
    # assert np.allclose(val, exact_quadratic(low, high))
    # assert np.all(err < 1e-13)
    # assert np.all(spd == 0)


def test_polynomial():
    def poly(x):
        return 2.0 * np.pi * x**4 - np.e * x**3 + 3.0 * x**2 - 4.0 * x + 8.0

    def exact_poly(a, b):
        def exact(x):
            return (
                (2.0 * np.pi / 5.0) * x**5
                - np.e / 4.0 * x**4
                + x**3
                - 2.0 * x**2
                + 8.0 * x
            )

        return exact(b) - exact(a)

    center, hwidth, vol = cp.region.region(np.asarray([-1.0]), np.asarray([1.0]))
    value, error, split_dim = cp.gauss_kronrod.gauss_kronrod(poly, center, hwidth, vol)

    assert np.allclose(value, exact_poly(-1.0, 1.0))
    assert np.all(error < 1e-13)
    assert np.all(split_dim == 0)

    center, hwidth, vol = cp.region.region(np.asarray([-np.pi]), np.asarray([np.pi]))
    value, error, split_dim = cp.gauss_kronrod.gauss_kronrod(poly, center, hwidth, vol)

    assert np.allclose(value, exact_poly(-np.pi, np.pi))
    assert np.all(error < 1e-13)
    assert np.all(split_dim == 0)

    # low = np.random.uniform(-100, 100, size=(1, 100))
    # high = low + np.random.uniform(0, 100, size=(1, 100))
    #
    # c, h, v = cp.region.region(low, high)
    #
    # val, err, spd = cp.gauss_kronrod.gauss_kronrod(poly, c, h, v)
    # assert np.allclose(val, exact_poly(low, high))
    # assert np.all(err < 1e-8)
    # assert np.all(spd == 0)


def test_high_polynomial():
    def poly(x):
        return 2.0 * np.pi * x**20 - np.e * x**3 + 3.0 * x**2 - 4.0 * x + 8.0

    def exact_poly(a, b):
        def exact(x):
            return (
                (2.0 * np.pi / 21.0) * x**21
                - np.e / 4.0 * x**4
                + x**3
                - 2.0 * x**2
                + 8.0 * x
            )

        return exact(b) - exact(a)

    center, hwidth, vol = cp.region.region(np.asarray([-1.0]), np.asarray([1.0]))
    value, error, split_dim = cp.gauss_kronrod.gauss_kronrod(poly, center, hwidth, vol)

    assert np.allclose(value, exact_poly(-1.0, 1.0))
    assert value.shape == error.shape
    # assert np.all(error < 1e-13)
    assert np.all(split_dim == 0)

    center, hwidth, vol = cp.region.region(np.asarray([-np.pi]), np.asarray([np.pi]))
    value, error, split_dim = cp.gauss_kronrod.gauss_kronrod(poly, center, hwidth, vol)

    assert np.allclose(value, exact_poly(-np.pi, np.pi))
    assert value.shape == error.shape
    # assert np.all(error < 1e-13)
    assert np.all(split_dim == 0)

    # low = np.random.uniform(-100, 100, size=(1, 100))
    # high = low + np.random.uniform(0, 100, size=(1, 100))
    #
    # c, h, v = cp.region.region(low, high)
    #
    # value, error, split_dim = cp.gauss_kronrod.gauss_kronrod(poly, c, h, v)
    # assert np.allclose(value, exact_poly(low, high))
    # assert value.shape == error.shape
    # # assert np.all(err < 1e-8)
    # assert np.all(split_dim == 0)
