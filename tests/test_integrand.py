import numpy as np

import cubepy as cp

# import pytest


def test_quadratic():
    def quadratic(x):
        return x**2

    def exact_quadratic(a, b):
        def exact(x):
            return x**3 / 3.0

        return exact(b) - exact(a)

    v, _ = cp.integrate(quadratic, -1.0, 1.0)
    assert np.allclose(v, exact_quadratic(-1, 1))

    v, _ = cp.integrate(quadratic, -np.pi, np.pi)
    assert np.allclose(v, exact_quadratic(-np.pi, np.pi))

    rng = np.random.default_rng()
    low = rng.uniform(-10, 10, 50)
    high = low + rng.uniform(2, 32, 50)
    v, _ = cp.integrate(quadratic, low, high)
    assert np.allclose(v, exact_quadratic(low, high))


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

    v, _ = cp.integrate(poly, -1.0, 1.0)
    assert np.allclose(v, exact_poly(-1.0, 1.0))

    v, _ = cp.integrate(poly, -np.pi, np.pi)
    assert np.allclose(v, exact_poly(-np.pi, np.pi))


def test_high_polynomial():
    def high_poly(x):
        return 2.0 * np.pi * x**20 - np.e * x**3 + 3.0 * x**2 - 4.0 * x + 8.0

    def exact_high_poly(a, b):
        def exact(x):
            return (
                (2.0 * np.pi / 21.0) * x**21
                - np.e / 4.0 * x**4
                + x**3
                - 2.0 * x**2
                + 8.0 * x
            )

        return exact(b) - exact(a)

    v, _ = cp.integrate(high_poly, -1.0, 1.0)
    np.allclose(v, exact_high_poly(-1, 1))

    v, _ = cp.integrate(high_poly, -np.pi, np.pi)
    np.allclose(v, exact_high_poly(-np.pi, np.pi))


def test_brick():
    def integrand_brick(x):
        return np.ones_like(x)

    def exact_brick(r):
        return np.prod(r, axis=0)

    value, error = cp.integrate(integrand_brick, [0.0], [1.0])
    assert np.allclose(value, exact_brick(1.0))
    assert np.all(error < 1e-6)

    lo = [[0, 0, 0]]
    hi = [[1, 2, 3]]
    value, error = cp.integrate(integrand_brick, lo, hi)
    assert np.allclose(value, exact_brick(hi))
    # assert np.all(error < 1e-6)


def exact_sphere(r):
    return (4.0 / 3.0) * np.pi * r**3


def test_sphere():
    def integrand_sphere(r, _, phi):
        return np.sin(phi) * r**2

    value, _ = cp.integrate(integrand_sphere, [0.0, 0.0, 0.0], [1.0, 2 * np.pi, np.pi])
    assert np.allclose(value, exact_sphere(1.0))

    value, _ = cp.integrate(integrand_sphere, [0.0, 0.0, 0.0], [3.0, 2 * np.pi, np.pi])
    assert np.allclose(value, exact_sphere(3.0))


def test_sphere_v():
    def integrand_sphere(r, _, phi):
        return np.sin(phi) * r**2

    radii = np.linspace(1, 100, 50)

    value, _ = cp.integrate(
        integrand_sphere,
        [0.0, 0.0, 0.0],
        [radii, 2 * np.pi, np.pi],
    )

    assert np.allclose(value, exact_sphere(radii))


def test_ellipsoid():
    def integrand_ellipsoid(rho, _, theta, a, b, c):
        return a * b * c * rho**2 * np.sin(theta)

    def exact_ellipsoid(axes):
        return (4 / 3) * np.pi * np.prod(axes, axis=0)

    value, error = cp.integrate(
        integrand_ellipsoid, [0.0, 0.0, 0.0], [1.0, 2.0 * np.pi, np.pi], args=(1, 2, 3)
    )

    assert np.allclose(value, exact_ellipsoid([1, 2, 3]))
    # assert np.all(error < 1e-5)


def test_multi():
    def integrand(x, y):
        return 1 + 8 * x * y

    low = np.array(
        [
            10000 * [0.0],
            10000 * [1.0],
        ]
    )

    high = np.array(
        [
            10000 * [3.0],
            10000 * [2.0],
        ]
    )

    value, error = cp.integrate(integrand, low, high)

    assert np.allclose(value, 57)
    assert np.all(error < 1e-6)


def test_van_dooren_de_riddler_simple_1():
    lo = np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
    hi = np.array([2.0, 1.0, (np.pi / 2.0), 1.0, 1.0, 1.0])

    value, error = cp.integrate(
        lambda x0, x1, x2, x3, x4, x5: (x0 * x1**2 * np.sin(x2)) / (4 + x3 + x4 + x5),
        lo,
        hi,
    )

    assert np.allclose(value, 1.434761888397263)
    assert np.all(error < 1e-2)


def test_van_dooren_de_riddler_simple_2():
    value, error = cp.integrate(
        lambda x0, x1, x2, x3: x2**2 * x3 * np.exp(x2 * x3) * (1 + x0 + x1) ** -2,
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 2.0]),
    )

    assert np.allclose(value, 0.5753641449035616)
    assert np.all(error < 1e-4)


#
#################################################################################
def test_van_dooren_de_riddler_simple_3():
    value, error = cp.integrate(
        lambda x, y, z: 8 / (1 + 2 * (x + y + z)),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
    )

    assert np.allclose(value, 2.152142832595894)
    assert np.all(error < 1e-4)


#
#
# def test_van_dooren_de_riddler_oscillating_4():
#     value, error = cp.integrate(
#         lambda x, y, z, a, b: np.cos(x + y + z + a + b),
#         np.array([0, 0, 0, 0, 0]),
#         np.array([np.pi, np.pi, np.pi, np.pi, 0.5 * np.pi]),
#     )
#
#     assert np.allclose(value, 16.0)
#     assert np.all(error < 1e-3)
#
#
# def test_van_dooren_de_riddler_oscillating_5():
#     value, error = cp.integrate(
#         lambda x, _, __, ___: np.sin(10 * x),
#         np.array([0, 0, 0, 0]),
#         np.array([1, 1, 1, 1]),
#     )
#
#     assert np.allclose(value, 0.1839071529076452)
#     assert np.all(error < 1e-4)
#
#
# def test_van_dooren_de_riddler_oscillating_6():
#     value, error = cp.integrate(
#         lambda x, y: np.cos(x + y),
#         np.array([0, 0]),
#         np.array([3 * np.pi, 3 * np.pi]),
#     )
#
#     assert np.allclose(value, -4)
#     assert np.all(error < 1e-4)
#
#
# def test_van_dooren_de_riddler_peaked_7():
#     value, error = cp.integrate(
#         lambda x, y, z: (x + y + z) ** -2,
#         np.array([0, 0, 0]),
#         np.array([1, 1, 1]),
#         rtol=1e-7,
#         atol=1e-7,
#     )
#
#     assert np.allclose(value, 0.8630462173553432)
#     assert np.all(error < 1e-4)
#
#
# def test_van_dooren_de_riddler_peaked_9():
#     def f(x, y):
#         return np.reciprocal((x**2 + 1e-4) * ((y + 0.25) ** 2 + 1e-4))
#
#     value, error = cp.integrate(f, [0, 0], [1, 1])
#     assert np.allclose(value, 499.1249442241215)
#     assert np.all(error < 1e-3)
#
#     value, error = cp.integrate(f, np.zeros((2, 100)), np.ones((2, 100)))
#     assert np.allclose(value, 499.1249442241215)
#     assert np.all(error < 1e-3)
#
#
# def test_van_dooren_de_riddler_peaked_10():
#     value, error = cp.integrate(
#         lambda v, u: np.exp(np.abs((v + u) - 1)),
#         np.array([0, 0]),
#         np.array([1, 1]),
#     )
#
#     assert np.allclose(value, 1.436563656918090)
#     assert np.all(error < 1e-4)
