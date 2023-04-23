import numpy as np

import cubepy as cp

# import pytest


# def test_brick():
#     def integrand_brick(x):
#         return np.ones_like(x)
#
#     def exact_brick(r):
#         return np.prod(r, axis=0)
#
#     value, error = cp.integrate(integrand_brick, 0.0, 1.0)
#     assert np.allclose(value, exact_brick(1.0))
#     assert np.all(error < 1e-6)
#
#     lo = [0, 0, 0]
#     hi = [1, 2, 3]
#     value, error = cp.integrate(integrand_brick, lo, hi)
#     assert np.allclose(value, exact_brick(hi))
#     assert np.all(error < 1e-6)
#
#
def test_sphere():
    def integrand_sphere(r, _, phi):
        return r**2 * np.sin(phi)

    def exact_sphere(r):
        return (4.0 / 3.0) * np.pi * r**3

    value, error = cp.integrate(
        integrand_sphere, [0.0, 0.0, 0.0], [1.0, 2 * np.pi, np.pi]
    )

    assert np.allclose(value, exact_sphere(1.0))
    assert np.all(error < 1e-5)

    value, error = cp.integrate(
        integrand_sphere, [0.0, 0.0, 0.0], [3.0, 2 * np.pi, np.pi]
    )
    assert np.allclose(value, exact_sphere(3.0))

    def integrand_sphere_(r_i, _, phi):
        return (3 * np.sqrt(3.0) * r_i) ** 2 * np.sin(phi)

    value, error = cp.integrate(
        integrand_sphere_, [0.0, 0.0, 0.0], [1.0, 2 * np.pi, np.pi]
    )
    assert np.allclose(value, exact_sphere(3.0))

    # assert np.all(error < 1e-5)

    def integrand_sphere_v(r, _, phi, radius):
        return (np.sin(phi) * r**2)[..., None] * (radius) ** 3

    # radii = np.linspace(1, 100, int(1e6))
    radii = np.array([1, 100])

    value, error = cp.integrate(
        integrand_sphere_v,
        [0.0, 0.0, 0.0],
        [1.0, 2 * np.pi, np.pi],
        args=(radii,),
    )

    assert np.allclose(
        value,
        exact_sphere(
            radii,
        ),
    )
    assert np.all(error < 1e-5)


#
#
# def test_ellipsoid():
#     def integrand_ellipsoid(x, a, b, c):
#         rho, _, theta = x
#         return a * b * c * rho**2 * np.sin(theta)
#
#     def exact_ellipsoid(axes):
#         return (4 / 3) * np.pi * np.prod(axes, axis=0)
#
#     value, error = cp.integrate(
#         integrand_ellipsoid, [0.0, 0.0, 0.0], [1.0, 2.0 * np.pi, np.pi], args=(1, 2, 3)
#     )
#
#     assert np.allclose(value, exact_ellipsoid([1, 2, 3]))
#     assert np.all(error < 1e-5)
#
#
# def test_multi():
#     def integrand(x):
#         return 1 + 8 * x[0] * x[1]
#
#     low = np.array(
#         [
#             1000000 * [0.0],
#             1000000 * [1.0],
#         ]
#     )
#
#     high = np.array(
#         [
#             1000000 * [3.0],
#             1000000 * [2.0],
#         ]
#     )
#
#     value, error = cp.integrate(integrand, low, high)
#
#     assert np.allclose(value, 57)
#     assert np.all(error < 1e-6)
#
#
# def test_van_dooren_de_riddler_simple_1():
#     lo = np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0])
#     hi = np.array([2.0, 1.0, (np.pi / 2.0), 1.0, 1.0, 1.0])
#
#     value, error = cp.integrate(
#         lambda x: (x[0] * x[1] ** 2 * np.sin(x[2])) / (4 + x[3] + x[4] + x[5]), lo, hi
#     )
#
#     assert np.allclose(value, 1.434761888397263)
#     assert np.all(error < 1e-2)
#
#
# def test_van_dooren_de_riddler_simple_2():
#     value, error = cp.integrate(
#         lambda x: x[2] ** 2 * x[3] * np.exp(x[2] * x[3]) * (1 + x[0] + x[1]) ** -2,
#         np.array([0.0, 0.0, 0.0, 0.0]),
#         np.array([1.0, 1.0, 1.0, 2.0]),
#     )
#
#     assert np.allclose(value, 0.5753641449035616)
#     assert np.all(error < 1e-4)
#
#
# def test_van_dooren_de_riddler_simple_3():
#     value, error = cp.integrate(
#         lambda x: 8 / (1 + 2 * (np.sum(x, axis=0))),
#         np.array([0.0, 0.0, 0.0]),
#         np.array([1.0, 1.0, 1.0]),
#     )
#
#     assert np.allclose(value, 2.152142832595894)
#     assert np.all(error < 1e-4)
#
#
# def test_van_dooren_de_riddler_oscillating_4():
#     value, error = cp.integrate(
#         lambda x: np.cos(np.sum(x, axis=0)),
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
#         lambda x: np.sin(10 * x[0]), np.array([0, 0, 0, 0]), np.array([1, 1, 1, 1])
#     )
#
#     assert np.allclose(value, 0.1839071529076452)
#     assert np.all(error < 1e-4)
#
#
# def test_van_dooren_de_riddler_oscillating_6():
#     value, error = cp.integrate(
#         lambda x: np.cos(np.sum(x, axis=0)),
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
#         lambda x: np.sum(x, axis=0) ** -2,
#         np.array([0, 0, 0]),
#         np.array([1, 1, 1]),
#         abstol=1e-7,
#         reltol=1e-7,
#     )
#
#     assert np.allclose(value, 0.8630462173553432)
#     assert np.all(error < 1e-4)
#
#
# def test_van_dooren_de_riddler_peaked_9():
#     def f(v):
#         x = v[0]
#         y = v[1]
#
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
#         lambda v: np.exp(np.abs(np.sum(v, axis=0) - 1)),
#         np.array([0, 0]),
#         np.array([1, 1]),
#     )
#
#     assert np.allclose(value, 1.436563656918090)
#     assert np.all(error < 1e-4)
#
#
# if __name__ == "__main__":
#     test_multi()
#     test_van_dooren_de_riddler_peaked_10()
