import numpy as np
import pytest

import cubepy


def integrand_brick(x):
    return np.ones_like(x)


def integrand_sphere(x):
    r, _, phi = x
    return r ** 2 * np.sin(phi)


def integrand_ellipsoid(x, a, b, c):
    rho, _, theta = x
    return a * b * c * rho ** 2 * np.sin(theta)


def integrand_ellipsoid_v(x, r):
    rho = np.asarray(x[0])
    # phi = np.array(x[1])
    theta = np.asarray(x[2])
    return np.prod(r, axis=0) * rho ** 2 * np.sin(theta)


def exact_brick(r):
    return np.prod(r, axis=0)


def exact_sphere(r):
    return (4 / 3) * np.pi * r ** 3


def exact_ellipsoid(axes):
    return (4 / 3) * np.pi * np.prod(axes, axis=0)


def test_brick():

    value, error = cubepy.integrate(integrand_brick, np.array([0.0]), np.array([1.0]))

    assert np.allclose(value, exact_brick(1.0))
    assert np.all(error < 1e-6)


def test_multi():
    def integrand(x):
        return 1 + 8 * x[0] * x[1]

    low = np.array(
        [
            100000 * [0.0],
            100000 * [1.0],
        ]
    )

    high = np.array(
        [
            100000 * [3.0],
            100000 * [2.0],
        ]
    )

    value, error = cubepy.integrate(integrand, low, high)
    print(value, error)

    # assert np.allclose(value, 57)
    # assert np.all(error < 1e-6)


def test_van_dooren_de_riddler_simple_1():

    low = np.array(
        [
            [0.0],
            [0.0],
            [0.0],
            [-1.0],
            [-1.0],
            [-1.0],
        ]
    )

    high = np.array(
        [
            [2.0],
            [1.0],
            [0.5 * np.pi],
            [1.0],
            [1.0],
            [1.0],
        ]
    )

    value, error = cubepy.integrate(
        lambda x: (x[0] * x[1] ** 2 * np.sin(x[2])) / (4 + x[3] + x[4] + x[5]),
        low,
        high,
    )

    assert np.allclose(value, 1.434761888397263)
    assert np.all(error < 1e-2)


def test_van_dooren_de_riddler_simple_2():

    value, error = cubepy.integrate(
        lambda x: x[2] ** 2 * x[3] * np.exp(x[2] * x[3]) * (1 + x[0] + x[1]) ** -2,
        np.expand_dims(np.array([0.0, 0.0, 0.0, 0.0]), 1),
        np.expand_dims(np.array([1.0, 1.0, 1.0, 2.0]), 1),
    )

    assert np.allclose(value, 0.5753641449035616)
    assert np.all(error < 1e-4)


def test_van_dooren_de_riddler_simple_3():

    value, error = cubepy.integrate(
        lambda x: 8 / (1 + 2 * (np.sum(x, axis=0))),
        np.expand_dims(np.array([0.0, 0.0, 0.0]), 1),
        np.expand_dims(np.array([1.0, 1.0, 1.0]), 1),
    )

    assert np.allclose(value, 2.152142832595894)
    assert np.all(error < 1e-4)


def test_van_dooren_de_riddler_oscillating_4():

    value, error = cubepy.integrate(
        lambda x: np.cos(np.sum(x, axis=0)),
        np.expand_dims(np.array([0, 0, 0, 0, 0]), 1),
        np.expand_dims(np.array([np.pi, np.pi, np.pi, np.pi, 0.5 * np.pi]), 1),
    )

    assert np.allclose(value, 16.0)
    assert np.all(error < 1e-3)


def test_van_dooren_de_riddler_oscillating_5():

    value, error = cubepy.integrate(
        lambda x: np.sin(10.0 * x[0]),
        np.expand_dims(np.array([0, 0, 0, 0]), 1),
        np.expand_dims(np.array([1, 1, 1, 1]), 1),
    )

    assert np.allclose(value, 0.1839071529076452)
    assert np.all(error < 1e-4)


def test_van_dooren_de_riddler_oscillating_6():

    value, error = cubepy.integrate(
        lambda x: np.cos(np.sum(x, axis=0)),
        np.expand_dims(np.array([0, 0]), 1),
        np.expand_dims(np.array([3 * np.pi, 3 * np.pi]), 1),
    )

    assert np.allclose(value, -4)
    assert np.all(error < 1e-4)


def test_van_dooren_de_riddler_peaked_7():

    value, error = cubepy.integrate(
        lambda x: np.sum(x, axis=0) ** -2,
        np.expand_dims(np.array([0, 0, 0]), 1),
        np.expand_dims(np.array([1, 1, 1]), 1),
        abstol=1e-7,
        reltol=1e-7,
    )

    assert np.allclose(value, 0.8630462173553432)
    assert np.all(error < 1e-4)


@pytest.mark.skip("Known bad result")
def test_van_dooren_de_riddler_peaked_8():
    def f(v):
        x = v[0]
        y = v[1]

        return (605.0 * y) / (
            (1.0 + 120.0 * (1.0 - y)) * ((1.0 + 120.0 * y)) ** 2
            + 25.0 * x ** 2 * y ** 2
        )

    value, error = cubepy.integrate(
        f,
        np.expand_dims(np.array([0, 0]), 1),
        np.expand_dims(np.array([1, 1]), 1),
    )

    assert np.allclose(value, 1.047591113142868)
    assert np.all(error < 1e-4)


def test_van_dooren_de_riddler_peaked_9():
    def f(v):
        x = v[0]
        y = v[1]

        return np.reciprocal((x ** 2 + 0.0001) * ((y + 0.25) ** 2 + 0.0001))

    value, error = cubepy.integrate(
        f,
        np.expand_dims(np.array([0, 0]), 1),
        np.expand_dims(np.array([1, 1]), 1),
    )

    assert np.allclose(value, 499.1249442241215)
    assert np.all(error < 1e-3)


def test_van_dooren_de_riddler_peaked_10():
    value, error = cubepy.integrate(
        lambda v: np.exp(np.abs(np.sum(v, axis=0) - 1)),
        np.expand_dims(np.array([0, 0]), 1),
        np.expand_dims(np.array([1, 1]), 1),
    )

    assert np.allclose(value, 1.436563656918090)
    assert np.all(error < 1e-4)


if __name__ == "__main__":
    test_multi()
    test_van_dooren_de_riddler_peaked_10()
