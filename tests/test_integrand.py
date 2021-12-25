import numpy as np

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

    print(value, error)


def test_multi():
    def integrand(x):
        return 1 + 8 * x[0] * x[1]

    def exact(r):
        return np.prod(r, axis=0)

    low = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    high = np.array(
        [
            [3.0],
            [2.0],
        ]
    )

    value, error = cubepy.integrate(integrand, low, high)

    assert np.allclose(value, exact(1.0))

    print(value, error)


if __name__ == "__main__":

    # N = int(1e8)
    # a = np.ones((4, N, 4))

    # from timeit import timeit
    # print(timeit(lambda: np.prod(a, 0), number=1))
    # print(timeit(lambda: np.prod(a, -1), number=1))

    test_multi()
