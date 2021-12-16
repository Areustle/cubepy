from timeit import timeit

import numpy as np
import pytest

import cubepy


def integrand_brick(_):
    return 1.0


def integrand_sphere(x):
    r, _, phi = x
    return r ** 2 * np.sin(phi)


def integrand_ellipsoid(x, a, b, c):
    rho, _, theta = x
    return a * b * c * rho ** 2 * np.sin(theta)


def integrand_ellipsoid_v(x, r):
    rho = np.array(x[:, 0])
    # phi = np.array(x[:, 1])
    theta = np.array(x[:, 2])
    return np.prod(r, axis=0) * rho ** 2 * np.sin(theta)


def exact_brick(r):
    return np.prod(r, axis=0)


def exact_sphere(r):
    return (4 / 3) * np.pi * r ** 3


def exact_ellipsoid(axes):
    return (4 / 3) * np.pi * np.prod(axes, axis=0)


@pytest.mark.xfail(reason="Not implemented")
def test_brick():

    assert cubepy.integrate(integrand_brick, 0.0, 1.0) == exact_brick(1)


if __name__ == "__main__":

    N = int(1e8)
    a = np.ones((4, N, 4))

    print(timeit(lambda: np.prod(a, 0), number=1))
    print(timeit(lambda: np.prod(a, -1), number=1))
