import numpy as np

import cubepy as cp


def integrand_sphere_v(r, _, phi):
    return np.sin(phi) * r**2


radii = np.linspace(1, 100, int(1e6))

value, error = cp.integrate(
    integrand_sphere_v,
    [0.0, 0.0, 0.0],
    [radii, 2 * np.pi, np.pi],
    # rtol=1e-3,
    # atol=1e-3,
)


def exact_sphere(r):
    return (4.0 / 3.0) * np.pi * r**3


print("Computed Value", value)
print("Exact Value", exact_sphere(radii))
print("Absolute Error", value - exact_sphere(radii))
print("Relative Error", (value - exact_sphere(radii)) / exact_sphere(radii))
print("Integral Error", error)
