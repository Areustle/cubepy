import numpy as np

import cubepy as cp


def integrand_sphere_v(r, _, phi):
    return np.sin(phi) * r**2


radii = np.linspace(1, 5, int(5))

value, error = cp.integrate(
    integrand_sphere_v,
    [0.0, 0.0, 0.0],
    [radii, 2 * np.pi, np.pi],
    # rtol=1e-3,
    # atol=1e-3
    itermax=4,
)
