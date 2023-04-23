import numpy as np

import cubepy as cp


def integrand_sphere_v(r, _, phi, radius):
    return (np.sin(phi) * r**2)[..., None] * (radius) ** 3


radii = np.linspace(1, 100, int(1e6))
# radii = np.array([1, 100])

value, error = cp.integrate(
    integrand_sphere_v,
    [0.0, 0.0, 0.0],
    [1.0, 2 * np.pi, np.pi],
    args=(radii,),
)
