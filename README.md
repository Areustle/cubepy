# 🧊 CubePy 🐍

Batched Numerical Cubature: Adaptive Multi-dimensional Integration in Python.

Finally, a _fully_ vectorized multi-dimensional numerical integrator! CubePy is the
High-Performance, deterministic numerical integrator you've been waiting for.

Integrate functions with:

- [x] High Dimensional Domains
- [x] High Dimensional Co-Domains
- [x] Multiple limits of integration
- [x] Independent batches computed concurrently.
- [x] No Monte-Carlo!

CubePy is a fully-vectorized python package for performing numerical integration on
multi-dimensional vector functions. CubePy performs these operations efficiently using
numpy and the Genz-Malik Adaptive cubature algorithm.

Functions with 2+D domain use an adaptive Genz Malik Cubature scheme.
Functions with 1D domain use an adaptive Gauss Kronrod Quadrature scheme.

The adaptive regional subdivision is performed independently for un-converged regions.
Local convergence is obtained when the global tolerance values exceed a local
region's absolute and relative error estimate Using Bernsten's Error Estimation formula.
Global convergence is obtained when all regions are locally converged.

## Example Usage

### The volume of a sphere, a 3 dimensional integral.

```python
## The analytical volume of a sphere for validation.
def volume_of_sphere(radius):
    return (4.0 / 3.0) * np.pi * radius**3

## The integrand function.
def sphere_integrand(r, theta, phi):
    return np.sin(phi) * r**2

## The upper and lower bounds of integration.
low = [0.0, 0.0, 0.0]
high = [1.0, 2 * np.pi, np.pi]

# The Integration!
value, error = cp.integrate(sphere_integrand, low, high)

# Confirm the results
assert np.allclose(value, volume_of_sphere(1.0))

# Try a larger sphere
high[0] = 42
value, error = cp.integrate(sphere_integrand, low, high)
assert np.allclose(value, volume_of_sphere(42))
```

### The Volume of _multiple_ spheres in a single call

Frequently we encounter the need to compute an integral on an array of bounds,
representing different events in our dataset. CubePy was designed to handle these exact
cases. Simply define one or more of your integration bounds as an array of entries and
call cp.integrate

```python

## 1,000,000 radii between 1 and 42
radii = np.linspace(1, 42, int(1e6))

## The upper and lower bounds of integration.
low = [0.0, 0.0, 0.0]
high = [radii, 2 * np.pi, np.pi]

# The Integration!
value, error = cp.integrate(sphere_integrand, low, high)
# Confirm the results
assert np.allclose(value, volume_of_sphere(radii))
```

### Single Dimensional integral support

The routine supports single dimensional integrals as well, with an adaptive Gauss
Kronrod scheme.

```python
def degree_20_polynomial(x):
    return 2.0 * np.pi * x**20 - np.e * x**3 + 3.0 * x**2 - 4.0 * x + 8.0

def exact_degree_20_polynomial(a, b):
    def exact(x):
        return (
            (2.0 * np.pi / 21.0) * x**21
            - np.e / 4.0 * x**4
            + x**3
            - 2.0 * x**2
            + 8.0 * x
        )

    return exact(b) - exact(a)

v, _ = cp.integrate(degree_20_polynomial, -1.0, 1.0)
np.allclose(v, exact_degree_20_polynomial(-1, 1))

v, _ = cp.integrate(degree_20_polynomial, -np.pi, np.pi)
np.allclose(v, exact_degree_20_polynomial(-np.pi, np.pi))
```

## References

- Berntsen, Espelid, Genz. An Adaptive Algorithm for the Approximate Calculation of Multiple Integrals
- Berntsen. Practical error estimation in adaptive multidimensional quadrature routines.
- Genz, Malik. An adaptive algorithm for numerical integration over an N-dimensional rectangular region.
- Van Dooren, De Ridder. An adaptive algorithm for numerical in- tegration over an N-dimensional cube.
