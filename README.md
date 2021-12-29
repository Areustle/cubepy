# CubePy

Numerical Cubature: Adaptive Multi-dimensional Integration in Python.

Finally, a _fully_ vectorized multi-dimensional numerical integrator! CubePy is the
High-Performance, deterministic numerical integrator you've been waiting for.

Integrate functions with:
 - [x] High Dimensional Domains
 - [x] High Dimensional Co-Domains
 - [x] Multiple limits of integration
 - [x] Independent events computed concurrently.
 - [x] No Monte-Carlo!

CubePy is a fully-vectorized python package for performing numerical integration on
multi-dimensional vector functions. CubePy performs these operations efficiently using
numpy and the Genz-Malik Adaptive cubature algorithm.

Functions with 2+D domain use an adaptive Genz Malik Cubature scheme.
Functions with 1D domain use an adaptive Gauss Kronrod Quadrature scheme.

The adaptive regional subdivision is performed independently for un-converged regions.
Local convergence is obtained when the global tolerance values exceed a local
region's error estimate.
Global convergence is obtained when all regions are locally converged.
