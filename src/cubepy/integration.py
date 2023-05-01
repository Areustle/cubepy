from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Any, Callable, Sequence

import numpy as np

from . import converged, input, region
from .gauss_kronrod import gauss_kronrod
from .genz_malik import genz_malik
from .points import gk_pts, gm_pts
from .type_aliases import NPF

__all__ = ["integrate"]


def integrate(
    f: Callable,
    low,
    high,
    args: Sequence[Any] = tuple([]),
    *,  # kwonly arguments
    domain_dim: None | int = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    itermax: int | float = 16,
) -> tuple[NPF, NPF]:
    """Numerical Cubature in multiple dimensions.

    Functions with 1D domain use an adaptive Gauss Kronrod Quadrature scheme.
    Functions with 2+D domain use an adaptive Genz Malik Cubature scheme.

    The adaptive regional subdivision is performed independently for un-converged
    regions.

    Local convergence is obtained when the tolerance values exceed a local
    region's error estimate by Bernsten's error estimation formula.

    Global convergence is obtained when all regions are locally converged.

    Parameters
    ==========
        f: Callable
            Integrand function: expected function signature is
            `f(x: NDArray, *args) -> NDArray`
            where `x` is the expected vector input, with domain in the leading dimension
            (`x.shape[0]`)
        low: Scalar, Iterable, or NDArray
            The finite integral lower bound. If a 1D vector the default behavior is to
            treat the length (`shape[0]`) as the domain of integration of 1 event.
            If a 2D ndarray, the default behavior is to treat the leading dimension
            (`shape[0]`) as the domain of integration, and the trailing dimension as the
            set of independent events over which to integrate. This default behavior
            is controlled by the `event_axis` parameter.
        high: Scalar, Iterable, or NDArray
            The finite integral high bound. Behavior is identical to `low` parameter.
        args: tuple, Optional
            The function arguments to pass into `f`.
        rtol: float, Optional
            Relative local error tolerance. Default 1e-5
        atol: float, Optional
            Absolute local error tolerance. Default 1e-8
        itermax: int or float, Optional
            Maximum number of subdivision iterations to perform on the integrand.
            Default is 100
    """

    # :::::::::::::::: Shapes ::::::::::::::::::
    # {low, high}       [ domain_dim, {events} ]
    # {center, hwidth}  [ domain_dim, regions, {events} ]
    # {volume}          [ regions ]

    # ---------------- Results -----------------
    # {value}       [ events ]
    # {error}       [ events ]
    # {split_dim}   [ events ]

    # ------------ Working vectors -------------
    # {cmask}   [ regions, events ]

    low, high, event_shape = input.parse_input(f, low, high, args, domain_dim)
    domain_dim = len(low)
    nevts = reduce(mul, event_shape, 1)
    active_evt_idx = np.arange(nevts)  # [ kept_evts ]
    aemsk = input.get_arg_evt_mask(args, event_shape)
    error = np.array([])

    # [ events ]
    result_value = np.zeros(nevts)
    result_error = np.zeros(nevts)
    # [ regions, events ]
    parent_value = np.zeros((1, nevts))

    # Prepare the integrand including applying the event mask to the maskable elements
    # in the args array.
    def _f(ev):
        return lambda x: f(*x, *(a[ev] if b else a for a, b in zip(args, aemsk)))

    # Create initial region
    center, halfwidth, vol = region.region(low, high)

    # prepare the integral rule
    rule_ = gauss_kronrod if len(center) == 1 else genz_malik
    pts_ = gk_pts if len(center) == 1 else gm_pts

    def rule(p, h, v, e):
        return rule_(_f(e), p, h, v)

    pts = pts_(center, halfwidth)

    iter: int = 1
    while iter < int(itermax):
        # Perform rule application on unconverged regions. [regions, events]
        value, error, split_dim = rule(pts, halfwidth, vol, active_evt_idx)

        iter += 1

        # Determine which regions are converged [ regions, active_events ]
        converge_mask = converged.converged(value, error, parent_value, rtol, atol)

        # Event indices of locally converged regions, events [ regions, events ]
        evtidx = np.broadcast_to(active_evt_idx, converge_mask.shape)[converge_mask]

        # Accumulate converged region results into correct events. bincount is most
        # efficient accumulator when multiple regions in the same event converge.
        result_value += np.bincount(evtidx, value[converge_mask], nevts)
        result_error += np.bincount(evtidx, error[converge_mask], nevts)

        if np.all(converge_mask):
            break

        # Locally unconverged regions. [ region, events ]
        unconverged_mask = ~converge_mask
        region_mask = np.any(unconverged_mask, axis=1)  # [ regions ]
        event_mask = np.any(unconverged_mask, axis=0)  # [ events ]
        active_mask = np.ix_(region_mask, event_mask)

        # Indices of unconverged events
        active_evt_idx = active_evt_idx[event_mask]  # [kept_evts]

        # update parent_values with active values
        parent_value = value[active_mask]  # [ kept_regions, kept_events ]

        # mask out the converged events from regions
        center = [c[region_mask] if c.shape[1] == 1 else c[active_mask] for c in center]
        halfwidth = [
            h[region_mask] if h.shape[1] == 1 else h[active_mask] for h in halfwidth
        ]
        vol = vol[event_mask]
        split_dim = split_dim[region_mask]

        # subdivide the un-converged regions [ kept_regions, kept_events ]
        center, halfwidth, vol = region.split(center, halfwidth, vol, split_dim)
        pts = pts_(center, halfwidth, pts=pts)

    if iter == int(itermax):
        raise RuntimeWarning(
            "Failed to converge within the iteration limit: ",
            itermax,
            "Maximum un-converged error estimate: ",
            np.amax(error),
        )

    result_value = np.reshape(result_value, event_shape)
    result_error = np.reshape(result_error, event_shape)

    return np.squeeze(result_value), np.squeeze(result_error)
