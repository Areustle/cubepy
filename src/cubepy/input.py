# BSD 3-Clause License
#
# Copyright (c) 2023, Alexander Reustle
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from functools import reduce
from itertools import combinations

# import inspect
from typing import Callable, Sequence, Tuple, Union

import numpy as np

from . import points
from .type_aliases import NPT, InputBoundsT  # ValidBoundsT

__all__ = ["parse_input"]


def validate_inputs(b: InputBoundsT, n):
    if isinstance(b, (int, float)):  # bounds are scalar
        if n != 1:
            raise RuntimeError(
                f"Expected {n} dimensional integral, but only given "
                f"1 Limit of Integration parameter: '{b}'."
            )
    elif isinstance(b, (list, tuple)):
        if n != len(b):
            raise RuntimeError(
                f"Expected {n} dimensional integral, but given "
                f"{len(b)} limit(s) of integration: '{b}'."
            )
        for x in b:
            if not isinstance(x, (int, float, np.ndarray)):
                raise RuntimeError(
                    f"Limit of integration type ({type(x)}) not recognized for {x}."
                )
        for x, y in combinations(b, 2):
            if np.asarray(x).shape != np.asarray(y).shape:
                raise RuntimeError("All Limits of Integration must have same shape")
    elif isinstance(b, np.ndarray):  # bounds are scalar
        if b.ndim > 0 and n != b.shape[0] and n != 1:
            raise RuntimeError(
                f"Expected {n} dimensional integral, but only given "
                f" {b.shape} shaped limit of integration parameter: '{b}'."
            )
    else:
        raise RuntimeError(
            f"Limit of integration type ({type(b)}) not recognized for {b}."
        )


def pre_format_bounds(b: InputBoundsT) -> NPT:
    """
    Convert the limits of integration into tuples of numpy arrays.

    If the input is an array, and that array has leading dimension equal to the
    dimension of integration, assume the array is a packed set of arrays.
    """
    if isinstance(b, (list, tuple)):
        return np.asarray(b)
    elif isinstance(b, np.ndarray):
        return b
    else:  # b is scalar
        return np.array([[b]])


def compatible_shapes(
    a: Union[None, Sequence[int]], b: Union[None, Sequence[int]]
) -> bool:
    r"""Check if shape a is broadcastable to shape b, or return true if either is None."""
    if a is None or b is None:
        return True
    return all((m == n) | (m == 1) | (n == 1) for m, n in zip(a[::-1], b[::-1]))


def parse_input(
    f: Callable,
    low: InputBoundsT,
    high: InputBoundsT,
    args=tuple(),
    nint: Union[None, int] = None,
) -> Tuple[NPT, NPT, Tuple[int, ...]]:
    """
    Format the limits of integration as N-tuples of mixed scalar & array data where each
    element is a bound of the integral.

    Also determine the event_shape.
    """
    signature_nint = f.__code__.co_argcount + f.__code__.co_kwonlyargcount - len(args)
    nint = signature_nint if nint is None else nint

    if isinstance(low, np.ndarray) and low.ndim > 1:
        raise RuntimeError("input arrays must be of lower dimension")
    if isinstance(high, np.ndarray) and high.ndim > 1:
        raise RuntimeError("input arrays must be of lower dimension")

    low = np.asarray(low).ravel()
    high = np.asarray(high).ravel()

    if low.ndim != 1 or high.ndim != 1:
        raise RuntimeError("Integrations Bounds not flat.")
    if low.size != high.size:
        raise RuntimeError("Bounds Length Mismatch.")
    if low.size != nint:
        raise RuntimeError("Bounds length not compatible with function signature.")

    event_shape = tuple(
        reduce(max, [a.shape for a in args if isinstance(a, np.ndarray)], ())
    )
    return low, high, event_shape


def parse_input_v2(
    f: Callable,
    low: InputBoundsT,
    high: InputBoundsT,
    args=tuple(),
    nint: Union[None, int] = None,
) -> Tuple[NPT, NPT, Tuple[int, ...]]:
    """
    Format the limits of integration as N-tuples of mixed scalar & array data where each
    element is a bound of the integral.

    Also determine the event_shape.
    """
    signature_nint = f.__code__.co_argcount + f.__code__.co_kwonlyargcount - len(args)
    nint = signature_nint if nint is None else nint

    # Validate the input bounds on their own terms.
    validate_inputs(low, nint)
    validate_inputs(high, nint)

    # convert the input bounds into mutable lists of numpy arrays
    low = pre_format_bounds(low)
    high = pre_format_bounds(high)

    if low.shape != high.shape:
        raise RuntimeError("Bounds Length Mismatch.")

    if low.ndim != 2:
        raise RuntimeError(f"Bounds Shape formatted incorrectly: ndim={low.ndim}")

    if low.shape[0] != nint:
        raise RuntimeError("Bounds Shape formatted incorrectly: nint={low.shape[0]}")

    # if not all([x.shape == y.shape for x, y in combinations(low + high, 2)]):
    #     raise RuntimeError("Limits of Integration must all have equal shape.")

    bounds_event_shape = low.shape[1:]
    args_event_shapes = [a.shape for a in args if isinstance(a, np.ndarray)]

    if not all(
        compatible_shapes(xa, xb)
        for xa, xb in combinations([bounds_event_shape, *args_event_shapes], 2)
    ):
        raise RuntimeError("Limits of Integration have shapes incompatible with args.")

    event_shape = reduce(max, args_event_shapes)

    return low, high, event_shape


def get_arg_evt_idx(args, event_shape):
    """
    The indices of the args sequence which are event_shape compatible.
    """
    return tuple(
        i
        for i, a in enumerate(args)
        if isinstance(a, np.ndarray) and compatible_shapes(a.shape, event_shape)
    )


def get_arg_evt_mask(args, event_shape):
    """
    The mask of the args sequence which are event_shape compatible.
    """
    return tuple(
        isinstance(a, np.ndarray) and compatible_shapes(a.shape, event_shape)
        for a in args
    )


def get_max_tile_length(center, tile_byte_limit):
    r"""Compute the maximum length of an event-wise tile of computation from the user
    provided parameters.
    """

    domain_dim = center.shape[0]
    pts = points.num_points(domain_dim) if domain_dim > 1 else 15
    event_size = center.itemsize * domain_dim * pts
    max_tile_len = tile_byte_limit / event_size
    max_tile_len = np.maximum(max_tile_len, 1)
    return max_tile_len
