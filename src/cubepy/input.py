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

# import inspect
from typing import Any, Callable, Sequence  # , Tuple, Union

import numpy as np


def compatible_shapes(a: None | Sequence[int], b: None | Sequence[int]) -> bool:
    r"""Check if shape a is broadcastable to shape b, or return true if either is None."""
    if a is None or b is None:
        return True
    return all((m == n) | (m == 1) | (n == 1) for m, n in zip(a[::-1], b[::-1]))


def parse_input(
    f: Callable, low, high, args: Sequence[Any] = tuple(), nint: int | None = None
):
    r"""
    Structure the user provided bounds of integration as lists of numpy arrays.
    Ensure the shapes of the bounds are compatible with the
    """
    signature_nint = f.__code__.co_argcount + f.__code__.co_kwonlyargcount - len(args)
    nint = signature_nint if nint is None else nint

    if len(low) != nint and not (nint == 1 and isinstance(low, np.ndarray)):
        raise RuntimeError("Bounds length not compatible with function signature.")
    if len(high) != nint and not (nint == 1 and isinstance(high, np.ndarray)):
        raise RuntimeError("Bounds length not compatible with function signature.")

    low = [np.asarray(x) for x in low]
    high = [np.asarray(x) for x in high]

    low_uniq = {x.shape if x.shape else (1,) for x in low}
    high_uniq = {x.shape if x.shape else (1,) for x in high}

    low = [x.ravel() for x in low]
    high = [x.ravel() for x in high]

    big_event_shapes = (low_uniq | high_uniq) - {(1,)}

    if len(big_event_shapes) not in (0, 1):
        raise RuntimeError(
            "Independent Bounds events must be either (), (1,) or the same shape. "
            f"Measured bounds shapes are {low_uniq} and {high_uniq}"
        )

    event_shape = big_event_shapes.pop() if big_event_shapes else (1,)
    acceptable_arg_shapes = {event_shape, (1,), ()}
    arg_shapes = {np.asarray(x).shape for x in args}

    if arg_shapes - acceptable_arg_shapes:
        raise RuntimeError(
            "Integrand funciton has non-integrable user-provided Arguments with "
            f"incompatible shapes {acceptable_arg_shapes}"
        )

    return low, high, event_shape


def get_arg_evt_mask(args, event_shape):
    """
    The mask of the args sequence which are event_shape compatible.
    """
    return tuple(
        isinstance(a, np.ndarray) and compatible_shapes(a.shape, event_shape)
        for a in args
    )
