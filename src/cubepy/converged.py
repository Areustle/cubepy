# BSD 3-Clause License
#
# Copyright (c) 2021, Alex Reustle
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

import numpy as np

from .type_aliases import NPB, NPF, NPT


def converged(
    result: NPF, local_error: NPF, parent_result: NPT, rtol: float, atol: float
) -> NPB:
    """Determine wether error values are below threshod for convergence."""

    r = result.shape[0] // 2
    E = local_error
    if r >= 1:
        E2 = np.abs(parent_result - (result[:r] + result[r:]))  # [ r/2, events ]
        ES = E[:r] + E[r:]
        inv_error = np.reciprocal(
            np.where(ES <= 2.0 * np.finfo(ES.dtype).eps, 1.0, ES)
        )  # [ r/2, events ]
        E[:r] += E2 * (0.25 + 0.5 * inv_error * E[:r])
        E[r:] += E2 * (0.25 + 0.5 * inv_error * E[r:])

    # {cmask}       [ regions, events ]
    return E <= (atol + rtol * np.abs(result))
