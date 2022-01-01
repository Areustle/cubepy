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

import sys
from copy import deepcopy
from multiprocessing.managers import SharedMemoryManager

from .gauss_kronrod import gauss_kronrod
from .genz_malik import genz_malik


class Rule:
    def __init__(self, f, domain_dim, evt_idx_arg, *args, parallel=False):
        # prepare the integral rule
        self.f = f
        self.rule = gauss_kronrod if domain_dim == 1 else genz_malik
        self.evt_idx_arg = evt_idx_arg
        self.parallel = parallel
        # self.smm = None
        # argsizes = map(lambda x: sys.getsizeof(x), args)

        # if self.parallel:
        #     self.smm = SharedMemoryManager()
        #     self.smm.start()

        #     buffers = [self.smm.SharedMemory(s) for s in argsizes]

        #     for a, b in zip(args, buffers):
        #         b.buf[:] = deepcopy(a)

        #     self.args = buffers

        # else:
        #     self.args = args

        self.args = args

        self._f = (
            lambda x, e, a: self.f(x, e, *a)
            if self.evt_idx_arg
            else lambda x, _, a: self.f(x, *a)
        )

    # def __del__(self):
    #     if self.smm is not None:
    #         self.smm.shutdown()

    def __call__(self, c, h, v, e):
        return self.rule(lambda x: self._f(x, e, self.args), c, h, v)
