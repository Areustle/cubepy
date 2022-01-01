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

from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np


class BufferedVectors:
    def __init__(self, range_dim, dtype=float, parallel=False):

        self.range_dim = range_dim
        self.dtype = dtype
        self.parallel = parallel

        self.smm = SharedMemoryManager()
        if self.parallel:
            self.smm.start()

        @dataclass
        class Vecmeta:
            buffer: SharedMemory
            shape: tuple
            dtype: type

        self.vecs = {
            "center": Vecmeta(self.smm.SharedMemory(1), (0, 0), dtype),
            "halfwidth": Vecmeta(self.smm.SharedMemory(1), (0, 0), dtype),
            "vol": Vecmeta(self.smm.SharedMemory(1), (0,), dtype),
            "evtidx": Vecmeta(self.smm.SharedMemory(1), (0,), np.intp),
            "value": Vecmeta(self.smm.SharedMemory(1), (0, 0), dtype),
            "error": Vecmeta(self.smm.SharedMemory(1), (0, 0), dtype),
            "split": Vecmeta(self.smm.SharedMemory(1), (0,), np.intp),
        }

    def __del__(self):
        if self.smm is not None:
            self.smm.shutdown()

    def _allocate(self, name, vector):
        """Grow the named vector buffer if necessary."""

        size = vector.nbytes if isinstance(vector, np.ndarray) else np.prod(vector)
        shape = vector.shape if isinstance(vector, np.ndarray) else vector

        if self.vecs[name].buffer.size < size:
            self.vecs[name].buffer.close()
            self.vecs[name].buffer.unlink()
            self.vecs[name].buffer = self.smm.SharedMemory(size)

        self.vecs[name].shape = shape

    def _update(self, vecname, vector):
        """Fill the named buffer with data."""

        self._allocate(vecname, vector)
        np.copyto(self.get_np(vecname), vector)

    def get_np(self, vecname):
        """Get a numpy array of the underlying named buffer's data."""

        return np.ndarray(
            self.vecs[vecname].shape,
            dtype=self.vecs[vecname].dtype,
            buffer=self.vecs[vecname].buffer.buf,
        )

    def update_inputs(self, center, halfwidth, vol, evtidx):
        """Update the input arrays."""

        self._update("center", center)
        self._update("halfwidth", halfwidth)
        self._update("vol", vol)
        self._update("evtidx", evtidx)

    def allocate_results(self, evt_len):
        """Allocate the result arrays."""
        self._allocate("value", self.dtype.itemsize * self.range_dim * evt_len)
        self._allocate("error", self.dtype.itemsize * self.range_dim * evt_len)
        self._allocate("split", self.dtype.itemsize * evt_len)

    def input_arrays(self):
        return (
            self.get_np("center"),
            self.get_np("halfwidth"),
            self.get_np("vol"),
            self.get_np("evtidx"),
        )

    def result_arrays(self):
        return self.get_np("value"), self.get_np("error"), self.get_np("split")
