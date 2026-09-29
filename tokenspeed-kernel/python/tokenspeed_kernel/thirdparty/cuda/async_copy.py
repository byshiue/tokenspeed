# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Copy-engine access through the CUDA runtime already loaded by PyTorch."""

import ctypes
import os
from pathlib import Path

import torch


class CudaAsyncCopy:
    """Keep PyTorch's loaded CUDA runtime alive while graphs use its copies."""

    def __init__(self):
        self.library = ctypes.CDLL(
            str(Path(torch.__file__).parent / "lib" / "libtorch_cuda.so"),
            mode=os.RTLD_NOLOAD | os.RTLD_LOCAL,
        )
        self.copy = self.library.cudaMemcpyAsync
        self.copy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.copy.restype = ctypes.c_int

    def device_to_device(self, destination, source, size_bytes, stream):
        """Enqueue a contiguous local/peer device copy on an explicit stream."""
        status = self.copy(destination, source, size_bytes, 3, stream)
        if status:
            raise RuntimeError(f"cudaMemcpyAsync failed with CUDA error {status}")
