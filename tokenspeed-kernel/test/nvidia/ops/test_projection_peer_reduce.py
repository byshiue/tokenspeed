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


"""Indirect peer loads preserve owner offsets, tails, and FP32 accumulation."""

import pytest
import torch
from tokenspeed_kernel.ops.communication.triton_projection import owner_reduce

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)


@pytest.mark.parametrize("rows,hidden", [(1, 7), (3, 513), (16, 7168), (65, 7168)])
def test_owner_reduce_aligned_bases_and_unaligned_owner_slices(rows, hidden):
    # Local buffers isolate addressing/math; the distributed harness covers
    # remote peer access and symmetric-memory publication/reuse barriers.
    torch.manual_seed(42)
    peers = [
        torch.randn(4 * rows, hidden, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    assert all(x.data_ptr() % 16 == 0 for x in peers)
    pointers = torch.tensor(
        [x.data_ptr() for x in peers], device="cuda", dtype=torch.uint64
    )
    for rank in range(4):
        guarded = torch.full(
            (rows * hidden + 16,), 123, device="cuda", dtype=torch.bfloat16
        )
        output = guarded[: rows * hidden].view(rows, hidden)
        expected = torch.zeros((rows, hidden), device="cuda", dtype=torch.float32)
        for peer in peers:
            expected += peer[rank * rows : (rank + 1) * rows].float()
        owner_reduce[((rows * hidden + 1023) // 1024,)](
            pointers, output, rows, hidden, 4, rank, 1024
        )
        torch.testing.assert_close(output, expected.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(
            guarded[rows * hidden :],
            torch.full_like(guarded[rows * hidden :], 123),
            rtol=0,
            atol=0,
        )
