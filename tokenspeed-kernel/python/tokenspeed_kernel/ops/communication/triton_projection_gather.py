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

"""Safe borrowed-input adapter for the existing multimem all-gather."""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from tokenspeed_kernel.ops.communication.triton import (
    _alloc_symm,
    all_gather,
    create_state,
)


class ProjectionGatherState:
    """Prepare BF16 multicast scratch before capture.

    Args:
        group: CUDA multicast-capable process group.
        max_rows: Maximum physical input rows per rank.
        hidden: Input channel count.
        device: This rank's selected CUDA device.
    """

    def __init__(self, group, max_rows: int, hidden: int, device: torch.device):
        self.group = group
        self.state = create_state(
            group=group,
            rank_in_group=group.rank(),
            max_tokens=max_rows * group.size(),
            hidden_size=hidden,
            device=device,
            max_numel=0,
            max_bytes=0,
            attnres_max_numel=0,
            attnres_max_rows=0,
        )
        self.handle = symm_mem.rendezvous(self.state.comm_buff, group=group)
        if not self.handle.multicast_ptr:
            raise RuntimeError("Projection all-gather requires CUDA multicast")
        # Keep the PyTorch barrier signals separate from the multimem kernel's
        # per-CTA signal pad; their protocols must not share live signal slots.
        self.reuse_buffer, self.reuse_handle = _alloc_symm(
            (1,), torch.bfloat16, device, group
        )
        self.max_rows = max_rows

    def gather(self, inputs: torch.Tensor) -> torch.Tensor:
        """Borrow rank-major [P*M,K] rows until the next gather on this state.

        All subgroup ranks supply the same physical M. Serialize this call and
        all consumers on one stream. The explicit pre-copy barrier prevents a
        fast rank from overwriting input another rank's preceding GEMM reads.
        """
        if not 0 < inputs.shape[0] <= self.max_rows:
            raise ValueError("Projection all-gather exceeds prepared capacity")
        self.reuse_handle.barrier(channel=0)
        return all_gather(
            self.state,
            inputs,
            tp_num_tokens=inputs.shape[0] * self.group.size(),
            token_list_in_group=None,
            safe=False,
        )

    def close(self) -> None:
        """Collectively release storage after graphs and borrowed consumers die."""
        if self.state is not None:
            torch.cuda.synchronize(self.state.device)
            dist.barrier(group=self.group)
            self.handle = None
            self.reuse_handle = None
            self.reuse_buffer = None
            self.state = None
