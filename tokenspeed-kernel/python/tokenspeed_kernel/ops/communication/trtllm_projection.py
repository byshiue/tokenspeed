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

"""One-shot Lamport reduction for serialized TP4 output projections."""

from ctypes import c_void_p

import torch
import torch.distributed as dist
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.signature import format_signatures


class ProjectionLamportState:
    """Own IPC scratch shared by sequential layers, not by concurrent streams.

    Args:
        group: Four-rank process group on a CUDA-IPC-accessible topology.
        max_rows: Prepared physical rows per rank, at most 128.
        hidden: Output width, a positive multiple of eight BF16 elements.
        device: Current rank's CUDA device, already selected by the caller.
    """

    def __init__(self, group, max_rows: int, hidden: int, device: torch.device):
        from tokenspeed_kernel.thirdparty.cuda.trtllm import (
            trtllm_create_ipc_workspace_for_reduce_scatter_fusion,
        )

        if group.size() != 4 or not 0 < max_rows <= 128 or hidden <= 0:
            raise ValueError("Lamport projection requires TP4 and 1..128 rows")
        if hidden % 8:
            raise ValueError("Lamport BF16 projection width must be divisible by 8")
        # Bound allocation and prevent the native wrapper silently selecting
        # two-shot on inputs larger than its signed-int32 Lamport address space.
        if 4 * 4 * max_rows * hidden * 2 >= 2**31 - 2**21:
            raise ValueError("Projection exceeds one-shot Lamport capacity")
        self.group = group
        self.max_rows = max_rows
        self.hidden = hidden
        self.buffer = torch.empty(
            (4 * max_rows, hidden), dtype=torch.bfloat16, device=device
        )
        self.handles, self.workspace = (
            trtllm_create_ipc_workspace_for_reduce_scatter_fusion(
                tp_rank=group.rank(),
                tp_size=4,
                max_token_num=4 * max_rows,
                hidden_dim=hidden,
                use_fp32_lamport=False,
                group=group,
                create_metadata=False,
            )
        )
        # The native helper's destroy routine frees shared IPC allocations,
        # but not the separately allocated local ring-control flags.
        self.control_ptr = self.workspace[-1].item()

    def input_buffer(self, rows: int) -> torch.Tensor:
        """Borrow a local GEMM destination; peers never read this buffer."""
        if self.handles is None or not 0 < rows <= self.max_rows:
            raise ValueError("Lamport projection is closed or exceeds capacity")
        return self.buffer[: 4 * rows]

    def close(self) -> None:
        """Collectively release IPC storage after all referencing graphs die."""
        from tokenspeed_kernel.thirdparty.cuda.cuda_ipc import cudart
        from tokenspeed_kernel.thirdparty.cuda.trtllm import (
            trtllm_destroy_ipc_workspace_for_reduce_scatter_fusion,
        )

        if self.handles is not None:
            torch.cuda.synchronize(self.buffer.device)
            dist.barrier(group=self.group)
            trtllm_destroy_ipc_workspace_for_reduce_scatter_fusion(
                self.handles, group=self.group
            )
            cudart.cudaFree(c_void_p(self.control_ptr))
            self.handles = None


@register_kernel(
    "communication",
    "projection_reduce_scatter",
    name="trtllm_projection_reduce_scatter",
    solution="trtllm",
    signatures=format_signatures(("partial",), "dense", {torch.bfloat16}),
)
def trtllm_projection_reduce_scatter(state, partial, rows):
    """Reduce BF16 [4*rows,H] partials into owned [rows,H] output.

    Prepare state collectively before capture. The native one-shot protocol
    publishes payloads into its own IPC ring and synchronizes their reuse;
    no symmetric-memory publication barrier is needed around local partials.
    Calls must be serialized on one stream. Output survives later state reuse.

    Args:
        state: Collectively prepared ProjectionLamportState.
        partial: Contiguous BF16 [4*rows,H] local GEMM partials.
        rows: Equal physical row count per rank, including padding.

    Returns:
        Owned BF16 [rows,H] tensor for the calling rank's token segment.
    """
    from tokenspeed_kernel.thirdparty.cuda.trtllm import trtllm_reducescatter_fusion

    destination = state.input_buffer(rows)
    if (
        partial.shape != destination.shape
        or partial.dtype != destination.dtype
        or partial.device != destination.device
        or not partial.is_contiguous()
    ):
        raise ValueError("Lamport projection partials have incompatible layout")
    out = torch.empty((rows, state.hidden), dtype=partial.dtype, device=partial.device)
    trtllm_reducescatter_fusion(
        reducescatter_in=partial,
        world_size=4,
        world_rank=state.group.rank(),
        token_num=4 * rows,
        hidden_dim=state.hidden,
        workspace_ptrs=state.workspace,
        trigger_completion_at_end=False,
        fp32_acc=True,
        num_token_current_rank=rows,
        pattern_code=0,
        launch_with_pdl=False,
        use_oneshot=True,
        reducescatter_out=out,
        add_in=None,
        residual_in=None,
        residual_out=None,
        norm_out=None,
        quant_out=None,
        scale_out=None,
        rms_gamma=None,
        rms_eps=None,
        scale_factor=None,
        layout_code=None,
        metadata=None,
    )
    return out
