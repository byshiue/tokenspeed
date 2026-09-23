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

"""Column-parallel linear execution over data-parallel token owners."""

import torch
import torch.distributed as dist
from tokenspeed_kernel import (
    fp8_linear_accepts_prepacked_input,
    fp8_linear_prepacked,
)
from tokenspeed_kernel.ops.communication.tokenspeed_a2a_lamport import (
    tokenspeed_a2a_lamport,
)
from tokenspeed_kernel.ops.communication.trtllm import (
    TrtllmAllGatherQuantState,
    TrtllmAllGatherState,
    trtllm_allgather,
    trtllm_allgather_fp8_quantize,
)
from tokenspeed_kernel.ops.gemm.flashinfer import has_flashinfer_fp8_blockscale

from tokenspeed.runtime.distributed.comm_ops import all_gather_single, all_to_all_single
from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.dp_row_parallel_linear import (
    prepare_lamport_projection_a2a,
)
from tokenspeed.runtime.layers.linear import ColumnParallelLinear


def column_projection_width(output_size: int, tp_size: int, block_rows: int) -> int:
    """Pad output width so each contiguous TP shard contains whole scale blocks."""
    if min(output_size, tp_size, block_rows) <= 0:
        raise ValueError("Projection sizes and scale-block rows must be positive")
    alignment = tp_size * block_rows
    return (output_size + alignment - 1) // alignment * alignment


class DPColumnParallelLinear:
    """Gather -> column Linear -> A2A, without changing attention/cache layout.

    Args:
        parallel: Prepared projection-only subgroup mapping.
        input_size: Complete input width K on every token owner.
        output_size: Logical output width, excluding tail padding.
        padded_output_size: Stored width N, divisible by TP and scale alignment.
        max_tokens: Prepared physical row capacity per rank.
        dtype: BF16 activation/output storage type.
        device: Current rank's selected CUDA device.
        allgather_backend: Explicit 'nccl' or 'trtllm'.
        a2a_backend: Explicit 'nccl' or 'tokenspeed_a2a_lamport'.

    Construct collectively before capture/cache budgeting. Calls and consumers
    must be serialized on one stream; close only after referencing graphs die.
    Compatible TP2/TP4 FP8 linears fuse gather and activation quantization by
    default. Other plans and larger row counts retain gather followed by Linear.
    """

    def __init__(
        self,
        parallel: DenseLayerMapping,
        input_size: int,
        output_size: int,
        padded_output_size: int,
        max_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        allgather_backend: str,
        a2a_backend: str,
    ):
        if allgather_backend not in ("nccl", "trtllm"):
            raise ValueError("Invalid column projection all-gather backend")
        if a2a_backend not in ("nccl", "tokenspeed_a2a_lamport"):
            raise ValueError("Invalid column projection A2A backend")
        if (
            parallel.tp_size not in (2, 4, 8, 16)
            or min(input_size, output_size, max_tokens) <= 0
            or padded_output_size < output_size
            or padded_output_size % parallel.tp_size
            or dtype != torch.bfloat16
        ):
            raise ValueError("Invalid BF16 column projection dimensions")
        self.parallel = parallel
        self.input_size = input_size
        self.output_size = output_size
        self.padded_output_size = padded_output_size
        self.max_tokens = max_tokens
        self.closed = False
        self.send = torch.empty((max_tokens, input_size), dtype=dtype, device=device)
        self.gathered = torch.empty(
            (parallel.tp_size * max_tokens, input_size), dtype=dtype, device=device
        )
        self.received = torch.empty(
            max_tokens * padded_output_size, dtype=dtype, device=device
        )
        group = pg_manager.get_process_group("nccl", parallel.tp_group)
        self.gather_state = None
        if allgather_backend == "trtllm":
            # Fused gather/quantization has TP2/TP4 numerical coverage. Its
            # inherited BF16 gather shares the IPC ring for other linear plans;
            # larger groups retain the ordinary gather until validated.
            if parallel.tp_size in (2, 4) and has_flashinfer_fp8_blockscale():
                self.gather_state = TrtllmAllGatherQuantState(
                    group,
                    min(max_tokens, 128),
                    input_size,
                    device,
                    torch.cuda.get_device_properties(device).multi_processor_count,
                )
            else:
                self.gather_state = TrtllmAllGatherState(
                    group, min(max_tokens, 128), input_size, device, True
                )
        self.lamport_a2a = None
        if a2a_backend == "tokenspeed_a2a_lamport":
            self.lamport_a2a = prepare_lamport_projection_a2a(
                group, max_tokens, padded_output_size, dtype, device
            )

    def _padded_inputs(self, inputs: torch.Tensor, rows: int) -> torch.Tensor:
        """Use identical owner padding for BF16 and fused-quantized gathers."""
        if inputs.shape[0] == rows and inputs.is_contiguous():
            return inputs
        send = self.send[:rows]
        send.zero_()
        send[: inputs.shape[0]].copy_(inputs)
        return send

    def gather_inputs(self, inputs: torch.Tensor, rows: int) -> torch.Tensor:
        """Gather equal padded rank-major token blocks; return borrowed storage."""
        send = self._padded_inputs(inputs, rows)
        if self.gather_state is not None and rows <= self.gather_state.max_rows:
            return trtllm_allgather(self.gather_state, send)
        gathered = self.gathered[: self.parallel.tp_size * rows]
        all_gather_single(gathered, send, self.parallel.tp_group, backend=None)
        return gathered

    def restore_outputs(self, local: torch.Tensor, rows: int) -> torch.Tensor:
        """Exchange output channel shards and return owned padded [rows,N]."""
        size = self.parallel.tp_size
        shard = self.padded_output_size // size
        if self.lamport_a2a is not None and rows <= self.lamport_a2a.max_rows:
            # Write directly into owned storage: later layers may reuse the
            # communicator while a caller still retains this projection result.
            output = local.new_empty((rows, self.padded_output_size))
            return tokenspeed_a2a_lamport(
                self.lamport_a2a, local, inverse=True, out=output
            )
        received = self.received[: rows * self.padded_output_size].view(
            size * rows, shard
        )
        all_to_all_single(received, local, self.parallel.tp_group, backend=None)
        # Received messages are source-channel-rank major. The copy restores
        # token-major channel order and gives the caller independent storage.
        return (
            received.view(size, rows, shard)
            .transpose(0, 1)
            .clone(memory_format=torch.contiguous_format)
            .view(rows, self.padded_output_size)
        )

    def forward(
        self, inputs: torch.Tensor, linear: ColumnParallelLinear, counts: list[int]
    ) -> torch.Tensor:
        """Return complete logical outputs [local_tokens,output_size].

        counts are agreed physical token counts for every world rank, including
        graph padding. Uneven/empty ranks participate using zero-padded rows;
        an entirely empty subgroup skips both collectives. Linear must load
        contiguous N shards with gather_output=False and no bias.
        """
        if (
            self.closed
            or len(counts) != self.parallel.world_size
            or any(count < 0 for count in counts)
            or inputs.ndim != 2
            or inputs.shape != (counts[self.parallel.rank], self.input_size)
            or inputs.dtype != self.send.dtype
            or inputs.device != self.send.device
            or linear.gather_output
            or linear.bias is not None
            or linear.tp_group != self.parallel.tp_group
            or linear.tp_rank != self.parallel.tp_rank
            or linear.tp_size != self.parallel.tp_size
            or linear.input_size != self.input_size
            or linear.output_size != self.padded_output_size
        ):
            raise ValueError("Incompatible column projection inputs, mapping or Linear")
        rows = max(counts[r] for r in self.parallel.tp_group)
        if (self.padded_output_size // self.parallel.tp_size) % 128:
            raise ValueError("Column projection shard splits a weight-scale block")
        if rows == 0:
            return inputs.new_empty((0, self.output_size))
        if rows > self.max_tokens:
            raise ValueError("Column projection exceeds prepared row capacity")
        plan = linear.quant_method.prepared_linear_plan(linear)
        num_tokens = self.parallel.tp_size * rows
        if (
            isinstance(self.gather_state, TrtllmAllGatherQuantState)
            and rows <= self.gather_state.max_rows
            and fp8_linear_accepts_prepacked_input(plan, num_tokens)
        ):
            # Preserve the layer's prepared GEMM/scale contract. Unsupported
            # plans (including BF16, MXFP8 and other GEMM backends) stay below.
            values, scales = trtllm_allgather_fp8_quantize(
                self.gather_state, self._padded_inputs(inputs, rows)
            )
            local = fp8_linear_prepacked(
                plan, values, linear.weight, scales, num_tokens, inputs.dtype, out=None
            )
        else:
            gathered = self.gather_inputs(inputs, rows)
            local, _ = linear(gathered)
        output = self.restore_outputs(local.contiguous(), rows)
        return output[: inputs.shape[0], : self.output_size]

    def close(self) -> None:
        """Collectively close communicators after all referencing graphs die."""
        if self.closed:
            return
        torch.cuda.synchronize(self.send.device)
        if self.lamport_a2a is not None:
            dist.barrier(group=self.lamport_a2a.group)
            self.lamport_a2a = None
        if self.gather_state is not None:
            self.gather_state.close()
        self.closed = True
