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

"""Prefetch immutable output-channel shards beside attention for a local GEMM."""

from functools import partial

import torch
from tokenspeed_kernel.ops.communication.cuda_weight_prefetch import (
    CudaWeightPrefetchSource,
    CudaWeightPrefetchWorkspace,
)

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.linear import ColumnParallelLinear, DPRowParallelLinear
from tokenspeed.runtime.utils.env import global_server_args_dict


class WeightPrefetchLinear(ColumnParallelLinear):
    """Use compute TP through M=64 and N-shard prefetch for larger local batches.

    Keep an N shard for direct-layout weight prefetch and a separate K shard
    for the ordinary activation-TP projection. Both load checkpoint codes and
    scales verbatim. Large owners also contribute their K shard when other
    owners in the subgroup use compute TP; empty ranks must participate too.

    Args:
        input_size: Full GEMM input width (K).
        output_size: Full output width (N).
        parallel: Independent weight-storage TP mapping.
        quant_config: Checkpoint quantization configuration for this layer.
        prefix: Checkpoint parameter prefix, unchanged by sharding.
    """

    COMPUTE_MAX_ROWS = 64

    def __init__(self, input_size, output_size, parallel, quant_config, prefix):
        if input_size % 128 or output_size % 128:
            raise ValueError("Weight prefetch requires full 128x128 weight blocks")
        shard_rows = (
            (output_size // 128 + parallel.tp_size - 1) // parallel.tp_size
        ) * 128
        super().__init__(
            input_size=input_size,
            output_size=shard_rows * parallel.tp_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            params_dtype=None,
            quant_config=quant_config,
            output_sizes=None,
            prefix=prefix,
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
            tp_group=parallel.tp_group,
            use_presharded_weights=False,
            override_kernel_name=None,
            interleave_linear_and_gate=False,
        )
        if (
            not isinstance(self.quant_method, Fp8LinearMethod)
            or not self.quant_method.block_quant
            or not self.quant_method.quant_config.is_checkpoint_fp8_serialized
            or tuple(self.quant_method.quant_config.weight_block_size) != (128, 128)
            or self.quant_method.quant_config.scale_fmt is not None
            or global_server_args_dict["dense_gemm_backend"] != "auto"
        ):
            raise ValueError(
                "Weight-only TP requires serialized FP8/FP32 128x128 scales and dense_gemm_backend=auto"
            )
        self.logical_output_size = output_size
        self.parallel = parallel
        self.source: CudaWeightPrefetchSource | None = None
        self.workspace: CudaWeightPrefetchWorkspace | None = None
        self.compute_projection = DPRowParallelLinear(
            input_size,
            output_size,
            parallel=parallel,
            params_dtype=None,
            quant_config=quant_config,
            prefix=prefix,
        )

    def weight_loader_v2(self, param, loaded_weight):
        """Load both shard layouts without retaining the full checkpoint matrix."""
        if param is self.weight:
            expected = (self.logical_output_size, self.input_size)
            compute_param = self.compute_projection.weight
        elif param is self.weight_scale_inv:
            expected = (self.logical_output_size // 128, self.input_size // 128)
            compute_param = self.compute_projection.weight_scale_inv
        else:
            raise ValueError(
                "Weight prefetch accepts only FP8 weights and block scales"
            )
        if tuple(loaded_weight.shape) != expected:
            raise ValueError(
                f"Expected checkpoint shape {expected}, got {loaded_weight.shape}"
            )
        rows = param.shape[0]
        start = self.tp_rank * rows
        valid = max(0, min(rows, loaded_weight.shape[0] - start))
        param.data.zero_()
        if valid:
            param.data[:valid].copy_(loaded_weight[start : start + valid])
        self.compute_projection.weight_loader_v2(compute_param, loaded_weight)

    def prepare(self, workspace):
        """Publish shards, cache layer scales and bind scratch before memory profiling."""
        if self.source is not None:
            if self.workspace is not workspace:
                raise RuntimeError(
                    "Cannot replace a prepared weight-prefetch workspace"
                )
            return
        group = pg_manager.get_process_group("nccl", self.parallel.tp_group)
        self.source = CudaWeightPrefetchSource(
            group, self.weight, self.weight_scale_inv, self.logical_output_size
        )
        workspace.prepare(self.source)
        # Replace, rather than retain, the loading allocation. The shard remains
        # immutable and mapped for the entire lifetime of all serving graphs.
        self.weight.data = self.source.weight
        self.weight_scale_inv.data = self.source.scales
        self.workspace = workspace

    def prefetch(self):
        """Fork weight refresh before attention; immutable scales are already cached."""
        if self.source is None or self.workspace is None:
            raise RuntimeError("Weight prefetch must be prepared before forward")
        main = torch.cuda.current_stream(self.weight.device)
        self.workspace.stream.wait_stream(main)
        with torch.cuda.stream(self.workspace.stream):
            self.workspace.gather(self.source)

    def attend(self, attention, *args, **kwargs):
        """Keep prefetch and its stream join inside the attention graph boundary.

        Full decode and eligible KDA prefill graphs capture both streams.
        An eager attention break runs this entire region, so no graph segment
        ends with an unjoined auxiliary stream or retains stale replay events.
        """
        self.prefetch()
        output = attention(*args, **kwargs)
        torch.cuda.current_stream(self.weight.device).wait_stream(self.workspace.stream)
        return output

    @break_point
    def _attend_break(self, attention, *args, **kwargs):
        return self.attend(attention, *args, **kwargs)

    def wrap_attention(self, attention, num_tokens, capture_ready):
        if num_tokens <= self.COMPUTE_MAX_ROWS:
            return attention
        # Match the backend's boundary: retain inline KDA prefill capture when
        # metadata is ready, but contain the fork/join inside an eager break.
        return partial(self.attend if capture_ready else self._attend_break, attention)

    def forward(self, inputs, counts):
        if self.workspace is None:
            raise RuntimeError("Weight prefetch must be prepared before forward")
        if (
            len(counts) != self.parallel.world_size
            or any(count < 0 for count in counts)
            or inputs.ndim != 2
            or inputs.shape != (counts[self.parallel.rank], self.input_size)
        ):
            raise ValueError("Weight prefetch requires matching physical token counts")
        large = inputs.shape[0] > self.COMPUTE_MAX_ROWS
        small_counts = [
            count if count <= self.COMPUTE_MAX_ROWS else 0 for count in counts
        ]
        # Filter token owners, never participants: all peers supply K-sharded
        # partial results for the small owners, including large and idle peers.
        small_output, _ = self.compute_projection(
            inputs[:0] if large else inputs, small_counts
        )
        if not large:
            return small_output, None
        prepared = self.workspace.prepare_input(inputs, self.source)
        output = self.workspace.project(inputs, prepared, self.source)
        return output, None
