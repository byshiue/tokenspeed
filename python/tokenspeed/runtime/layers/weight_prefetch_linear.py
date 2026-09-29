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

"""Share K-sharded weights between activation TP and full-weight prefetch."""

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
from tokenspeed.runtime.layers.linear import DPRowParallelLinear
from tokenspeed.runtime.utils.env import global_server_args_dict


class WeightPrefetchLinear(DPRowParallelLinear):
    """Use activation TP below 128 local rows and prefetch weights otherwise.

    Both routes share one K shard and the ordinary checkpoint loader. Large
    owners still compute the small owners' partial outputs when a group is
    mixed; every peer must enter the same activation collectives.

    Args:
        input_size: Full GEMM input width (K).
        output_size: Full output width (N).
        parallel: Independent weight-storage TP mapping.
        quant_config: Checkpoint quantization configuration for this layer.
        prefix: Checkpoint parameter prefix, unchanged by sharding.
    """

    PREFETCH_MIN_ROWS = 128

    def __init__(self, input_size, output_size, parallel, quant_config, prefix):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel=parallel,
            params_dtype=None,
            quant_config=quant_config,
            prefix=prefix,
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
        self.source: CudaWeightPrefetchSource | None = None
        self.workspace: CudaWeightPrefetchWorkspace | None = None

    def prepare(self, workspace):
        """Publish immutable shards and bind shared scratch before memory profiling."""
        if self.source is not None:
            if self.workspace is not workspace:
                raise RuntimeError(
                    "Cannot replace a prepared weight-prefetch workspace"
                )
            return
        group = pg_manager.get_process_group("nccl", self.parallel.tp_group)
        self.source = CudaWeightPrefetchSource(
            group, self.weight, self.weight_scale_inv, self.input_size
        )
        # Replace, rather than retain, the loading allocation. The shard remains
        # immutable and mapped for the entire lifetime of all serving graphs.
        self.weight.data = self.source.weight
        self.weight_scale_inv.data = self.source.scales
        # Refresh the ordinary sharded GEMM plan against the published storage.
        self.quant_method.process_weights_after_loading(self)
        self.workspace = workspace

    def prefetch(self):
        """Fork weight/scales refresh before attention on this layer's main stream."""
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
        if num_tokens < self.PREFETCH_MIN_ROWS:
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
        large = inputs.shape[0] >= self.PREFETCH_MIN_ROWS
        small_counts = [
            count if count < self.PREFETCH_MIN_ROWS else 0 for count in counts
        ]
        # Filtering only the owners, not the participants, keeps mixed groups
        # collective-safe. Each peer contributes its one persistent K shard.
        small_output, _ = super().forward(inputs[:0] if large else inputs, small_counts)
        if not large:
            return small_output, None
        shape = (self.logical_output_size, self.input_size)
        prepared = self.workspace.prepare_input(inputs, shape)
        output = self.workspace.project(inputs, prepared, shape)
        return output, None
