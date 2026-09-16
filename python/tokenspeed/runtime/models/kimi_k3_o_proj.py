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

"""Attention-DP to output-projection-TP exchange for Kimi-K3.

Only the projection changes ownership. Attention/cache rows and the residual
stream remain local to their original DP rank, including on empty ranks.
"""

import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.flashinfer import (
    create_projection_a2a,
    flashinfer_projection_a2a,
)
from tokenspeed_kernel.ops.communication.triton import triton_pack_projection_input

from tokenspeed.runtime.distributed.comm_ops import all_to_all_single, reduce_scatter
from tokenspeed.runtime.distributed.mapping import (
    AttentionLayerMapping,
    DenseLayerMapping,
    LinearAttnLayerMapping,
    Mapping,
)
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.linear import RowParallelLinear
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext

logger = logging.getLogger(__name__)
ENV_NAME = "TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE"
A2A_ENV_NAME = "TOKENSPEED_KIMI_K3_O_PROJ_A2A_BACKEND"
# Conservative measured envelope; larger/uneven batches use the NCCL path.
FLASHINFER_MAX_TOKENS = 16


def projection_a2a_backend(value: str) -> str:
    if value not in ("nccl", "auto", "flashinfer"):
        raise ValueError(f"{A2A_ENV_NAME} must be nccl, auto, or flashinfer")
    return value


def projection_mapping(mapping: Mapping, value: str) -> DenseLayerMapping:
    """Resolve projection-only TP without changing attention/cache mappings."""
    try:
        size = int(value)
    except ValueError as exc:
        raise ValueError(f"{ENV_NAME} must be a positive integer") from exc
    if size < 1 or mapping.world_size % size:
        raise ValueError(f"{ENV_NAME} must be a positive divisor of world size")
    if size > 1 and (
        mapping.attn.dp_size != mapping.world_size
        or mapping.linear_attn.tp_size != 1
        or mapping.moe.ep_size != mapping.world_size
        or mapping.pp_size != 1
    ):
        raise ValueError(
            f"{ENV_NAME}>1 requires attention/linear-attention TP1, "
            "attention DP == MoE EP == world size, and PP1"
        )
    return DenseLayerMapping(
        rank=mapping.rank,
        world_size=mapping.world_size,
        tp_size=size,
        dp_size=mapping.world_size // size,
    )


def initialize_projection_parallelism(mapping: Mapping) -> None:
    """Agree on the setting before constructing/loading projection shards."""
    value = os.environ.get(ENV_NAME, "1")
    a2a_value = os.environ.get(A2A_ENV_NAME, "nccl")
    if dist.is_initialized() and mapping.world_size > 1:
        # Every rank participates, even when its local setting is disabled or
        # malformed. Reject disagreement before entering differently sized groups.
        pg_manager.init_process_group(mapping.world_group, backend="gloo")
        values = [None] * mapping.world_size
        dist.all_gather_object(
            values,
            (value, a2a_value),
            group=pg_manager.get_process_group("gloo", mapping.world_group),
        )
        if len(set(values)) != 1:
            raise ValueError(
                f"Projection TP/A2A settings differ across ranks: {values}"
            )
    projection_a2a_backend(a2a_value)
    parallel = projection_mapping(mapping, value)
    if parallel.tp_size > 1:
        pg_manager.init_process_group(parallel.tp_group, backend=None)
        # Materialize backend/NCCL resources outside any model CUDA graph.
        probe = torch.zeros((parallel.tp_size, 1), dtype=torch.bfloat16, device="cuda")
        received = torch.empty_like(probe)
        all_to_all_single(received, probe, parallel.tp_group, backend=None)
        reduce_scatter(received, parallel.tp_group, backend=None)
        logger.info(
            "Kimi-K3 output projection TP%s: %s", parallel.tp_size, parallel.tp_group
        )


class ProjectionWorkspace:
    """Reusable exchange scratch, shared by sequential KDA/MLA layers."""

    def __init__(
        self,
        max_tokens: int,
        max_input_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # All-to-all exchanges P * M * (K/P) elements: M*K, not P*M*K.
        self.max_tokens = max_tokens
        self.send = torch.empty(max_tokens * max_input_size, dtype=dtype, device=device)
        self.recv = torch.empty_like(self.send)
        self.a2a = None
        self.a2a_reason = None
        self._a2a_initialized = False

    def initialize_a2a(self, parallel: DenseLayerMapping, max_input_size: int) -> None:
        """Collectively prepare optional IPC/JIT resources before graph capture.

        The communicator is shared by sequential layers and lives as long as
        this workspace. Never replace it while captured graphs reference it.
        """
        if self._a2a_initialized:
            return
        backend = projection_a2a_backend(os.environ.get(A2A_ENV_NAME, "nccl"))
        if backend != "nccl":
            self.a2a, self.a2a_reason = create_projection_a2a(
                group=pg_manager.get_process_group("nccl", parallel.tp_group),
                max_elems=min(self.max_tokens, FLASHINFER_MAX_TOKENS) * max_input_size,
                dtype=self.send.dtype,
                device=self.send.device,
                backend=backend,
            )
            logger.info(
                "Projection A2A: %s (%s)",
                "flashinfer" if self.a2a is not None else "nccl",
                self.a2a_reason or "NVLink",
            )
        self._a2a_initialized = True

    def use_flashinfer(
        self, parallel: DenseLayerMapping, counts: list[int], input_size: int
    ) -> bool:
        """Choose identically on every subgroup rank, using only host metadata."""
        rows = counts[parallel.rank]
        return (
            self.a2a is not None
            and 0 < rows <= FLASHINFER_MAX_TOKENS
            and input_size % (8 * parallel.tp_size) == 0
            and all(counts[r] == rows for r in parallel.tp_group)
        )

    def close(self) -> None:
        """Collectively release IPC resources after all referencing graphs die."""
        if self.a2a is not None:
            self.a2a.close()
            self.a2a = None


class KimiOutputProjection:
    """Run a sharded projection and return outputs to the original DP owner.

    The row-parallel Linear remains registered as self_attn.o_proj, preserving
    checkpoint names and the existing packed-weight/scale shard loaders.
    """

    def __init__(self, parallel: DenseLayerMapping, input_size: int) -> None:
        self.parallel = parallel
        self.input_size = input_size
        self.workspace: ProjectionWorkspace | None = None

    def forward(
        self, inputs: torch.Tensor, linear: RowParallelLinear, counts: list[int]
    ) -> torch.Tensor:
        """Project local [tokens, channels] rows using subgroup token counts.

        Returns complete [local_tokens, hidden] rows in the original order.
        All subgroup ranks must call, including ranks with no local tokens.
        """
        parallel = self.parallel
        if (
            len(counts) != parallel.world_size
            or counts[parallel.rank] != inputs.shape[0]
            or any(count < 0 for count in counts)
            or inputs.ndim != 2
            or inputs.shape[1] != self.input_size
        ):
            raise ValueError(
                "Output projection requires matching collective token counts"
            )
        max_tokens = max(counts[r] for r in parallel.tp_group)
        if max_tokens == 0:
            return inputs.new_empty((0, linear.output_size))
        workspace = self.workspace
        if workspace is None or max_tokens > workspace.max_tokens:
            raise RuntimeError(
                "Output projection workspace must be prepared before forward"
            )
        if (
            inputs.dtype != workspace.send.dtype
            or inputs.device != workspace.send.device
        ):
            raise ValueError(
                "Output projection inputs must match the prepared workspace"
            )
        size = parallel.tp_size
        shard = self.input_size // size
        elements = max_tokens * self.input_size
        send = workspace.send[:elements].view(size, max_tokens, shard)
        recv = workspace.recv[:elements].view(size * max_tokens, shard)
        # Equal-sized messages permit capture and avoid device-to-host counts.
        # Rank-major output segments are exactly reduce-scatter's owner ordering.
        # One row already has rank-major byte order. Other shapes fuse the
        # transpose and padding, preserving the original BF16/FP16 values.
        if workspace.use_flashinfer(parallel, counts, self.input_size):
            recv = flashinfer_projection_a2a(workspace.a2a, inputs.contiguous())
        else:
            packed = triton_pack_projection_input(inputs, send)
            all_to_all_single(recv, packed, parallel.tp_group, backend=None)
        partial, _ = linear(recv)
        output = reduce_scatter(partial.contiguous(), parallel.tp_group, backend=None)
        return output[: inputs.shape[0]]


def make_output_projection(
    *,
    mapping: Mapping,
    input_size: int,
    output_size: int,
    quant_config: QuantizationConfig | None,
    prefix: str,
    default_parallel: AttentionLayerMapping | LinearAttnLayerMapping,
    reduce_results: bool,
) -> tuple[RowParallelLinear, KimiOutputProjection | None]:
    """Construct projection shards and optional DP/TP exchange, without full weights."""
    value = os.environ.get(ENV_NAME, "1")
    parallel = None if value == "1" else projection_mapping(mapping, value)
    enabled = parallel is not None and parallel.tp_size > 1
    selected = parallel if enabled else default_parallel
    if input_size % selected.tp_size:
        raise ValueError("Output projection channels must divide projection TP size")
    linear = RowParallelLinear(
        input_size,
        output_size,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=False,
        params_dtype=None,
        reduce_results=False if enabled else reduce_results,
        quant_config=quant_config,
        prefix=prefix,
        tp_rank=selected.tp_rank,
        tp_size=selected.tp_size,
        tp_group=selected.tp_group,
        use_presharded_weights=False,
        override_kernel_name=None,
        interleave_linear_and_gate=False,
    )
    if enabled:
        # Mixed checkpoints resolve quantization per Linear. Inspect that
        # resolved method, not the model-level default (e.g. MXFP8 vs FP8).
        resolved = getattr(linear.quant_method, "quant_config", None)
        block = getattr(resolved, "weight_block_size", None)
        alignment = block[-1] if block else getattr(resolved, "group_size", 1)
        if linear.input_size_per_partition % alignment:
            raise ValueError("Output projection TP shard splits a quantization block")
    return linear, KimiOutputProjection(parallel, input_size) if enabled else None


def project_attention_output(
    inputs: torch.Tensor,
    linear: RowParallelLinear,
    exchange: KimiOutputProjection | None,
    ctx: "ForwardContext",
) -> torch.Tensor:
    """Apply the ordinary projection or its projection-only TP exchange."""
    if exchange is None:
        output, _ = linear(inputs)
        return output
    counts = ctx.collective_global_num_tokens
    if counts is None:
        counts = ctx.global_num_tokens
    if counts is None:
        raise ValueError("Output projection TP requires collective token counts")
    return exchange.forward(inputs, linear, counts)
