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

"""Kimi-K3 configuration and integration for projection-only tensor parallelism."""

import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import (
    AttentionLayerMapping,
    DenseLayerMapping,
    LinearAttnLayerMapping,
    Mapping,
)
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.attention.o_proj import (
    DistributedOutputProjection,
    initialize_projection_group,
)
from tokenspeed.runtime.layers.attention.o_proj import (
    make_output_projection as make_distributed_output_projection,
)
from tokenspeed.runtime.layers.attention.o_proj import (
    projection_mapping as make_projection_mapping,
)
from tokenspeed.runtime.layers.linear import RowParallelLinear
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext

logger = logging.getLogger(__name__)
ENV_NAME = "TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE"
A2A_ENV_NAME = "TOKENSPEED_KIMI_K3_O_PROJ_A2A_BACKEND"
RS_ENV_NAME = "TOKENSPEED_KIMI_K3_O_PROJ_RS_BACKEND"
DEFAULT_A2A_BACKEND = "flashinfer"
DEFAULT_RS_BACKEND = "triton_peer"


def projection_rs_backend(value: str) -> str:
    if value not in ("nccl", "triton_rsag", "triton_peer"):
        raise ValueError(f"{RS_ENV_NAME} must be nccl, triton_rsag, or triton_peer")
    return value


def projection_a2a_backend(value: str) -> str:
    if value not in ("nccl", "auto", "flashinfer", "flashinfer_quantized"):
        raise ValueError(f"Invalid {A2A_ENV_NAME}: {value}")
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
    return make_projection_mapping(mapping.rank, mapping.world_size, size)


def initialize_projection_parallelism(mapping: Mapping) -> None:
    """Agree on the setting before constructing/loading projection shards."""
    value = os.environ.get(ENV_NAME, "1")
    a2a_value = os.environ.get(A2A_ENV_NAME, DEFAULT_A2A_BACKEND)
    rs_value = os.environ.get(RS_ENV_NAME, DEFAULT_RS_BACKEND)
    if dist.is_initialized() and mapping.world_size > 1:
        # Every rank participates, even when its local setting is disabled or
        # malformed. Reject disagreement before entering differently sized groups.
        pg_manager.init_process_group(mapping.world_group, backend="gloo")
        values = [None] * mapping.world_size
        dist.all_gather_object(
            values,
            (value, a2a_value, rs_value),
            group=pg_manager.get_process_group("gloo", mapping.world_group),
        )
        if len(set(values)) != 1:
            raise ValueError(
                f"Projection TP/A2A/RS settings differ across ranks: {values}"
            )
    projection_a2a_backend(a2a_value)
    projection_rs_backend(rs_value)
    parallel = projection_mapping(mapping, value)
    if parallel.tp_size > 1:
        initialize_projection_group(parallel)
        logger.info(
            "Kimi-K3 output projection TP%s: %s", parallel.tp_size, parallel.tp_group
        )


def make_output_projection(
    *,
    mapping: Mapping,
    input_size: int,
    output_size: int,
    quant_config: QuantizationConfig | None,
    prefix: str,
    default_parallel: AttentionLayerMapping | LinearAttnLayerMapping,
    reduce_results: bool,
) -> tuple[RowParallelLinear, DistributedOutputProjection | None]:
    """Construct projection shards and optional DP/TP exchange, without full weights."""
    value = os.environ.get(ENV_NAME, "1")
    parallel = None if value == "1" else projection_mapping(mapping, value)
    return make_distributed_output_projection(
        parallel=parallel,
        input_size=input_size,
        output_size=output_size,
        quant_config=quant_config,
        prefix=prefix,
        default_parallel=default_parallel,
        reduce_results=reduce_results,
    )


def project_attention_output(
    inputs: torch.Tensor,
    linear: RowParallelLinear,
    exchange: DistributedOutputProjection | None,
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
