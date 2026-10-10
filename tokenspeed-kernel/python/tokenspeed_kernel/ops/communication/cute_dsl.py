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

"""Prepared, fully resident TP4 block-FP8 column projection on GB300."""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

if TYPE_CHECKING:
    from tokenspeed_kernel.thirdparty.cute_dsl.fused_tp4_projection.workspace import (
        FusedTP4ProjectionState,
    )

cute_dsl_fused_tp4_projection = error_fn


def create_fused_tp4_projection_state(
    group: dist.ProcessGroup,
    device: torch.device,
) -> "FusedTP4ProjectionState":
    """Collectively allocate and compile a workspace before graph capture.

    Args:
        group: Four node-local eligible peers, in output-shard order.
        device: This peer's GB300 CUDA device.

    Returns:
        Model-owned state; serialize calls and close after destroying graphs.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.fused_tp4_projection.workspace import (
        FusedTP4ProjectionState,
    )

    return FusedTP4ProjectionState(group, device)


def fused_tp4_projection_supported(
    group_size: int,
    input_size: int,
    output_size: int,
    max_rows: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Return whether the static layout and device match the validated schedule.

    Args:
        group_size: Number of ranks in a node-local projection group.
        input_size: Full input channel count.
        output_size: Full padded output channel count.
        max_rows: Prepared capacity per owner; calls require exactly 128 rows.
        dtype: Unquantized activation/output dtype.
        device: Device owning the workspace.

    Returns:
        Whether this fixed schedule can be prepared. The caller must separately
        agree on node locality and eligibility across the group.
    """
    return (
        current_platform().is_nvidia
        and device.type == "cuda"
        and (group_size, input_size, output_size) == (4, 7168, 49664)
        and max_rows >= 128
        and dtype == torch.bfloat16
        and torch.cuda.get_device_capability(device) == (10, 3)
        and torch.cuda.get_device_properties(device).multi_processor_count == 152
    )


if current_platform().is_nvidia:

    @register_kernel(
        "communication",
        "column_projection",
        name="cute_dsl_fused_tp4_projection",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 3),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures={
            format_signature(
                inputs=dense_tensor_format(torch.bfloat16),
                weight=tensor_format(
                    "mxfp8",
                    torch.float8_e4m3fn,
                    scale=ScaleFormat(
                        storage_dtype=torch.float32,
                        granularity="block",
                        block_shape=(128, 128),
                    ),
                ),
                out=dense_tensor_format(torch.bfloat16),
            )
        },
        priority=Priority.SPECIALIZED,
    )
    def cute_dsl_fused_tp4_projection(
        state: "FusedTP4ProjectionState",
        inputs: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize/exchange owner inputs, multiply, and inverse-exchange outputs.

        Args:
            state: Collectively prepared workspace, shared only by serial calls.
            inputs: Contiguous BF16 [128, 7168] owner activations on every rank.
            weight: Contiguous FP8 E4M3 [12416, 7168] local output-channel shard.
            weight_scales: Contiguous FP32 [97, 56] scales for 128x128 blocks.
            out: Caller-owned contiguous BF16 [128, 49664] destination.

        Returns:
            out, valid across subsequent calls. All four peers must participate
            in the same order. Destroy referencing graphs before closing state.
        """
        return state.run(inputs, weight, weight_scales, out)
