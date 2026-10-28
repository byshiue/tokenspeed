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

"""Gluon prefill convolution for the calibrated SM103 KDA layout."""

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
    CausalConv1dPrefillMetadata,
)
from tokenspeed_kernel.ops.conv._gluon.causal_conv1d import (
    gluon_causal_conv1d_prefill_kernel,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def _use_parallel_tactic(
    num_tokens: int, num_requests: int, *, exact_workload: bool
) -> bool:
    # Capacity graphs can replay a much smaller workload than their buffers.
    # Only exact host boundaries can select the calibrated parallel range.
    return (
        exact_workload
        and num_requests > 0
        and 2048 <= num_tokens <= 8192
        and num_tokens >= 128 * num_requests
    )


@register_kernel(
    "conv",
    "causal_conv1d_prefill",
    name="gluon_causal_conv1d_prefill_sm103",
    solution="gluon",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(10, 3),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "width": frozenset({4}),
        "state_layout": frozenset({"channel_major"}),
        "channels": frozenset({36864}),
        "channel_contiguous": frozenset({True}),
        "token_stride": frozenset({36864, 49376, 49408}),
        "weight_dtype": frozenset({torch.bfloat16}),
        "bias_dtype": frozenset({None, torch.bfloat16}),
        "state_dtype": frozenset({torch.bfloat16}),
        "indexed_state": frozenset({True}),
        "pad_slot_id": frozenset({-1}),
        "block_m": frozenset({8}),
    },
    priority=Priority.SPECIALIZED,
)
def gluon_causal_conv1d_prefill_sm103(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    activation: str | bool | None,
    pad_slot_id: int,
    validate_data: bool,
    *,
    prefill_metadata: CausalConv1dPrefillMetadata,
) -> torch.Tensor:
    """Convolve prefill tokens and update their indexed history.

    Args:
        x: BF16 input [36864, tokens]. Channels are contiguous; the token
            stride is 36864, 49376 or 49408. The latter two are QKV views
            of the fused K3 projection with 16-row or 128-row padding.
        weight: BF16 per-channel taps [36864, 4].
        bias: Optional channel bias.
        conv_states: BF16 history [slots, 36864, history], updated in place.
        query_start_loc: Int32/int64 packed request boundaries [requests + 1].
        cache_indices: Int32/int64 state slot for each request; -1 is padding.
        has_initial_state: Boolean history-valid flag for each request.
        activation: Apply SiLU for True, "silu", or "swish".
        pad_slot_id: Padding slot sentinel; this solution requires -1.
        validate_data: Check tensor shapes, strides and request-vector types.
        prefill_metadata: Shared eight-token program maps for this forward.

    Returns:
        Output with the input's shape and dtype. Padding rows are untouched.
        Both tactics consume the same metadata without per-layer preparation.
    """
    if pad_slot_id != -1 or prefill_metadata.block_m != 8:
        raise ValueError("SM103 convolution needs pad slot -1 and eight-token maps")
    if (
        x.ndim != 2
        or x.shape[0] != 36864
        or x.stride(0) != 1
        or x.stride(1) not in (36864, 49376, 49408)
    ):
        raise ValueError("SM103 convolution requires a calibrated K3 input layout")
    if (
        prefill_metadata.batch_indices.device != x.device
        or prefill_metadata.chunk_offsets.device != x.device
    ):
        raise ValueError("convolution metadata must be on the input device")
    if weight.shape != (x.shape[0], 4):
        raise ValueError("SM103 convolution requires four taps per input channel")
    if cache_indices is None or has_initial_state is None:
        raise ValueError("SM103 convolution requires indexed history state")
    if (
        conv_states.ndim != 3
        or conv_states.shape[1] != x.shape[0]
        or conv_states.shape[2] < 3
    ):
        raise ValueError("SM103 convolution needs three history values per channel")
    if validate_data:
        assert x.ndim == 2 and x.shape[0] == 36864 and x.stride(0) == 1
        assert weight.shape == (36864, 4) and weight.stride(1) == 1
        assert conv_states.ndim == 3 and conv_states.shape[1] == 36864
        assert conv_states.shape[2] >= 3
        assert x.dtype == weight.dtype == conv_states.dtype == torch.bfloat16
        assert query_start_loc.numel() == cache_indices.numel() + 1
        assert has_initial_state.numel() == cache_indices.numel()
        assert query_start_loc.dtype in (torch.int32, torch.int64)
        assert cache_indices.dtype in (torch.int32, torch.int64)
        assert has_initial_state.dtype == torch.bool
        for tensor in (
            query_start_loc,
            cache_indices,
            has_initial_state,
            prefill_metadata.batch_indices,
            prefill_metadata.chunk_offsets,
        ):
            assert tensor.ndim == 1 and tensor.is_contiguous()
            assert tensor.device == x.device
        if bias is not None:
            assert bias.shape == (36864,) and bias.stride(0) == 1
            assert bias.device == x.device

    # Fix the token stride even for a one-token view. empty_like can preserve
    # the projection's unused stride in that case and select the generic body.
    out = torch.empty(
        (x.shape[1], x.shape[0]), dtype=x.dtype, device=x.device
    ).transpose(0, 1)
    programs = prefill_metadata.batch_indices.numel()
    if programs == 0:
        return out
    boundaries_cpu = prefill_metadata.cu_seqlens_cpu
    # x can include an outer graph's padding. Plan from the live host mirror
    # when available; never infer actual work from the allocated token extent.
    tokens = int(boundaries_cpu[-1]) if boundaries_cpu is not None else x.shape[1]
    parallel = _use_parallel_tactic(
        tokens, cache_indices.numel(), exact_workload=boundaries_cpu is not None
    )
    block_n = 512 if parallel else 256
    channels = x.shape[0]
    channel_programs = triton.cdiv(channels, block_n)
    grid = (channel_programs, programs) if parallel else (programs, channel_programs)
    enable_pdl = pdl_enabled()
    options = {"num_warps": 4, "num_stages": 2}
    if parallel:
        options["maxnreg"] = 64
    if enable_pdl:
        options["launch_pdl"] = True
    args = (
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        prefill_metadata.batch_indices,
        prefill_metadata.chunk_offsets,
        out,
        x.stride(1),
        x.stride(0),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        out.stride(1),
        out.stride(0),
    )
    parameters = dict(
        CHANNELS=channels,
        WIDTH=4,
        HAS_BIAS=bias is not None,
        SILU=activation is True or activation in ("silu", "swish"),
        WINDOW=parallel,
        TOKEN_PARALLEL=parallel,
        BLOCK_M=8,
        BLOCK_N=block_n,
        WINDOW_ROWS=16,
        NUM_WARPS=4,
        ENABLE_PDL=enable_pdl,
        MODEL_TOKEN_STRIDE=x.stride(1),
        **options,
    )
    from tokenspeed_kernel.ops.tuning import is_warming_up

    if is_warming_up():
        # Compile the other tactic without launching it or updating history.
        # The normal launch below prepares this tactic during the same scope.
        other = parameters | {
            "WINDOW": not parallel,
            "TOKEN_PARALLEL": not parallel,
            "BLOCK_N": 256 if parallel else 512,
        }
        if parallel:
            del other["maxnreg"]
        else:
            other["maxnreg"] = 64
        gluon_causal_conv1d_prefill_kernel.warmup(*args, grid=(1, 1), **other)
    gluon_causal_conv1d_prefill_kernel[grid](*args, **parameters)
    return out
