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

"""CuTe DSL prefill convolution with fixed tactics and dynamic request maps."""

import math
from dataclasses import dataclass
from threading import Lock

import torch
from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
    CausalConv1dPrefillMetadata,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel.ops.conv._cute_dsl.causal_conv1d import CausalConv1dPrefill

    _AVAILABLE = True
except (ImportError, NotImplementedError):
    _AVAILABLE = False


@dataclass(frozen=True)
class PrefillConfig:
    block_m: int
    block_n: int
    threads: int
    token_parallel: bool
    min_blocks_per_mp: int


_COMPILED = {}
_COMPILE_LOCK = Lock()
_ROLLING = PrefillConfig(8, 256, 128, False, 1)
_PARALLEL = PrefillConfig(8, 512, 128, True, 10)


def _prepare_causal_conv1d_prefill(
    x,
    weight,
    bias,
    state,
    slots,
    history,
    boundaries,
    out,
    *,
    metadata,
    config: PrefillConfig,
    silu: bool,
    enable_pdl: bool,
):
    """Prepare one tactic without launching it or changing output/history."""
    import cuda.bindings.driver as cuda
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import make_ptr

    if metadata.block_m != config.block_m:
        raise ValueError("convolution and metadata must use the same token tile")
    channels = x.shape[0]
    if x.ndim != 2 or x.stride(0) != 1:
        raise ValueError("convolution input must have contiguous channels")
    if out.shape != x.shape or out.stride() != (1, channels):
        raise ValueError("convolution output must use packed token rows")
    if weight.shape != (channels, 4):
        raise ValueError("convolution requires four taps per channel")
    if state.ndim != 3 or state.shape[1] != channels or state.shape[2] < 3:
        raise ValueError("convolution requires three history values per channel")
    if any(t.dtype != torch.bfloat16 for t in (x, weight, state, out)):
        raise ValueError("convolution requires BF16 input, weight, state and output")
    if bias is not None and (
        bias.shape != (channels,) or bias.dtype != x.dtype or not bias.is_contiguous()
    ):
        raise ValueError("convolution bias must be a contiguous BF16 channel vector")
    tensors = (
        x,
        weight,
        x if bias is None else bias,
        state,
        slots,
        history,
        boundaries,
        metadata.batch_indices,
        metadata.chunk_offsets,
        out,
    )
    if any(t.device != x.device for t in tensors[7:9]):
        raise ValueError("convolution metadata must be on the input device")
    if any(t.device != x.device for t in tensors):
        raise ValueError("convolution tensors must be on the same device")
    if (
        slots.dtype not in (torch.int32, torch.int64)
        or boundaries.dtype not in (torch.int32, torch.int64)
        or history.dtype != torch.bool
    ):
        raise ValueError(
            "request slots and boundaries must be integers; history must be boolean"
        )
    if any(t.ndim != 1 or not t.is_contiguous() for t in tensors[4:9]):
        raise ValueError("request metadata must use contiguous vectors")
    if not metadata.batch_indices.numel():
        return None, (), None
    # CUDA stores torch.bool as one byte; CuTe i1 pointers are register-only.
    types = {
        torch.bfloat16: cutlass.BFloat16,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
        torch.bool: cutlass.Uint8,
    }
    alignment = math.gcd(16, x.data_ptr(), out.data_ptr())
    weight_alignment = math.gcd(16, weight.data_ptr())
    args = tuple(
        make_ptr(
            types[t.dtype],
            t.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=(
                alignment
                if i in (0, 9)
                else weight_alignment if i == 1 else t.element_size()
            ),
        )
        for i, t in enumerate(tensors)
    )
    args += (cutlass.Int32(metadata.batch_indices.numel()),)
    # All entries describe fixed model/tactic layouts. No token count, request
    # count or boundary value enters this cache key.
    key = (
        x.device.index,
        torch.cuda.get_device_capability(x.device),
        config,
        channels,
        x.stride(1),
        alignment,
        weight_alignment,
        state.stride(),
        weight.stride(),
        tuple(t.dtype for t in tensors),
        bias is not None,
        silu,
        enable_pdl,
    )
    with torch.cuda.device(x.device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = _COMPILED.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "warm the convolution tactic before CUDA Graph capture"
                )
            with _COMPILE_LOCK:
                compiled = _COMPILED.get(key)
                if compiled is None:
                    kernel = CausalConv1dPrefill(
                        channels=channels,
                        input_stride=x.stride(1),
                        pointer_alignment=alignment,
                        weight_alignment=weight_alignment,
                        state_strides=state.stride(),
                        weight_strides=weight.stride(),
                        block_m=config.block_m,
                        block_n=config.block_n,
                        threads=config.threads,
                        token_parallel=config.token_parallel,
                        min_blocks_per_mp=config.min_blocks_per_mp,
                        use_mixed_add=torch.cuda.get_device_capability(x.device)
                        >= (10, 0),
                        has_bias=bias is not None,
                        silu=silu,
                        use_pdl=enable_pdl,
                    )
                    compiled = cute.compile(kernel, *args, stream)
                    _COMPILED[key] = compiled
    return compiled, args, stream


def launch_causal_conv1d_prefill(
    x,
    weight,
    bias,
    state,
    slots,
    history,
    boundaries,
    out,
    *,
    metadata,
    config: PrefillConfig,
    silu: bool,
    enable_pdl: bool,
):
    """Run four-tap BF16 convolution and update each non-padding request's state.

    Input and output have shape [channels, tokens] with contiguous channels.
    Weight has shape [channels, 4]. State has shape [slots, channels, history]
    with at least three history values. Slots use -1 for padding. Metadata must
    use config.block_m. Output must not alias input. Output and state are
    caller-owned and updated in place. Returns the compiled executor, or None
    for an empty map. Token and request counts do not specialize the executor.
    """
    with torch.cuda.device(x.device):
        compiled, args, stream = _prepare_causal_conv1d_prefill(
            x,
            weight,
            bias,
            state,
            slots,
            history,
            boundaries,
            out,
            metadata=metadata,
            config=config,
            silu=silu,
            enable_pdl=enable_pdl,
        )
        if compiled is not None:
            compiled(*args, stream)
    return compiled


def _use_parallel_tactic(
    num_tokens: int, num_requests: int, *, exact_workload: bool
) -> bool:
    # Capacity graphs may replay far fewer live tokens than their buffer size.
    return (
        exact_workload
        and num_requests > 0
        and 2048 <= num_tokens <= 8192
        and num_tokens >= 128 * num_requests
    )


if _AVAILABLE:

    @register_kernel(
        "conv",
        "causal_conv1d_prefill",
        name="cute_dsl_causal_conv1d_prefill",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
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
    def cute_dsl_causal_conv1d_prefill(
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
        """Convolve KDA prefill tokens through the common convolution facade.

        Args:
            x: BF16 [36864, tokens], with contiguous channels. Token stride
                is 36864, 49376 or 49408, including fused projection views.
            weight: BF16 [36864, 4] taps in oldest-to-newest order.
            bias: Optional contiguous BF16 channel vector.
            conv_states: BF16 [slots, 36864, history], updated in place.
            query_start_loc: Packed request boundaries [requests + 1].
            cache_indices: Integer state slots [requests]; -1 marks padding.
            has_initial_state: Boolean history-valid flags [requests].
            activation: True, "silu", or "swish" applies SiLU; None/False omit it.
            pad_slot_id: Padding sentinel; this solution requires -1.
            validate_data: Check matching request-vector lengths.
            prefill_metadata: Eight-token maps shared by all layers this forward.

        Returns:
            BF16 output with the input shape and packed token rows. Padding
            rows are untouched. Small and capacity workloads use the rolling
            tactic in the same kernel; exact large chunks use parallel windows.
        """
        if pad_slot_id != -1:
            raise ValueError("CuTe convolution requires padding slot -1")
        if activation not in (None, False, True, "silu", "swish"):
            raise ValueError("convolution activation must be SiLU or disabled")
        if cache_indices is None or has_initial_state is None:
            raise ValueError("CuTe convolution requires indexed history state")
        if validate_data and (
            query_start_loc.numel() != cache_indices.numel() + 1
            or has_initial_state.numel() != cache_indices.numel()
        ):
            raise ValueError("convolution request vectors must have matching lengths")
        out = torch.empty((x.shape[1], x.shape[0]), dtype=x.dtype, device=x.device).T
        boundaries_cpu = prefill_metadata.cu_seqlens_cpu
        tokens = int(boundaries_cpu[-1]) if boundaries_cpu is not None else x.shape[1]
        parallel = _use_parallel_tactic(
            tokens, cache_indices.numel(), exact_workload=boundaries_cpu is not None
        )
        config = _PARALLEL if parallel else _ROLLING
        silu = activation is True or activation in ("silu", "swish")
        enable_pdl = pdl_enabled()
        args = (
            x,
            weight,
            bias,
            conv_states,
            cache_indices,
            has_initial_state,
            query_start_loc,
            out,
        )
        from tokenspeed_kernel.ops.tuning import is_warming_up

        if is_warming_up():
            # Compile the other tactic without updating history a second time.
            _prepare_causal_conv1d_prefill(
                *args,
                metadata=prefill_metadata,
                config=_ROLLING if parallel else _PARALLEL,
                silu=silu,
                enable_pdl=enable_pdl,
            )
        launch_causal_conv1d_prefill(
            *args,
            metadata=prefill_metadata,
            config=config,
            silu=silu,
            enable_pdl=enable_pdl,
        )
        return out
