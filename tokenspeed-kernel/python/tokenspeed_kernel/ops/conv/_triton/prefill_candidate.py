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

"""Unregistered prefill experiments. Serving continues to use causal_conv1d.py."""

from dataclasses import dataclass
from typing import Literal

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
    CausalConv1dPrefillMetadata,
)
from tokenspeed_kernel.ops.conv._gluon.prefill_candidate import (
    gluon_causal_conv1d_prefill_kernel,
)
from tokenspeed_kernel.ops.conv._triton.causal_conv1d import _causal_conv1d_fwd_kernel


@dataclass(frozen=True)
class PrefillCandidateConfig:
    implementation: Literal[
        "serial",
        "window",
        "direct",
        "gluon_serial",
        "gluon_direct",
        "gluon_window",
        "gluon_window_r64",
        "gluon_window_r48",
    ]
    block_m: int
    block_n: int
    num_warps: int

    def __post_init__(self):
        if self.implementation not in (
            "serial",
            "window",
            "direct",
            "gluon_serial",
            "gluon_direct",
            "gluon_window",
            "gluon_window_r64",
            "gluon_window_r48",
        ):
            raise ValueError("unsupported prefill implementation")
        if self.block_m not in (4, 8, 16, 32, 64, 128):
            raise ValueError("block_m must be 4, 8, 16, 32, 64, or 128")
        if self.block_n not in (64, 128, 256, 512, 1024) or self.num_warps not in (
            4,
            8,
        ):
            raise ValueError("unsupported channel tile or warp count")


_SERIAL_PREFILL = PrefillCandidateConfig("gluon_serial", 8, 256, 4)
_PARALLEL_PREFILL = PrefillCandidateConfig("gluon_window_r64", 8, 512, 4)


@dataclass(frozen=True)
class PrefillTacticPolicy:
    """Choose between rolling and parallel variants of the same Gluon kernel.

    Both tactics use eight-token metadata. Thresholds are explicit results of
    offline calibration for a fixed GPU, model layout, dtype and compiler.
    Selection requires exact host-known work, not just buffer capacities.
    Capacity-only graphs use the rolling tactic because replay can shrink the
    live workload. Neither selection path reads a CUDA tensor.
    """

    min_parallel_tokens: int
    min_tokens_per_request: int
    max_parallel_tokens: int

    def __post_init__(self):
        if self.min_parallel_tokens <= 0:
            raise ValueError("min_parallel_tokens must be positive")
        if self.min_tokens_per_request <= 0:
            raise ValueError("min_tokens_per_request must be positive")
        if self.max_parallel_tokens < self.min_parallel_tokens:
            raise ValueError("max_parallel_tokens must be at least the minimum")

    @property
    def block_m(self) -> int:
        return 8

    def select(
        self, num_tokens: int, num_requests: int, *, exact_workload: bool
    ) -> PrefillCandidateConfig:
        """Return one tactic within an explicitly calibrated workload range.

        The average length guards against batches with many costly first
        tiles, even when their total token count is large. Padded request slots
        count conservatively as requests. Set exact_workload only when the
        extents describe actual packed work throughout execution. Unknown or
        out-of-range work uses the rolling variant of the same kernel.
        """
        if num_tokens < 0 or num_requests < 0:
            raise ValueError("token and request extents must be nonnegative")
        if (
            exact_workload
            and num_requests > 0
            and self.min_parallel_tokens <= num_tokens <= self.max_parallel_tokens
            and num_tokens >= self.min_tokens_per_request * num_requests
        ):
            return _PARALLEL_PREFILL
        return _SERIAL_PREFILL


@triton.jit
def _load_window_values(
    X,
    State,
    start,
    rows,
    channels,
    slot,
    length,
    history,
    sx_token,
    sx_channel,
    ss_slot,
    ss_channel,
    ss_token,
    valid_rows,
    valid_channels,
    WIDTH: tl.constexpr,
):
    input_mask = (
        valid_rows[:, None]
        & (rows[:, None] >= 0)
        & (rows[:, None] < length)
        & valid_channels[None, :]
    )
    # Keep strides dynamic, but expose unit strides inside uniform branches.
    # Otherwise the compiler can place adjacent lanes on distant token rows.
    if sx_channel == 1:
        values = tl.load(
            X + (start + rows[:, None]) * sx_token + channels[None, :],
            input_mask,
            other=0,
        )
    elif sx_token == 1:
        values = tl.load(
            X + start + rows[:, None] + channels[None, :] * sx_channel,
            input_mask,
            other=0,
        )
    else:
        values = tl.load(
            X + (start + rows[:, None]) * sx_token + channels[None, :] * sx_channel,
            input_mask,
            other=0,
        )
    old = tl.load(
        State
        + slot * ss_slot
        + channels[None, :] * ss_channel
        + (rows[:, None] + WIDTH - 1) * ss_token,
        valid_rows[:, None]
        & (rows[:, None] < 0)
        & (rows[:, None] >= 1 - WIDTH)
        & history
        & valid_channels[None, :],
        other=0,
    )
    return tl.where(rows[:, None] < 0, old, values)


@triton.jit(
    do_not_specialize=["sx_token", "sx_channel", "so_token", "so_channel"],
    do_not_specialize_on_alignment=["Batches", "Offsets"],
)
def _causal_conv1d_prefill_candidate_kernel(
    X,
    Weight,
    Bias,
    State,
    Slots,
    History,
    Boundaries,
    Batches,
    Offsets,
    Out,
    sx_token,
    sx_channel,
    # Weight and state strides belong to fixed model layouts. Both input and
    # output strides can follow token counts for different tensor layouts.
    sw_channel: tl.constexpr,
    sw_tap: tl.constexpr,
    ss_slot: tl.constexpr,
    ss_channel: tl.constexpr,
    ss_token: tl.constexpr,
    so_token,
    so_channel,
    CHANNELS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    WINDOW_ROWS: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    request = tl.load(Batches + tl.program_id(0))
    if request == -1:
        return
    start = tl.load(Boundaries + request).to(tl.int64)
    length = tl.load(Boundaries + request + 1).to(tl.int64) - start
    token = tl.load(Offsets + tl.program_id(0)).to(tl.int64) * BLOCK_M
    if token >= length:
        return
    slot = tl.load(Slots + request).to(tl.int64)
    if slot == -1:
        return
    history = tl.load(History + request)
    channels = (tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    valid_channels = channels < CHANNELS
    positions = tl.arange(0, BLOCK_M)
    valid_outputs = token + positions < length

    if WINDOW:
        # Padding to a power of two is an IR shape, not an extra global read.
        window_positions = tl.arange(0, WINDOW_ROWS)
        window = _load_window_values(
            X,
            State,
            start,
            token + window_positions - (WIDTH - 1),
            channels,
            slot,
            length,
            history,
            sx_token,
            sx_channel,
            ss_slot,
            ss_channel,
            ss_token,
            window_positions < BLOCK_M + WIDTH - 1,
            valid_channels,
            WIDTH,
        )
    acc = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    if HAS_BIAS:
        bias = tl.load(Bias + channels, valid_channels, other=0).to(tl.float32)
        acc = tl.broadcast_to(bias[None, :], (BLOCK_M, BLOCK_N))
    for tap in tl.static_range(WIDTH):
        if WINDOW:
            indices = tl.broadcast_to((positions + tap)[:, None], (BLOCK_M, BLOCK_N))
            values = tl.gather(window, indices, axis=0)
        else:
            values = _load_window_values(
                X,
                State,
                start,
                token + positions + tap - (WIDTH - 1),
                channels,
                slot,
                length,
                history,
                sx_token,
                sx_channel,
                ss_slot,
                ss_channel,
                ss_token,
                valid_outputs,
                valid_channels,
                WIDTH,
            )
        weights = tl.load(
            Weight + channels * sw_channel + tap * sw_tap, valid_channels, other=0
        )
        # Match the old kernel's product type and ordered accumulation. In
        # particular, do not promote BF16 operands before the multiplication.
        if (
            HAS_BIAS
            and SILU
            and WIDTH == 3
            and X.dtype.element_ty == tl.float32
            and Weight.dtype.element_ty == tl.float32
            and tap == 0
        ):
            # The frozen baseline emits a separate first multiply for this
            # branch on SM103. Preserve its rounding before the bias add.
            acc += tl.extra.cuda.libdevice.mul_rn(values, weights[None, :])
        else:
            acc += values * weights[None, :]
    if SILU:
        acc = acc / (1 + tl.exp(-acc))
    output_rows = start + token + positions[:, None]
    output_mask = valid_outputs[:, None] & valid_channels[None, :]
    if so_channel == 1:
        tl.store(Out + output_rows * so_token + channels[None, :], acc, output_mask)
    elif so_token == 1:
        tl.store(Out + output_rows + channels[None, :] * so_channel, acc, output_mask)
    else:
        tl.store(
            Out + output_rows * so_token + channels[None, :] * so_channel,
            acc,
            output_mask,
        )

    if token == 0:
        # Only this tile reads old state. Other tiles read immutable input.
        # A short request takes its final state from old history + new input.
        state_positions = tl.arange(0, triton.next_power_of_2(WIDTH - 1))
        tail = _load_window_values(
            X,
            State,
            start,
            length - (WIDTH - 1) + state_positions,
            channels,
            slot,
            length,
            history,
            sx_token,
            sx_channel,
            ss_slot,
            ss_channel,
            ss_token,
            state_positions < WIDTH - 1,
            valid_channels,
            WIDTH,
        )
        # Finish every old-state load before any thread overwrites that state.
        tl.debug_barrier()
        tl.store(
            State
            + slot * ss_slot
            + channels[None, :] * ss_channel
            + state_positions[:, None] * ss_token,
            tail,
            (state_positions[:, None] < WIDTH - 1) & valid_channels[None, :],
        )


def launch_prefill_candidate(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state: torch.Tensor,
    slots: torch.Tensor,
    history: torch.Tensor,
    boundaries: torch.Tensor,
    out: torch.Tensor,
    *,
    metadata: CausalConv1dPrefillMetadata,
    config: PrefillCandidateConfig | PrefillTacticPolicy,
    silu: bool,
    enable_pdl: bool,
):
    """Launch a private experiment into caller-owned output and state buffers.

    x/out are [channels, packed_tokens], weight is [channels, width], and
    state is [slots, channels, history]. Boundaries partition the packed input.
    Slot -1 and inactive metadata entries leave output and state untouched.
    Each live request must own a distinct state slot. State must not alias x.
    Output must not alias either input or state.
    The metadata block size must match config. An explicit PrefillTacticPolicy
    selects from two fixed configurations before launching exactly one kernel.
    Graph capture pins this selection; graph replay does not run Python again.
    Returns the compiled kernel for inspection, or None for an empty schedule.
    No registry is modified.
    """
    if metadata.block_m != config.block_m:
        raise ValueError("metadata block_m must match the candidate")
    if x.ndim != 2 or weight.ndim != 2 or state.ndim != 3:
        raise ValueError("expected rank-2 input/weight and rank-3 state")
    channels, tokens = x.shape
    width = weight.shape[1]
    if width not in (2, 3, 4) or weight.shape[0] != channels:
        raise ValueError("expected width 2, 3, or 4 and matching channels")
    if out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError("output must match input shape and dtype")
    if state.shape[1] != channels or state.shape[2] < width - 1:
        raise ValueError("state must hold width-1 values per channel")
    if state.dtype != x.dtype:
        raise ValueError("state and input dtypes must match")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("input must be BF16, FP16, or FP32")
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("weight must be BF16, FP16, or FP32")
    if bias is not None and (bias.shape != (channels,) or bias.stride(0) != 1):
        raise ValueError("bias must be a contiguous channel vector")
    if boundaries.numel() != slots.numel() + 1 or history.numel() != slots.numel():
        raise ValueError(
            "boundaries, slots, and history must describe the same requests"
        )
    if slots.dtype not in (torch.int32, torch.int64) or history.dtype != torch.bool:
        raise ValueError("slots must be integer and history must be boolean")
    if boundaries.dtype not in (torch.int32, torch.int64):
        raise ValueError("boundaries must be integer")
    for tensor in (
        slots,
        history,
        boundaries,
        metadata.batch_indices,
        metadata.chunk_offsets,
    ):
        if tensor.ndim != 1 or not tensor.is_contiguous():
            raise ValueError("request vectors and metadata must be contiguous vectors")
    if metadata.batch_indices.numel() != metadata.chunk_offsets.numel():
        raise ValueError("metadata maps must have equal lengths")
    for tensor in (
        weight,
        state,
        slots,
        history,
        boundaries,
        out,
        metadata.batch_indices,
        metadata.chunk_offsets,
    ):
        if tensor.device != x.device:
            raise ValueError("all tensors must be on the input device")
    if bias is not None and bias.device != x.device:
        raise ValueError("bias must be on the input device")
    if metadata.batch_indices.numel() == 0:
        return None
    if isinstance(config, PrefillTacticPolicy):
        config = config.select(
            tokens,
            slots.numel(),
            exact_workload=metadata.cu_seqlens_cpu is not None,
        )
    grid = (metadata.batch_indices.numel(), triton.cdiv(channels, config.block_n))
    options = dict(num_warps=config.num_warps, num_stages=2)
    if enable_pdl:
        options["launch_pdl"] = True
    if config.implementation == "serial":
        return _causal_conv1d_fwd_kernel[grid](
            x,
            weight,
            bias,
            state,
            slots,
            history,
            boundaries,
            metadata.batch_indices,
            metadata.chunk_offsets,
            out,
            channels,
            tokens,
            state.shape[0],
            0,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            0,
            out.stride(0),
            out.stride(1),
            -1,
            HAS_BIAS=bias is not None,
            KERNEL_WIDTH=width,
            SILU_ACTIVATION=silu,
            HAS_INITIAL_STATES=True,
            HAS_CACHE=True,
            IS_CONTINUOUS_BATCHING=True,
            USE_PAD_SLOT=True,
            NP2_STATELEN=triton.next_power_of_2(width - 1),
            BLOCK_M=config.block_m,
            BLOCK_N=config.block_n,
            ENABLE_PDL=enable_pdl,
            **options,
        )
    if config.implementation in (
        "gluon_serial",
        "gluon_direct",
        "gluon_window",
        "gluon_window_r64",
        "gluon_window_r48",
    ):
        if config.implementation == "gluon_window_r64":
            options["maxnreg"] = 64
        elif config.implementation == "gluon_window_r48":
            options["maxnreg"] = 48
        launch_grid = (
            grid if config.implementation == "gluon_serial" else (grid[1], grid[0])
        )
        return gluon_causal_conv1d_prefill_kernel[launch_grid](
            x,
            weight,
            bias,
            state,
            slots,
            history,
            boundaries,
            metadata.batch_indices,
            metadata.chunk_offsets,
            out,
            x.stride(1),
            x.stride(0),
            weight.stride(0),
            weight.stride(1),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            out.stride(1),
            out.stride(0),
            CHANNELS=channels,
            WIDTH=width,
            HAS_BIAS=bias is not None,
            SILU=silu,
            WINDOW=config.implementation not in ("gluon_serial", "gluon_direct"),
            TOKEN_PARALLEL=config.implementation != "gluon_serial",
            BLOCK_M=config.block_m,
            BLOCK_N=config.block_n,
            WINDOW_ROWS=triton.next_power_of_2(config.block_m + width - 1),
            NUM_WARPS=config.num_warps,
            ENABLE_PDL=enable_pdl,
            **options,
        )
    return _causal_conv1d_prefill_candidate_kernel[grid](
        x,
        weight,
        bias,
        state,
        slots,
        history,
        boundaries,
        metadata.batch_indices,
        metadata.chunk_offsets,
        out,
        x.stride(1),
        x.stride(0),
        weight.stride(0),
        weight.stride(1),
        state.stride(0),
        state.stride(1),
        state.stride(2),
        out.stride(1),
        out.stride(0),
        CHANNELS=channels,
        WIDTH=width,
        HAS_BIAS=bias is not None,
        SILU=silu,
        WINDOW=config.implementation == "window",
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        WINDOW_ROWS=triton.next_power_of_2(config.block_m + width - 1),
        ENABLE_PDL=enable_pdl,
        **options,
    )
