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


"""Fused independent conv/verify-gate producers, retaining replay arithmetic."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.kda._triton.buffered_conv import (
    _buffered_conv,
    _validate_conv_inputs,
)
from tokenspeed_kernel.ops.attention.kda._triton.buffered_gate import (
    buffered_history_gate,
)
from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
    _gate_tiling_dot,
)


@triton.jit
def _dual_gate(
    FA,
    FB,
    A,
    BIAS,
    VERIFY,
    HISTORY,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
    FA_STRIDE: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    BOUND: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BF: tl.constexpr,
    LINEAR_BASE: tl.constexpr,
):
    linear = tl.program_id(0) - LINEAR_BASE
    key_blocks: tl.constexpr = triton.cdiv(DIM, BK)
    token_blocks: tl.constexpr = triton.cdiv(ROWS, BT)
    head = linear // (key_blocks * token_blocks)
    row = ((linear // key_blocks) % token_blocks) * BT + tl.arange(0, BT)
    key = (linear % key_blocks) * BK + tl.arange(0, BK)
    feature = tl.arange(0, BF)
    channel = head * DIM + key
    fb = tl.load(
        FB + channel[:, None] * RANK + feature[None, :],
        (key[:, None] < DIM) & (feature[None, :] < RANK),
        0,
    )
    fa = tl.load(
        FA + row[:, None] * FA_STRIDE + feature[None, :],
        (row[:, None] < ROWS) & (feature[None, :] < RANK),
        0,
    )
    if HISTORY is not None:
        bias = tl.load(BIAS + channel, key < DIM, 0).to(tl.float32)
        a = tl.load(A + head).to(tl.float32)
    raw = tl.dot(fa, tl.trans(fb))
    mask = (row[:, None] < ROWS) & (key[None, :] < DIM)
    offset = row[:, None] * GATE_STRIDE + channel[None, :]
    tl.store(VERIFY + offset, raw.to(tl.bfloat16), mask)
    if HISTORY is not None:
        # Accepted replay seeds the MMA accumulator with bias. Adding bias
        # to the rounded raw verify dot changes FP32 history, so keep both
        # accumulators while sharing operands and the launch.
        gate = tl.dot(fa, tl.trans(fb), tl.broadcast_to(bias[None, :], (BT, BK)))
        if BOUND is not None:
            gate = BOUND * tl.sigmoid(tl.exp(a) * gate)
        else:
            gate = -tl.exp(a) * tl.where(gate < 20.0, tl.log(1 + tl.exp(gate)), gate)
        tl.store(HISTORY + offset, gate, mask)


@triton.jit
def _buffered_producers(
    RAW,
    WEIGHT,
    STATE,
    READ,
    WIDTH,
    OK,
    CONV,
    PAYLOAD,
    HISTORY_CONV,
    FA,
    FB,
    A,
    BIAS,
    VERIFY_GATE,
    HISTORY_GATE,
    RS: tl.constexpr,
    CS: tl.constexpr,
    PS: tl.constexpr,
    SS: tl.constexpr,
    WS: tl.constexpr,
    HS: tl.constexpr,
    FA_STRIDE: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    WINDOW: tl.constexpr,
    CHANNELS: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    RANK: tl.constexpr,
    BOUND: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    CONV_BLOCK: tl.constexpr,
):
    conv_programs: tl.constexpr = ROWS * triton.cdiv(CHANNELS, CONV_BLOCK)
    gate_programs: tl.constexpr = HEADS * triton.cdiv(ROWS, BT) * triton.cdiv(DIM, BK)
    pid = tl.program_id(0)
    # Producers are independent. Disjoint CTA ranges share one launch without
    # a grid barrier or cross-CTA communication; recurrence waits for the launch.
    # Start the heavier gate CTAs first, reducing the tail of the shared launch.
    if pid >= gate_programs:
        _buffered_conv(
            RAW,
            WEIGHT,
            STATE,
            READ,
            WIDTH,
            OK,
            CONV,
            PAYLOAD,
            HISTORY_CONV,
            HS,
            RS,
            CS,
            PS,
            SS,
            WS,
            WINDOW,
            CHANNELS,
            CONV_BLOCK,
            gate_programs,
        )
    else:
        if ROWS >= 16:
            history = HISTORY_GATE
        else:
            history = None
        _dual_gate(
            FA,
            FB,
            A,
            BIAS,
            VERIFY_GATE,
            history,
            ROWS,
            HEADS,
            DIM,
            RANK,
            FA_STRIDE,
            GATE_STRIDE,
            BOUND,
            BT,
            BK,
            triton.next_power_of_2(max(RANK, 32)),
            0,
        )


def buffered_producers(
    raw,
    weight,
    state,
    read,
    width,
    ok,
    conv_out,
    payload,
    f_a,
    f_b,
    A_log,
    dt_bias,
    verify_gate,
    *,
    history_conv,
    history_gate,
    num_heads,
    head_dim,
    local_layers,
    lower_bound,
):
    """Write conv and gate scratch together, without allocating storage.

    Conv arguments follow buffered_conv: BF16 raw/payload [B,T,C], BF16
    weight [C,4] and state [pages,C,3], metadata [B], BF16/FP32 conv_out.
    C is 3*heads*head_dim. f_a is BF16 [B*T,rank], f_b BF16 [heads*head_dim,rank],
    A_log FP32 [heads] and dt_bias FP32 [heads*head_dim]. verify_gate is
    contiguous BF16/FP32 [B*T,heads*head_dim]; consumers round it to BF16.
    history_conv/history_gate are both None or caller-owned FP32
    [B,T,2*C/3] and contiguous [B*T,heads*head_dim] replay-order scratch.
    local_layers retains accepted replay's small-row reduction geometry.

    Returns None. All destinations are preallocated and distinct. Invalid/pad
    rows do not access conv state; gate scratch, like the GEMM it replaces,
    may be written for padding and must only be consumed by valid recurrence.
    """
    batch, window, channels = _validate_conv_inputs(
        raw, weight, state, read, width, ok, conv_out, payload, history_out=history_conv
    )
    rows = batch * window
    if (
        head_dim != 128
        or num_heads < 1
        or channels != 3 * num_heads * head_dim
        or local_layers < 1
        or f_a.ndim != 2
        or f_a.shape[0] != rows
        or f_a.shape[1] < 1
        or f_a.stride(1) != 1
        or f_a.stride(0) < 1
        or f_a.dtype != torch.bfloat16
        or f_b.dtype != torch.bfloat16
        or f_b.shape != (num_heads * head_dim, f_a.shape[1])
        or not f_b.is_contiguous()
        or verify_gate.shape != (rows, num_heads * head_dim)
        or verify_gate.dtype not in (torch.bfloat16, torch.float32)
        or not verify_gate.is_contiguous()
        or A_log.shape != (num_heads,)
        or A_log.dtype != torch.float32
        or not A_log.is_contiguous()
        or dt_bias.shape != (num_heads * head_dim,)
        or dt_bias.dtype != torch.float32
        or not dt_bias.is_contiguous()
        or ((history_conv is None) != (history_gate is None))
    ):
        raise ValueError("invalid buffered producer geometry or dtype")
    if history_gate is not None and (
        history_gate.shape != verify_gate.shape
        or history_gate.dtype != torch.float32
        or not history_gate.is_contiguous()
        or (rows < 16 and f_a.shape[1] & (f_a.shape[1] - 1))
    ):
        raise ValueError("invalid buffered producer history scratch")
    extra = () if history_gate is None else (history_gate,)
    if any(
        not t.is_cuda or t.device != raw.device
        for t in (f_a, f_b, A_log, dt_bias, verify_gate, *extra)
    ):
        raise ValueError("all buffered producers must share one GPU")
    block_t, block_k = _gate_tiling_dot(rows, head_dim)
    conv_programs = rows * triton.cdiv(channels, 256)
    gate_programs = (
        num_heads * triton.cdiv(rows, block_t) * triton.cdiv(head_dim, block_k)
    )
    _buffered_producers[(conv_programs + gate_programs,)](
        raw,
        weight,
        state,
        read,
        width,
        ok,
        conv_out,
        payload,
        history_conv,
        f_a,
        f_b,
        A_log,
        dt_bias,
        verify_gate,
        history_gate,
        raw.stride(),
        conv_out.stride(),
        payload.stride(),
        state.stride(),
        weight.stride(),
        () if history_conv is None else history_conv.stride(),
        f_a.stride(0),
        verify_gate.stride(0),
        rows,
        window,
        channels,
        num_heads,
        head_dim,
        f_a.shape[1],
        lower_bound,
        block_t,
        block_k,
        256,
        num_warps=1,
    )
    if history_gate is not None and rows < 16:
        # Combining the scalar reduction with MMA CTAs changes register layout
        # and FP32 rounding. Retain the accepted-replay kernel for small rows.
        buffered_history_gate(
            f_a,
            f_b,
            A_log,
            dt_bias,
            history_gate,
            num_heads=num_heads,
            head_dim=head_dim,
            local_layers=local_layers,
            lower_bound=lower_bound,
        )
