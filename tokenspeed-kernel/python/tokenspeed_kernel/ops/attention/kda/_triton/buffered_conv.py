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

"""Four-tap convolution and accepted-window commit for buffered KDA.

Conv state stays at the accepted endpoint even when recurrent state lags.
Only this round's raw inputs need transient staging; accepted windows remain
LCM-owned. These private primitives do not grant checkpoint provenance.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _prepare_conv_blocks(
    TABLE,
    END,
    WIDTH,
    OK,
    READ,
    WRITES,
    COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    BLOCKS: tl.constexpr,
    GRAIN: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    end = tl.load(END + row).to(tl.int64)
    width = tl.load(WIDTH + row)
    old_ok = tl.load(OK + row)
    active = (width > 0) & old_ok
    src_col = (end - 1) // GRAIN
    src_in = active & (end > 0) & (src_col >= 0) & (src_col < COLS)
    src = tl.load(TABLE + row * TABLE_STRIDE + src_col, src_in, 0)
    token = tl.arange(0, BT)
    dst_col = (end + token) // GRAIN
    needed = active & (token < width)
    dst_in = needed & (dst_col >= 0) & (dst_col < COLS)
    dst = tl.load(TABLE + row * TABLE_STRIDE + dst_col, dst_in, 0)
    bad_dst = needed & ((~dst_in) | (dst <= 0) | (dst >= BLOCKS))
    good = (
        old_ok
        & (
            (~active)
            | (
                (end >= 0)
                & (width <= T)
                & ((end == 0) | (src_in & (src > 0) & (src < BLOCKS)))
                & (tl.sum(bad_dst.to(tl.int32), 0) == 0)
            )
        )
        & (width >= 0)
    )
    tl.store(READ + row, tl.where(active & good & (end > 0), src, -1))
    tl.store(WRITES + row * T + token, tl.where(needed & good, dst, -1), token < T)
    tl.store(OK + row, good)


def prepare_conv_blocks(table, end, width, ok, read, writes, *, blocks, grain):
    """Resolve current conv state and destinations for every possible acceptance.

    ``table`` is int32 [B, columns] raw state blocks; ``end``/``width`` int32
    [B] and ``ok`` bool [B] come from replay preparation. ``read`` is int32 [B]
    and ``writes`` int32 [B,T], caller-owned contiguous outputs. State at e uses
    slot (e-1)//grain; accepting a>0 uses slot (e+a-1)//grain. Fresh e=0 reads
    implicit zeros. Check all potential destinations before any layer stores.
    Invalid backing clears ok; it never restores a previously invalid row.
    """
    batch = end.numel()
    if (
        table.ndim != 2
        or table.shape[0] != batch
        or table.shape[1] < 1
        or table.dtype != torch.int32
        or table.stride(1) != 1
        or writes.ndim != 2
        or writes.shape[0] != batch
        or writes.shape[1] < 1
        or writes.dtype != torch.int32
        or not writes.is_contiguous()
        or blocks < 1
        or grain < 1
    ):
        raise ValueError("invalid conv block geometry or tables")
    for tensor, dtype in (
        (end, torch.int32),
        (width, torch.int32),
        (ok, torch.bool),
        (read, torch.int32),
    ):
        if (
            tensor.shape != (batch,)
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError("conv block positions require contiguous batch vectors")
    if any(
        not t.is_cuda or t.device != end.device
        for t in (table, end, width, ok, read, writes)
    ):
        raise ValueError("conv block metadata must share one GPU")
    if batch:
        _prepare_conv_blocks[(batch,)](
            table,
            end,
            width,
            ok,
            read,
            writes,
            table.shape[1],
            table.stride(0),
            blocks,
            grain,
            writes.shape[1],
            triton.next_power_of_2(writes.shape[1]),
            num_warps=1,
        )


@triton.jit
def _buffered_conv(
    RAW,
    WEIGHT,
    STATE,
    READ,
    WIDTH,
    OK,
    OUT,
    PAYLOAD,
    RAW_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    PAYLOAD_STRIDES: tl.constexpr,
    STATE_STRIDES: tl.constexpr,
    WEIGHT_STRIDES: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    packed_row = tl.program_id(0).to(tl.int64)
    row, token = packed_row // T, packed_row % T
    if token >= tl.load(WIDTH + row) or not tl.load(OK + row):
        return
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = channel < C
    page = tl.load(READ + row).to(tl.int64)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(4):
        window = token + tap
        raw_token = window - 3
        raw_value = tl.load(
            RAW
            + row * RAW_STRIDES[0]
            + raw_token * RAW_STRIDES[1]
            + channel * RAW_STRIDES[2],
            mask & (raw_token >= 0),
            0,
        )
        old_value = tl.load(
            STATE
            + page * STATE_STRIDES[0]
            + channel * STATE_STRIDES[1]
            + window * STATE_STRIDES[2],
            mask & (raw_token < 0) & (page >= 0),
            0,
        )
        value = tl.where(raw_token >= 0, raw_value, old_value).to(tl.float32)
        weight = tl.load(
            WEIGHT + channel * WEIGHT_STRIDES[0] + tap * WEIGHT_STRIDES[1], mask, 0
        ).to(tl.float32)
        acc = tl.fma(value, weight, acc)
        if tap == 3:
            # This tap already loaded the current token. Capture it for the
            # later accepted-window commit, without a separate copy launch.
            tl.store(
                PAYLOAD
                + row * PAYLOAD_STRIDES[0]
                + token * PAYLOAD_STRIDES[1]
                + channel * PAYLOAD_STRIDES[2],
                raw_value,
                mask,
            )
    activated = acc * tl.sigmoid(acc)
    tl.store(
        OUT + row * OUT_STRIDES[0] + token * OUT_STRIDES[1] + channel * OUT_STRIDES[2],
        activated,
        mask,
    )


def buffered_conv(raw, weight, state, read, width, ok, out, payload):
    """Compute four-tap conv+SiLU and capture BF16 raw candidates in one launch.

    ``raw/payload`` are positive-stride BF16 [B,T,C] tensors with distinct
    storage roles; ``out`` has the same shape and is BF16 or FP32. FP32 output
    retains the accumulator for accepted history; the recurrence rounds it to
    BF16 for verification without another producer launch. ``weight`` is BF16
    [C,4], ``state`` BF16 [blocks,C,3] oldest-first.
    ``read/width`` are contiguous int32 [B], ``ok`` bool [B]. The caller must
    run prepare_conv_blocks first and keep all metadata unchanged. Padding and
    invalid rows neither read state nor write output/payload. No state changes
    until acceptance; T=1 follows the same path. Returns None, allocating nothing.
    """
    if raw.ndim != 3 or min(raw.shape) < 1:
        raise ValueError("raw conv input must be nonempty [B,T,C]")
    batch, width_max, channels = raw.shape
    for tensor, shape in (
        (raw, raw.shape),
        (payload, raw.shape),
        (weight, (channels, 4)),
        (state, (state.shape[0], channels, 3)),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != torch.bfloat16
            or any(s <= 0 for s in tensor.stride())
        ):
            raise ValueError(
                "conv fields require declared BF16 shapes and positive strides"
            )
    if (
        out.shape != raw.shape
        or out.dtype not in (torch.bfloat16, torch.float32)
        or any(s <= 0 for s in out.stride())
    ):
        raise ValueError(
            "conv output requires matching BF16/FP32 shape and positive strides"
        )
    for tensor, dtype in ((read, torch.int32), (width, torch.int32), (ok, torch.bool)):
        if (
            tensor.shape != (batch,)
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError("conv metadata requires contiguous batch vectors")
    if any(
        not t.is_cuda or t.device != raw.device
        for t in (raw, weight, state, read, width, ok, out, payload)
    ):
        raise ValueError("conv tensors must share one GPU")
    _buffered_conv[(batch * width_max, triton.cdiv(channels, 256))](
        raw,
        weight,
        state,
        read,
        width,
        ok,
        out,
        payload,
        raw.stride(),
        out.stride(),
        payload.stride(),
        state.stride(),
        weight.stride(),
        width_max,
        channels,
        256,
        num_warps=4,
    )


@triton.jit
def _commit_conv_windows(
    PAYLOAD,
    CONV_PTRS,
    GROUPS,
    READ,
    WRITES,
    WIDTH,
    OK,
    ACCEPTED,
    PAYLOAD_STRIDES: tl.constexpr,
    CONV_STRIDES: tl.constexpr,
    READ_STRIDE: tl.constexpr,
    WRITE_STRIDES: tl.constexpr,
    OK_STRIDE: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    layer, row, tile = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    group = tl.load(GROUPS + layer).to(tl.int64)
    count = tl.load(ACCEPTED + row)
    if (
        not tl.load(OK + group * OK_STRIDE + row)
        or count <= 0
        or count > tl.load(WIDTH + row)
    ):
        return
    src = tl.load(READ + group * READ_STRIDE + row).to(tl.int64)
    dst = tl.load(
        WRITES
        + group * WRITE_STRIDES[0]
        + row * WRITE_STRIDES[1]
        + (count - 1) * WRITE_STRIDES[2]
    ).to(tl.int64)
    if dst < 0:
        return
    state = tl.load(CONV_PTRS + layer).to(tl.pointer_type(tl.bfloat16))
    channel = tile * BLOCK + tl.arange(0, BLOCK)
    tap = tl.arange(0, 4)
    mask = (tap[:, None] < 3) & (channel[None, :] < C)
    source_tap = tap + count
    old = tl.load(
        state
        + src * CONV_STRIDES[0]
        + channel[None, :] * CONV_STRIDES[1]
        + source_tap[:, None] * CONV_STRIDES[2],
        mask & (source_tap[:, None] < 3) & (src >= 0),
        0,
    )
    raw = tl.load(
        PAYLOAD
        + layer.to(tl.int64) * PAYLOAD_STRIDES[0]
        + row * PAYLOAD_STRIDES[1]
        + (source_tap[:, None] - 3).to(tl.int64) * PAYLOAD_STRIDES[2]
        + channel[None, :] * PAYLOAD_STRIDES[3],
        mask & (source_tap[:, None] >= 3),
        0,
    )
    # Load the complete shifted window before the in-place store. One program
    # owns a channel range, so no other program races its source reads.
    value = tl.where(source_tap[:, None] < 3, old, raw)
    tl.store(
        state
        + dst * CONV_STRIDES[0]
        + channel[None, :] * CONV_STRIDES[1]
        + tap[:, None] * CONV_STRIDES[2],
        value,
        mask,
    )


def commit_conv_windows(
    payload, conv_ptrs, groups, read, writes, width, ok, accepted, *, conv_strides
):
    """Commit all layers' accepted raw-input suffixes in one GPU launch.

    ``payload`` is BF16 [layers,max_bs,T,C] transient storage. ``conv_ptrs``
    int64 [layers] are stable LCM conv-field pointers with ``conv_strides``;
    ``groups`` int32 [layers] maps them to read/ok [groups,max_bs] and writes
    [groups,max_bs,T]. Only accepted.numel() live rows run. Width/acceptance
    are int32, counts already include the target input. Zero or invalid counts
    skip all stores (the subsequent stamp commit reports invalid acceptance).
    Destinations must be request-writable and all layer forwards must finish
    first. No stamps or recurrent state are changed by this operation.
    """
    if payload.ndim != 4 or min(payload.shape) < 1 or payload.dtype != torch.bfloat16:
        raise ValueError("conv payload requires BF16 [layers,max_bs,T,C]")
    layers, max_bs, window, channels = payload.shape
    batch = accepted.numel()
    if (
        batch > max_bs
        or accepted.shape != (batch,)
        or accepted.dtype != torch.int32
        or not accepted.is_contiguous()
        or width.shape != (max_bs,)
        or width.dtype != torch.int32
        or not width.is_contiguous()
        or read.ndim != 2
        or read.shape[1] != max_bs
        or read.dtype != torch.int32
        or ok.shape != read.shape
        or ok.dtype != torch.bool
        or writes.shape != (*read.shape, window)
        or writes.dtype != torch.int32
        or read.stride(1) != 1
        or ok.stride(1) != 1
        or len(conv_strides) != 3
        or any(s <= 0 for s in conv_strides)
    ):
        raise ValueError("invalid batched conv commit metadata")
    for tensor, dtype in ((conv_ptrs, torch.int64), (groups, torch.int32)):
        if (
            tensor.shape != (layers,)
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError("conv descriptors require contiguous layer vectors")
    if any(
        not t.is_cuda or t.device != payload.device
        for t in (payload, conv_ptrs, groups, read, writes, width, ok, accepted)
    ):
        raise ValueError("conv commit tensors must share one GPU")
    if batch:
        _commit_conv_windows[(layers, batch, triton.cdiv(channels, 128))](
            payload,
            conv_ptrs,
            groups,
            read,
            writes,
            width,
            ok,
            accepted,
            payload.stride(),
            conv_strides,
            read.stride(0),
            writes.stride(),
            ok.stride(0),
            window,
            channels,
            128,
            num_warps=1,
        )
