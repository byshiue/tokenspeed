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


"""Unregistered paged KDA recurrence, consuming LCM-owned field views.

This replaces the dense-ring prototype, not the serving kernels. Position
refresh/commit use buffered_metadata; endpoint publication and runtime dispatch
remain gated. No buffers are allocated here, and T=1/T>1 use the same operation.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _check_recurrent_blocks(
    HISTORY_TABLE,
    STATE_TABLE,
    END,
    CHECKPOINT,
    LENGTH,
    VALID,
    FLUSH,
    OK,
    HISTORY_COLUMNS: tl.constexpr,
    STATE_COLUMNS: tl.constexpr,
    HISTORY_TABLE_STRIDE: tl.constexpr,
    STATE_TABLE_STRIDE: tl.constexpr,
    HISTORY_BLOCKS: tl.constexpr,
    STATE_BLOCKS: tl.constexpr,
    ROWS: tl.constexpr,
    STATE_GRAIN: tl.constexpr,
    T: tl.constexpr,
    L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    width = tl.load(VALID + row)
    if width == 0 or not tl.load(OK + row):
        return
    end = tl.load(END + row).to(tl.int64)
    checkpoint = tl.load(CHECKPOINT + row)
    length = tl.load(LENGTH + row).to(tl.int64)
    flush = tl.load(FLUSH + row)
    ok = (
        (width > 0)
        & (width <= T)
        & (checkpoint >= 0)
        & (end >= checkpoint)
        & (length == end - checkpoint)
        & (length <= L - T)
        & (flush == (length + 2 * T > L))
    )
    # Check the entire read/write range before the first candidate or state
    # store. Separate validation avoids cross-head races on the validity flag.
    first = checkpoint // ROWS
    last = (end + width - 1) // ROWS
    columns = first + tl.arange(0, BLOCK)
    needed = columns <= last
    in_bounds = needed & (columns >= 0) & (columns < HISTORY_COLUMNS)
    blocks = tl.load(HISTORY_TABLE + row * HISTORY_TABLE_STRIDE + columns, in_bounds, 0)
    ok &= (
        tl.sum(
            (needed & ((~in_bounds) | (blocks <= 0) | (blocks >= HISTORY_BLOCKS))).to(
                tl.int32
            ),
            0,
        )
        == 0
    )
    # c=0 is the implicit all-zero recurrent state, not a readable cache block.
    src_column = (checkpoint - 1) // STATE_GRAIN
    src_in_bounds = (checkpoint > 0) & (src_column < STATE_COLUMNS)
    src = tl.load(STATE_TABLE + row * STATE_TABLE_STRIDE + src_column, src_in_bounds, 0)
    ok &= (checkpoint == 0) | (src_in_bounds & (src > 0) & (src < STATE_BLOCKS))
    dst_column = (end - 1) // STATE_GRAIN
    dst_in_bounds = flush & (end > 0) & (dst_column < STATE_COLUMNS)
    dst = tl.load(STATE_TABLE + row * STATE_TABLE_STRIDE + dst_column, dst_in_bounds, 0)
    ok &= (~flush) | (dst_in_bounds & (dst > 0) & (dst < STATE_BLOCKS))
    tl.store(OK + row, ok)


@triton.jit
def _history_offset(TABLE, row, token, table_stride: tl.constexpr, rows: tl.constexpr):
    block = tl.load(TABLE + row * table_stride + token // rows).to(tl.int64)
    return block, token % rows


@triton.jit
def _buffered_recurrent(
    Q,
    K,
    V,
    D,
    BETA,
    STATE,
    HK,
    HU,
    HD,
    HISTORY_TABLE,
    STATE_TABLE,
    END,
    CHECKPOINT,
    LENGTH,
    VALID,
    FLUSH,
    OK,
    OUT,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    T: tl.constexpr,
    ROWS: tl.constexpr,
    STATE_GRAIN: tl.constexpr,
    HISTORY_TABLE_STRIDE: tl.constexpr,
    STATE_TABLE_STRIDE: tl.constexpr,
    STATE_STRIDES: tl.constexpr,
    HK_STRIDES: tl.constexpr,
    HU_STRIDES: tl.constexpr,
    HD_STRIDES: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head, tile = tl.program_id(1), tl.program_id(2)
    head = head.to(tl.int64)
    width = tl.load(VALID + row)
    if width == 0 or not tl.load(OK + row):
        return
    end = tl.load(END + row).to(tl.int64)
    checkpoint = tl.load(CHECKPOINT + row)
    length = tl.load(LENGTH + row)
    flush = tl.load(FLUSH + row)
    kk = tl.arange(0, BK).to(tl.int64)
    vv = (tile * BV + tl.arange(0, BV)).to(tl.int64)
    mask = (vv[:, None] < DV) & (kk[None, :] < DK)
    src_column = (checkpoint - 1) // STATE_GRAIN
    src = tl.load(
        STATE_TABLE + row * STATE_TABLE_STRIDE + src_column, checkpoint > 0, 0
    ).to(tl.int64)
    state_feature = (
        head * STATE_STRIDES[1]
        + vv[:, None] * STATE_STRIDES[2]
        + kk[None, :] * STATE_STRIDES[3]
    )
    state = tl.load(
        STATE + src * STATE_STRIDES[0] + state_feature, mask & (checkpoint > 0), 0
    )
    # Absolute token positions replace ring offsets. LCM owns the sliding
    # residency; no modulo-capacity placement or per-request allocation lives here.
    for offset in range(length):
        block, token_row = _history_offset(
            HISTORY_TABLE, row, checkpoint + offset, HISTORY_TABLE_STRIDE, ROWS
        )
        hk = tl.load(
            HK
            + block * HK_STRIDES[0]
            + token_row * HK_STRIDES[1]
            + head * HK_STRIDES[2]
            + kk * HK_STRIDES[3],
            kk < DK,
            0,
        )
        hd = tl.load(
            HD
            + block * HD_STRIDES[0]
            + token_row * HD_STRIDES[1]
            + head * HD_STRIDES[2]
            + kk * HD_STRIDES[3],
            kk < DK,
            0,
        )
        hu = tl.load(
            HU
            + block * HU_STRIDES[0]
            + token_row * HU_STRIDES[1]
            + head * HU_STRIDES[2]
            + vv * HU_STRIDES[3],
            vv < DV,
            0,
        )
        state = state * hd[None, :] + hu[:, None] * hk[None, :]
    if flush:
        dst = tl.load(
            STATE_TABLE + row * STATE_TABLE_STRIDE + (end - 1) // STATE_GRAIN
        ).to(tl.int64)
        # Only previously accepted history is flushed, never current candidates.
        # The destination must be request-writable, not a published snapshot.
        tl.store(STATE + dst * STATE_STRIDES[0] + state_feature, state, mask)
    for token in range(width):
        q = tl.load(Q + ((row * T + token) * H + head) * DK + kk, kk < DK, 0)
        k = tl.load(K + ((row * T + token) * H + head) * DK + kk, kk < DK, 0)
        v = tl.load(V + ((row * T + token) * H + head) * DV + vv, vv < DV, 0)
        d = tl.load(D + ((row * T + token) * H + head) * DK + kk, kk < DK, 0)
        beta = tl.load(BETA + (row * T + token) * H + head)
        state *= d[None, :]
        correction = beta * (v - tl.sum(state * k[None, :], 1))
        block, token_row = _history_offset(
            HISTORY_TABLE, row, end + token, HISTORY_TABLE_STRIDE, ROWS
        )
        if tile == 0:
            tl.store(
                HK
                + block * HK_STRIDES[0]
                + token_row * HK_STRIDES[1]
                + head * HK_STRIDES[2]
                + kk * HK_STRIDES[3],
                k,
                kk < DK,
            )
            tl.store(
                HD
                + block * HD_STRIDES[0]
                + token_row * HD_STRIDES[1]
                + head * HD_STRIDES[2]
                + kk * HD_STRIDES[3],
                d,
                kk < DK,
            )
        tl.store(
            HU
            + block * HU_STRIDES[0]
            + token_row * HU_STRIDES[1]
            + head * HU_STRIDES[2]
            + vv * HU_STRIDES[3],
            correction,
            vv < DV,
        )
        state += correction[:, None] * k[None, :]
        out = tl.sum(state * q[None, :], 1) * (DK**-0.5)
        tl.store(OUT + ((row * T + token) * H + head) * DV + vv, out, vv < DV)


def buffered_recurrent(
    query,
    key,
    value,
    decay,
    beta,
    state_pool,
    history_key,
    history_correction,
    history_decay,
    history_block_table,
    state_block_table,
    end,
    checkpoint,
    length,
    valid,
    flushed,
    ok,
    out,
    *,
    capacity,
    state_block_tokens,
) -> None:
    """Compute paged candidate history/outputs and optionally flush accepted state.

    Args:
        query/key/decay: Contiguous FP32 [B,T,H,K], with normalized Q/K and
            multiplicative decay. T is the fixed maximum execution width.
        value/out: Contiguous FP32 [B,T,H,V] input and caller-owned output.
        beta: Contiguous FP32 [B,T,H] sigmoid-transformed update weights.
        state_pool: FP32 [blocks,H,V,K] cache-owned recurrent checkpoints.
        history_key/history_correction/history_decay: FP32 [blocks,rows,H,K/V/K]
            LCM field views; all four cache fields accept explicit positive strides.
        history_block_table/state_block_table: Int32 [B,columns] raw absolute
            block tables, with contiguous columns and null ids 0/-1. Resolve
            from current tables each call; active writable blocks must be exclusive.
        end: Int32 [B] accepted endpoints e, excluding the current input.
        checkpoint: Int64 [B] materialized positions c from prepare_positions.
        length: Int32 [B] committed lengths e-c from prepare_positions.
        valid: Int32 [B] valid input widths; zero is idle/padding.
        flushed: Bool [B] capacity-flush decisions from prepare_positions.
        ok: Bool [B] in/out validity. A backing/position failure clears it and
            suppresses all stores for that row. Serving must handle failure.
        capacity: Logical L, matching prepare_positions, with L >= 2*T.
        state_block_tokens: State-group checkpoint granularity, independent of
            history rows, logical capacity, and prefix identity.

    Returns:
        None. No allocation/readback. Only valid outputs/candidates are written;
        full state is stored only for a flush, at the pre-candidate endpoint e.
        Follow with commit_positions after acceptance, before metadata reuse.
        This primitive neither grants publication provenance nor commits conv.
        Its two launches are backing validation then recurrence, in the same
        order for eager and CUDA graphs. They are not registered for serving.
    """
    if query.ndim != 4 or value.ndim != 4 or history_key.ndim != 4:
        raise ValueError("query, value and history must be rank-four tensors")
    batch, width, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    blocks, rows = history_key.shape[:2]
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int) or not 0 < v <= 2**31 - 1
            for v in (capacity, state_block_tokens)
        )
        or min(width, heads, key_dim, value_dim, blocks, rows) < 1
        or capacity < 2 * width
    ):
        raise ValueError(
            "positive geometry and capacity >= two maximum windows required"
        )
    dense = (
        (query, (batch, width, heads, key_dim), torch.float32),
        (key, query.shape, torch.float32),
        (decay, query.shape, torch.float32),
        (value, (batch, width, heads, value_dim), torch.float32),
        (beta, (batch, width, heads), torch.float32),
        (out, value.shape, torch.float32),
        (end, (batch,), torch.int32),
        (checkpoint, (batch,), torch.int64),
        (length, (batch,), torch.int32),
        (valid, (batch,), torch.int32),
        (flushed, (batch,), torch.bool),
        (ok, (batch,), torch.bool),
    )
    for tensor, shape, dtype in dense:
        if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(
                f"expected contiguous {dtype} tensor of shape {tuple(shape)}"
            )
    if state_pool.ndim != 4 or state_pool.shape[0] < 1:
        raise ValueError("state pool must contain a null block and have rank four")
    for tensor, shape in (
        (state_pool, (state_pool.shape[0], heads, value_dim, key_dim)),
        (history_key, (blocks, rows, heads, key_dim)),
        (history_correction, (blocks, rows, heads, value_dim)),
        (history_decay, history_key.shape),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != torch.float32
            or any(s <= 0 for s in tensor.stride())
        ):
            raise ValueError(
                "cache fields must be FP32 with declared shapes and positive strides"
            )
    for table in (history_block_table, state_block_table):
        if (
            table.ndim != 2
            or table.shape[0] != batch
            or table.shape[1] < 1
            or table.dtype != torch.int32
            or table.stride(1) != 1
        ):
            raise ValueError(
                "raw block tables must be int32 batch matrices with contiguous columns"
            )
    tensors = [t for t, _, _ in dense] + [
        state_pool,
        history_key,
        history_correction,
        history_decay,
        history_block_table,
        state_block_table,
    ]
    if any(not t.is_cuda or t.device != query.device for t in tensors):
        raise ValueError("all buffered recurrence tensors must share a GPU")
    if batch == 0:
        return
    _check_recurrent_blocks[(batch,)](
        history_block_table,
        state_block_table,
        end,
        checkpoint,
        length,
        valid,
        flushed,
        ok,
        history_block_table.shape[1],
        state_block_table.shape[1],
        history_block_table.stride(0),
        state_block_table.stride(0),
        blocks,
        state_pool.shape[0],
        rows,
        state_block_tokens,
        width,
        capacity,
        triton.next_power_of_2(triton.cdiv(capacity, rows) + 1),
    )
    _buffered_recurrent[(batch, heads, triton.cdiv(value_dim, 32))](
        query,
        key,
        value,
        decay,
        beta,
        state_pool,
        history_key,
        history_correction,
        history_decay,
        history_block_table,
        state_block_table,
        end,
        checkpoint,
        length,
        valid,
        flushed,
        ok,
        out,
        heads,
        key_dim,
        value_dim,
        width,
        rows,
        state_block_tokens,
        history_block_table.stride(0),
        state_block_table.stride(0),
        state_pool.stride(),
        history_key.stride(),
        history_correction.stride(),
        history_decay.stride(),
        triton.next_power_of_2(key_dim),
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
