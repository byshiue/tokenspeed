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


def validate_recurrent_blocks(
    history_block_table,
    state_block_table,
    end,
    checkpoint,
    length,
    valid,
    flushed,
    ok,
    *,
    history_blocks,
    state_blocks,
    history_block_tokens,
    state_block_tokens,
    capacity,
    max_window,
) -> None:
    """Validate a group's entire recurrent read/write range before any layer runs.

    Tables are int32 [B, columns] raw block IDs. Position vectors are the
    outputs of ``prepare_positions``: end/length/valid int32, checkpoint int64,
    flushed/ok bool. All tensors share one GPU. Invalid rows clear ``ok``;
    this never restores a row rejected by position preparation.

    The required geometry describes the group's cache-owned fields and fixed
    capacity/window, not one layer's activations. Layers in the same group
    share these positions, tables and page counts, so one validation launch
    covers them all. Tables/positions must not change between validation and
    the last recurrence consumer. This check does not grant publication or
    request-writable ownership of a checkpoint.
    """
    geometry = (
        history_blocks,
        state_blocks,
        history_block_tokens,
        state_block_tokens,
        capacity,
        max_window,
    )
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in geometry):
        raise ValueError("recurrent block geometry must be positive integers")
    if capacity < 2 * max_window:
        raise ValueError("capacity must cover two maximum windows")
    batch = end.numel()
    for tensor, dtype in (
        (end, torch.int32),
        (checkpoint, torch.int64),
        (length, torch.int32),
        (valid, torch.int32),
        (flushed, torch.bool),
        (ok, torch.bool),
    ):
        if (
            tensor.shape != (batch,)
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "positions require contiguous batch vectors of declared dtype"
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
                "raw block tables require int32 batch rows and contiguous columns"
            )
    tensors = (
        history_block_table,
        state_block_table,
        end,
        checkpoint,
        length,
        valid,
        flushed,
        ok,
    )
    if any(not t.is_cuda or t.device != end.device for t in tensors):
        raise ValueError("recurrent block metadata must share a GPU")
    if batch:
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
            history_blocks,
            state_blocks,
            history_block_tokens,
            state_block_tokens,
            max_window,
            capacity,
            triton.next_power_of_2(triton.cdiv(capacity, history_block_tokens) + 1),
        )


@triton.jit
def _history_offset(TABLE, row, token, table_stride: tl.constexpr, rows: tl.constexpr):
    block = tl.load(TABLE + row * table_stride + token // rows).to(tl.int64)
    return block, token % rows


@triton.jit
def _multiply(left, right):
    return left * right


@triton.jit
def _input_offset(row, token, head, feature, strides: tl.constexpr):
    # Widen before multiplying: a strided producer view can span more than
    # signed-int32 element offsets even though its logical T is small.
    return (
        row * strides[0]
        + token.to(tl.int64) * strides[1]
        + head * strides[2]
        + feature * strides[3]
    )


@triton.jit
def _buffered_recurrent(
    Q,
    K,
    V,
    D,
    BETA,
    A_LOG,
    DT_BIAS,
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
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    D_STRIDES: tl.constexpr,
    BETA_STRIDES: tl.constexpr,
    OUT_STRIDES: tl.constexpr,
    TRANSFORM_INPUTS: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
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
    BH: tl.constexpr,
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
    # Within a history tile, S' = S * prod(D) + U^T @ (K * suffix(D)).
    # Corrections are already known for accepted history, so reconstruct it in
    # parallel; only the new candidates below still depend on one another.
    # Shift D before the reverse product to obtain an exclusive suffix without
    # division (decay may be zero). Invalid rows are the affine identity.
    hh = tl.arange(0, BH).to(tl.int64)
    for offset in range(tl.cdiv(length, BH)):
        history_index = offset * BH + hh
        position = checkpoint + history_index
        live = history_index < length
        block = tl.load(
            HISTORY_TABLE + row * HISTORY_TABLE_STRIDE + position // ROWS, live, 0
        ).to(tl.int64)
        token_row = position % ROWS
        hk = tl.load(
            HK
            + block[:, None] * HK_STRIDES[0]
            + token_row[:, None] * HK_STRIDES[1]
            + head * HK_STRIDES[2]
            + kk[None, :] * HK_STRIDES[3],
            live[:, None] & (kk[None, :] < DK),
            0,
        )
        next_live = live & (hh + 1 < BH) & (history_index + 1 < length)
        next_block = tl.load(
            HISTORY_TABLE + row * HISTORY_TABLE_STRIDE + (position + 1) // ROWS,
            next_live,
            0,
        ).to(tl.int64)
        next_decay = tl.load(
            HD
            + next_block[:, None] * HD_STRIDES[0]
            + ((position + 1) % ROWS)[:, None] * HD_STRIDES[1]
            + head * HD_STRIDES[2]
            + kk[None, :] * HD_STRIDES[3],
            next_live[:, None] & (kk[None, :] < DK),
            1,
        )
        suffix = tl.associative_scan(next_decay, 0, _multiply, reverse=True)
        first_decay = tl.load(
            HD
            + block[:, None] * HD_STRIDES[0]
            + token_row[:, None] * HD_STRIDES[1]
            + head * HD_STRIDES[2]
            + kk[None, :] * HD_STRIDES[3],
            (hh[:, None] == 0) & live[:, None] & (kk[None, :] < DK),
            1,
        )
        product = tl.sum(tl.where(hh[:, None] == 0, suffix * first_decay, 0), 0)
        hu = tl.load(
            HU
            + block[None, :] * HU_STRIDES[0]
            + token_row[None, :] * HU_STRIDES[1]
            + head * HU_STRIDES[2]
            + vv[:, None] * HU_STRIDES[3],
            (vv[:, None] < DV) & live[None, :],
            0,
        )
        # FP32 reductions keep the recurrent tile in its ordinary layout. A
        # tensor-core dot needs extra layout conversions and register storage
        # at this small history width; no history quantization is required here.
        state = state * product[None, :] + tl.sum(
            hu[:, :, None] * (hk * suffix)[None, :, :], 1
        )
    if flush:
        dst = tl.load(
            STATE_TABLE + row * STATE_TABLE_STRIDE + (end - 1) // STATE_GRAIN
        ).to(tl.int64)
        # Only previously accepted history is flushed, never current candidates.
        # The destination must be request-writable, not a published snapshot.
        tl.store(STATE + dst * STATE_STRIDES[0] + state_feature, state, mask)
    if TRANSFORM_INPUTS:
        a_scale = tl.exp(tl.load(A_LOG + head).to(tl.float32))
        bias = tl.load(DT_BIAS + head * DK + kk, kk < DK, 0).to(tl.float32)
    for token in range(width):
        q = tl.load(Q + _input_offset(row, token, head, kk, Q_STRIDES), kk < DK, 0).to(
            tl.float32
        )
        k = tl.load(K + _input_offset(row, token, head, kk, K_STRIDES), kk < DK, 0).to(
            tl.float32
        )
        v = tl.load(V + _input_offset(row, token, head, vv, V_STRIDES), vv < DV, 0).to(
            tl.float32
        )
        d = tl.load(D + _input_offset(row, token, head, kk, D_STRIDES), kk < DK, 0).to(
            tl.float32
        )
        beta = tl.load(
            BETA
            + row * BETA_STRIDES[0]
            + token.to(tl.int64) * BETA_STRIDES[1]
            + head * BETA_STRIDES[2]
        ).to(tl.float32)
        if TRANSFORM_INPUTS:
            # Consume the serving split producers directly: BF16 conv(+SiLU)
            # Q/K/V and raw f_b gate. Keep normalized K, correction and decay
            # in FP32; no per-layer prepared-input tensors or launches are needed.
            q /= tl.sqrt(tl.sum(q * q, 0) + 1e-6)
            k /= tl.sqrt(tl.sum(k * k, 0) + 1e-6)
            raw_gate = d + bias
            if LOWER_BOUND is not None:
                log_decay = LOWER_BOUND * tl.sigmoid(a_scale * raw_gate)
            else:
                log_decay = -a_scale * tl.where(
                    raw_gate < 20.0, tl.log(1 + tl.exp(raw_gate)), raw_gate
                )
            d = tl.exp(log_decay)
            beta = tl.sigmoid(beta)
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
        tl.store(OUT + _input_offset(row, token, head, vv, OUT_STRIDES), out, vv < DV)


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
    transform_inputs,
    A_log,
    dt_bias,
    lower_bound,
) -> None:
    """Compute paged candidates using already-validated per-group positions.

    ``prepare_positions`` and ``validate_recurrent_blocks`` must precede every
    round, once per group rather than once per layer. All participating layers
    must share the validated geometry and unchanged tables/positions. This
    function issues only the recurrence launch; invalid rows skip all stores.

    Args:
        query/key/decay: [B,T,H,K] inputs; T is the fixed maximum execution
            width. Positive strides allow zero-copy Q/K views of packed conv
            output. Prepared inputs are FP32 normalized Q/K and decay factors.
        value/out: [B,T,H,V] input and caller-owned output with positive strides.
        beta: [B,T,H] update weights (already sigmoid in the prepared case).
        transform_inputs: True consumes BF16 conv(+SiLU) Q/K/V and BF16/FP32
            raw f_b gate and beta logits. Apply Q/K normalization, gate/decay
            and beta transforms inside the recurrence. False consumes prepared
            FP32 inputs for the reference/benchmark contract. This is an input
            representation, not a standard/speculative execution mode.
        A_log/dt_bias: Contiguous FP32 [H]/[H*K] gate parameters when transforming;
            otherwise both must be None. No transformed-input workspace is built.
        lower_bound: Optional nonpositive log-decay bound for transformed inputs;
            None selects the softplus gate, and is required for prepared inputs.
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
    if not isinstance(transform_inputs, bool):
        raise ValueError(
            "transform_inputs must explicitly select the input representation"
        )
    input_dtype = torch.bfloat16 if transform_inputs else torch.float32
    data = (
        (query, (batch, width, heads, key_dim), (input_dtype,)),
        (key, query.shape, (input_dtype,)),
        (value, (batch, width, heads, value_dim), (input_dtype,)),
        (
            decay,
            query.shape,
            (torch.bfloat16, torch.float32) if transform_inputs else (torch.float32,),
        ),
        (
            beta,
            (batch, width, heads),
            (torch.bfloat16, torch.float32) if transform_inputs else (torch.float32,),
        ),
        (out, value.shape, (input_dtype,)),
    )
    for tensor, shape, dtypes in data:
        if (
            tensor.shape != shape
            or tensor.dtype not in dtypes
            or any(s <= 0 for s in tensor.stride())
        ):
            raise ValueError(
                "input/output fields require declared shapes/dtypes and positive strides"
            )
    parameters = []
    if transform_inputs:
        for tensor, shape in ((A_log, (heads,)), (dt_bias, (heads * key_dim,))):
            if (
                tensor is None
                or tensor.shape != shape
                or tensor.dtype != torch.float32
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "transformed inputs require contiguous FP32 A_log and dt_bias"
                )
            parameters.append(tensor)
        if lower_bound is not None and (
            not isinstance(lower_bound, (int, float))
            or isinstance(lower_bound, bool)
            or not -float("inf") < lower_bound <= 0
        ):
            raise ValueError("lower_bound must be finite and nonpositive")
    elif A_log is not None or dt_bias is not None or lower_bound is not None:
        raise ValueError("prepared inputs must not supply gate parameters")
    dense = (
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
    tensors = (
        [t for t, _, _ in dense]
        + [t for t, _, _ in data]
        + parameters
        + [
            state_pool,
            history_key,
            history_correction,
            history_decay,
            history_block_table,
            state_block_table,
        ]
    )
    if any(not t.is_cuda or t.device != query.device for t in tensors):
        raise ValueError("all buffered recurrence tensors must share a GPU")
    if batch == 0:
        return
    # The measured minimum-capacity native-input cases favor more, smaller
    # programs. Longer histories retain the wider FP32 reconstruction tile.
    narrow_tile = (
        transform_inputs
        and capacity == 2 * width
        and width in (1, 4)
        and key_dim == value_dim == 128
    )
    value_tile = 8 if narrow_tile else 32
    history_tile = min(
        4 if narrow_tile else 8, triton.next_power_of_2(capacity - width)
    )
    _buffered_recurrent[(batch, heads, triton.cdiv(value_dim, value_tile))](
        query,
        key,
        value,
        decay,
        beta,
        A_log,
        dt_bias,
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
        query.stride(),
        key.stride(),
        value.stride(),
        decay.stride(),
        beta.stride(),
        out.stride(),
        transform_inputs,
        lower_bound,
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
        value_tile,
        history_tile,
        num_warps=1 if narrow_tile else 4,
        num_stages=1,
        enable_fp_fusion=False,
    )
