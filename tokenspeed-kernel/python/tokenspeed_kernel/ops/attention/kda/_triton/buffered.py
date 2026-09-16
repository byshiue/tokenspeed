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


"""Paged KDA recurrence, consuming LCM-owned field views.

Position refresh/commit use buffered_metadata. Cache ownership and endpoint
publication belong to the caller. No buffers are allocated here, and T=1/T>1
use the same operation.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import gl, gluon, tl, triton
from tokenspeed_kernel.ops.attention.kda._triton.verify_math import (
    verify_normalize,
    verify_recurrence,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


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
    HANDOFF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    width = tl.load(VALID + row)
    if (width == 0 and not HANDOFF) or not tl.load(OK + row):
        return
    end = tl.load(END + row).to(tl.int64)
    checkpoint = tl.load(CHECKPOINT + row)
    length = tl.load(LENGTH + row).to(tl.int64)
    flush = tl.load(FLUSH + row)
    ok = (
        (width == 0 if HANDOFF else width > 0)
        & (width <= T)
        & (checkpoint >= 0)
        & (end >= checkpoint)
        & (length == end - checkpoint)
        & (length <= L - T)
        & (flush == (False if HANDOFF else length + 2 * T > L))
    )
    # Check the entire read/write range before the first candidate or state
    # store. Separate validation avoids cross-head races on the validity flag.
    first = checkpoint // ROWS
    last = (end + width - 1) // ROWS
    columns = first + tl.arange(0, BLOCK)
    needed = columns <= last
    if HANDOFF:
        needed &= length > 0
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
    needs_dst = (end > 0) if HANDOFF else flush
    dst_in_bounds = needs_dst & (end > 0) & (dst_column < STATE_COLUMNS)
    dst = tl.load(STATE_TABLE + row * STATE_TABLE_STRIDE + dst_column, dst_in_bounds, 0)
    ok &= (~needs_dst) | (dst_in_bounds & (dst > 0) & (dst < STATE_BLOCKS))
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
    for_handoff,
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
    for_handoff validates zero-width live rows and only [c,e), plus the exact
    destination S_e. It requires no candidate backing and no completed flush.
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
            for_handoff,
            triton.next_power_of_2(triton.cdiv(capacity, history_block_tokens) + 1),
        )


@gluon.jit
def _history_offset(TABLE, row, token, table_stride: gl.constexpr, rows: gl.constexpr):
    block = gl.load(TABLE + row * table_stride + token // rows).to(gl.int64)
    return block, token % rows


@gluon.jit
def _multiply(left, right):
    return left * right


@gluon.jit
def _sigmoid(value):
    # Same expression as tl.sigmoid; Gluon does not export that wrapper.
    return 1 / (1 + gl.exp(-value))


@gluon.jit
def _input_offset(row, token, head, feature, strides: gl.constexpr):
    # Widen before multiplying: a strided producer view can span more than
    # signed-int32 element offsets even though its logical T is small.
    return (
        row * strides[0]
        + token.to(gl.int64) * strides[1]
        + head * strides[2]
        + feature * strides[3]
    )


@gluon.jit
def _reconstruct_history(
    state,
    HK,
    HU,
    HD,
    HISTORY_TABLE,
    row,
    head,
    checkpoint,
    length,
    kk,
    vv,
    HISTORY_TABLE_STRIDE: gl.constexpr,
    ROWS: gl.constexpr,
    HK_STRIDES: gl.constexpr,
    HU_STRIDES: gl.constexpr,
    HD_STRIDES: gl.constexpr,
    DK: gl.constexpr,
    DV: gl.constexpr,
    BH: gl.constexpr,
    STATE_HISTORY_LAYOUT: gl.constexpr,
):
    history_layout: gl.constexpr = gl.SliceLayout(0, STATE_HISTORY_LAYOUT)
    correction_layout: gl.constexpr = gl.SliceLayout(2, STATE_HISTORY_LAYOUT)
    kk_h = gl.convert_layout(kk, gl.SliceLayout(0, history_layout))
    vv_u = gl.convert_layout(vv, gl.SliceLayout(1, correction_layout))
    hh = gl.arange(0, BH, layout=gl.SliceLayout(1, history_layout)).to(gl.int64)
    for offset in range(gl.cdiv(length, BH)):
        index = offset * BH + hh
        position = checkpoint + index
        live = index < length
        block = gl.load(
            HISTORY_TABLE + row * HISTORY_TABLE_STRIDE + position // ROWS, live, 0
        ).to(gl.int64)
        token = position % ROWS
        hk = gl.load(
            HK
            + block[:, None] * HK_STRIDES[0]
            + token[:, None] * HK_STRIDES[1]
            + head * HK_STRIDES[2]
            + kk_h[None, :] * HK_STRIDES[3],
            live[:, None] & (kk_h[None, :] < DK),
            0,
        )
        hd = gl.load(
            HD
            + block[:, None] * HD_STRIDES[0]
            + token[:, None] * HD_STRIDES[1]
            + head * HD_STRIDES[2]
            + kk_h[None, :] * HD_STRIDES[3],
            live[:, None] & (kk_h[None, :] < DK),
            1,
        )
        block_u = gl.convert_layout(block, gl.SliceLayout(0, correction_layout))
        token_u = gl.convert_layout(token, gl.SliceLayout(0, correction_layout))
        live_u = gl.convert_layout(live, gl.SliceLayout(0, correction_layout))
        hu = gl.load(
            HU
            + block_u[None, :] * HU_STRIDES[0]
            + token_u[None, :] * HU_STRIDES[1]
            + head * HU_STRIDES[2]
            + vv_u[:, None] * HU_STRIDES[3],
            (vv_u[:, None] < DV) & live_u[None, :],
            0,
        )
        for token_index in gl.static_range(BH):
            if offset * BH + token_index < length:
                # Selecting a length-one axis adds no reduction arithmetic.
                hk_one = gl.gather(
                    hk,
                    gl.full((1, kk.shape[0]), token_index, gl.int32, history_layout),
                    0,
                )
                hd_one = gl.gather(
                    hd,
                    gl.full((1, kk.shape[0]), token_index, gl.int32, history_layout),
                    0,
                )
                hu_one = gl.gather(
                    hu,
                    gl.full((vv.shape[0], 1), token_index, gl.int32, correction_layout),
                    1,
                )
                key = gl.convert_layout(gl.sum(hk_one, 0), kk.type.layout)
                decay = gl.convert_layout(gl.sum(hd_one, 0), kk.type.layout)
                correction = gl.convert_layout(gl.sum(hu_one, 1), vv.type.layout)
                state = state * decay[None, :]
                state = gl.fma(correction[:, None], key[None, :], state)
    return state


@gluon.jit
def _buffered_recurrent(
    Q,
    K,
    V,
    D,
    BETA,
    HISTORY_K_INPUT,
    HISTORY_V_ACC,
    HISTORY_LOG_INPUT,
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
    Q_STRIDES: gl.constexpr,
    K_STRIDES: gl.constexpr,
    V_STRIDES: gl.constexpr,
    D_STRIDES: gl.constexpr,
    BETA_STRIDES: gl.constexpr,
    OUT_STRIDES: gl.constexpr,
    HISTORY_INPUT_STRIDES: gl.constexpr,
    HAS_HISTORY_INPUTS: gl.constexpr,
    TRANSFORM_INPUTS: gl.constexpr,
    FP32_PRODUCERS: gl.constexpr,
    LOWER_BOUND: gl.constexpr,
    H: gl.constexpr,
    DK: gl.constexpr,
    DV: gl.constexpr,
    T: gl.constexpr,
    ROWS: gl.constexpr,
    STATE_GRAIN: gl.constexpr,
    HISTORY_TABLE_STRIDE: gl.constexpr,
    STATE_TABLE_STRIDE: gl.constexpr,
    STATE_STRIDES: gl.constexpr,
    HK_STRIDES: gl.constexpr,
    HU_STRIDES: gl.constexpr,
    HD_STRIDES: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    BH: gl.constexpr,
):
    row = gl.program_id(0).to(gl.int64)
    head, tile = gl.program_id(1), gl.program_id(2)
    head = head.to(gl.int64)
    width = gl.load(VALID + row)
    if width == 0 or not gl.load(OK + row):
        return
    end = gl.load(END + row).to(gl.int64)
    checkpoint = gl.load(CHECKPOINT + row)
    length = gl.load(LENGTH + row)
    flush = gl.load(FLUSH + row)
    state_history_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [1, 1, 32], [gl.num_warps(), 1, 1], [2, 0, 1]
    )
    state_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [1, 32], [gl.num_warps(), 1], [1, 0]
    )
    kk = gl.arange(0, BK, layout=gl.SliceLayout(0, state_layout)).to(gl.int64)
    vv = (tile * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, state_layout))).to(
        gl.int64
    )
    mask = (vv[:, None] < DV) & (kk[None, :] < DK)
    src_column = (checkpoint - 1) // STATE_GRAIN
    src = gl.load(
        STATE_TABLE + row * STATE_TABLE_STRIDE + src_column, checkpoint > 0, 0
    ).to(gl.int64)
    state_feature = (
        head * STATE_STRIDES[1]
        + vv[:, None] * STATE_STRIDES[2]
        + kk[None, :] * STATE_STRIDES[3]
    )
    state = gl.load(
        STATE + src * STATE_STRIDES[0] + state_feature, mask & (checkpoint > 0), 0
    )
    # Absolute positions use LCM's sliding residency, not a private dense ring.
    state = _reconstruct_history(
        state,
        HK,
        HU,
        HD,
        HISTORY_TABLE,
        row,
        head,
        checkpoint,
        length,
        kk,
        vv,
        HISTORY_TABLE_STRIDE,
        ROWS,
        HK_STRIDES,
        HU_STRIDES,
        HD_STRIDES,
        DK,
        DV,
        BH,
        state_history_layout,
    )
    if flush:
        dst = gl.load(
            STATE_TABLE + row * STATE_TABLE_STRIDE + (end - 1) // STATE_GRAIN
        ).to(gl.int64)
        # Only previously accepted history is flushed, never current candidates.
        # The destination must be request-writable, not a published snapshot.
        gl.store(STATE + dst * STATE_STRIDES[0] + state_feature, state, mask)
    # Verification and accepted-history updates share one reconstruction.
    # The verify phase uses the shared explicit arithmetic; history keeps
    # accepted replay's producer precision. Neither phase adds a state buffer.
    dual_history: gl.constexpr = FP32_PRODUCERS and T > 1
    align_verify: gl.constexpr = TRANSFORM_INPUTS and T > 1
    initial_state = state
    normalization_layout: gl.constexpr = gl.BlockedLayout(
        [1], [32], [gl.num_warps()], [0]
    )
    nk = gl.arange(0, BK, layout=normalization_layout).to(gl.int64)
    if TRANSFORM_INPUTS:
        a_scale = gl.exp(gl.load(A_LOG + head).to(gl.float32))
        bias = gl.load(DT_BIAS + head * DK + nk, nk < DK, 0).to(gl.float32)
    # Interleave the independent verify/history chains per token. This exposes
    # instruction-level overlap while keeping their state and rounding separate.
    verify_state = initial_state
    replay_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [gl.num_warps(), 1], [1, 0]
    )
    replay_state = gl.convert_layout(initial_state, replay_layout)
    for token_index in gl.static_range(T):
        token = width * 0 + token_index
        if token < width:
            for phase in gl.static_range(2 if dual_history else 1):
                if phase == 1:
                    # Match accepted replay's one-key-per-lane register ownership.
                    # Value rows are independent; retain the candidate's smaller BV.
                    history_state_layout: gl.constexpr = gl.BlockedLayout(
                        [1, 1], [1, 32], [gl.num_warps(), 1], [1, 0]
                    )
                    phase_kk = gl.arange(
                        0, BK, layout=gl.SliceLayout(0, history_state_layout)
                    ).to(gl.int64)
                    phase_vv = (
                        tile * BV
                        + gl.arange(
                            0, BV, layout=gl.SliceLayout(1, history_state_layout)
                        )
                    ).to(gl.int64)
                    state = replay_state
                else:
                    phase_kk = kk
                    phase_vv = vv
                    state = verify_state
                if phase == 0 and align_verify and token_index == 0:
                    # Match the ordinary verify's zero-initialized state accumulator.
                    state += 0.0
                if phase == 1:
                    # Original accepted replay normalizes K within one warp. Other
                    # warps replicate the vector; verify retains its own layout.
                    replay_normalization_layout: gl.constexpr = gl.SliceLayout(
                        0,
                        gl.BlockedLayout([1, 1], [1, 32], [gl.num_warps(), 1], [1, 0]),
                    )
                    phase_keys = gl.arange(
                        0, BK, layout=replay_normalization_layout
                    ).to(gl.int64)
                    if TRANSFORM_INPUTS:
                        phase_bias = gl.convert_layout(bias, phase_keys.type.layout)
                else:
                    phase_keys = nk
                    if TRANSFORM_INPUTS:
                        phase_bias = bias
                if phase == 0:
                    q = gl.load(
                        Q + _input_offset(row, token, head, phase_keys, Q_STRIDES),
                        phase_keys < DK,
                        0,
                    ).to(gl.float32)
                k = gl.load(
                    K + _input_offset(row, token, head, phase_keys, K_STRIDES),
                    phase_keys < DK,
                    0,
                ).to(gl.float32)
                v = gl.load(
                    V + _input_offset(row, token, head, phase_vv, V_STRIDES),
                    phase_vv < DV,
                    0,
                ).to(gl.float32)
                d = gl.load(
                    D + _input_offset(row, token, head, phase_keys, D_STRIDES),
                    phase_keys < DK,
                    0,
                ).to(gl.float32)
                if phase == 1 and HAS_HISTORY_INPUTS:
                    k = gl.load(
                        HISTORY_K_INPUT
                        + _input_offset(
                            row, token, head, phase_keys, HISTORY_INPUT_STRIDES[0]
                        ),
                        phase_keys < DK,
                        0,
                    ).to(gl.float32)
                if FP32_PRODUCERS and phase == 0:
                    q = q.to(gl.bfloat16).to(gl.float32)
                    k = k.to(gl.bfloat16).to(gl.float32)
                    v = v.to(gl.bfloat16).to(gl.float32)
                    d = d.to(gl.bfloat16).to(gl.float32)
                if TRANSFORM_INPUTS:
                    gate = d + phase_bias
                    if LOWER_BOUND is not None:
                        log_decay = LOWER_BOUND * _sigmoid(a_scale * gate)
                    else:
                        log_decay = -a_scale * gl.where(
                            gate < 20.0, gl.log(1 + gl.exp(gate)), gate
                        )
                    if phase == 0 and not align_verify:
                        q /= gl.sqrt(gl.sum(q * q, 0) + 1e-6)
                    if phase == 1:
                        # The original PTX rounds squares before their reduction.
                        # An opaque multiply prevents contraction into reduction FMAs,
                        # without disabling FMA for the recurrent update/projection.
                        squared = gl.inline_asm_elementwise(
                            "mul.rn.f32 $0, $1, $2;",
                            constraints="=f,f,f",
                            args=(k, k),
                            dtype=gl.float32,
                            is_pure=True,
                            pack=1,
                        )
                    else:
                        squared = k * k
                    if phase == 0 and align_verify:
                        q, k = verify_normalize(q, k, DK**-0.5, gl)
                    else:
                        k /= gl.sqrt(gl.sum(squared, 0) + 1e-6)
                    if phase == 1 and HAS_HISTORY_INPUTS:
                        log_decay = gl.load(
                            HISTORY_LOG_INPUT
                            + _input_offset(
                                row, token, head, phase_keys, HISTORY_INPUT_STRIDES[2]
                            ),
                            phase_keys < DK,
                            0,
                        ).to(gl.float32)
                    d = gl.exp(log_decay)
                d = gl.convert_layout(d, phase_kk.type.layout)
                k = gl.convert_layout(k, phase_kk.type.layout)
                if phase == 0 and align_verify:
                    beta = _sigmoid(
                        gl.load(
                            BETA
                            + row * BETA_STRIDES[0]
                            + token.to(gl.int64) * BETA_STRIDES[1]
                            + head * BETA_STRIDES[2]
                        ).to(gl.float32)
                    )
                    q = gl.convert_layout(q, phase_kk.type.layout)
                    state, output, correction = verify_recurrence(
                        state, q, k, v, d, beta, gl
                    )
                else:
                    state *= d[None, :]
                    projection = gl.sum(state * k[None, :], 1)
                    if phase == 1 and HAS_HISTORY_INPUTS:
                        # Replay fuses SiLU's final multiply with the subtraction.
                        v_acc = gl.load(
                            HISTORY_V_ACC
                            + _input_offset(
                                row, token, head, phase_vv, HISTORY_INPUT_STRIDES[1]
                            ),
                            phase_vv < DV,
                            0,
                        ).to(gl.float32)
                        correction = gl.fma(v_acc, _sigmoid(v_acc), -projection)
                    else:
                        correction = v - projection
                    beta = gl.load(
                        BETA
                        + row * BETA_STRIDES[0]
                        + token.to(gl.int64) * BETA_STRIDES[1]
                        + head * BETA_STRIDES[2]
                    ).to(gl.float32)
                    if TRANSFORM_INPUTS:
                        beta = _sigmoid(beta)
                    correction *= beta
                    if phase == 1:
                        state = gl.fma(correction[:, None], k[None, :], state)
                    else:
                        state += correction[:, None] * k[None, :]
                if phase == 0:
                    if not align_verify:
                        q = gl.convert_layout(q, phase_kk.type.layout)
                        output = gl.sum(state * q[None, :], 1)
                        output *= DK**-0.5
                    gl.store(
                        OUT + _input_offset(row, token, head, phase_vv, OUT_STRIDES),
                        output,
                        phase_vv < DV,
                    )
                if phase == 1 or not dual_history:
                    block, token_row = _history_offset(
                        HISTORY_TABLE, row, end + token, HISTORY_TABLE_STRIDE, ROWS
                    )
                    if tile == 0:
                        gl.store(
                            HK
                            + block * HK_STRIDES[0]
                            + token_row * HK_STRIDES[1]
                            + head * HK_STRIDES[2]
                            + phase_kk * HK_STRIDES[3],
                            k,
                            phase_kk < DK,
                        )
                        gl.store(
                            HD
                            + block * HD_STRIDES[0]
                            + token_row * HD_STRIDES[1]
                            + head * HD_STRIDES[2]
                            + phase_kk * HD_STRIDES[3],
                            d,
                            phase_kk < DK,
                        )
                    gl.store(
                        HU
                        + block * HU_STRIDES[0]
                        + token_row * HU_STRIDES[1]
                        + head * HU_STRIDES[2]
                        + phase_vv * HU_STRIDES[3],
                        correction,
                        phase_vv < DV,
                    )
                if phase == 0:
                    verify_state = state
                else:
                    replay_state = state
            state = verify_state


@register_kernel(
    "attention",
    "kda_buffered_recurrent",
    name="triton_kda_buffered_recurrent",
    solution="triton",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(10, 0),
        max_arch_version=ArchVersion(10, 99),
    ),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.bfloat16, torch.float32}
    ),
    priority=Priority.SPECIALIZED,
    traits={"head_dim": frozenset({128}), "recurrent_layout": frozenset({"v_major"})},
    tags={"nvidia", "cuda_graph", "buffered_replay"},
)
def triton_kda_buffered_recurrent(
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
    history_inputs,
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
        transform_inputs: True consumes conv(+SiLU) Q/K/V, raw f_b gate and
            beta logits, and writes BF16 verification outputs. BF16 Q/K/V use
            their supplied precision throughout. FP32 Q/K/V require an FP32
            gate: round producers to BF16 for verification, while T>1 uses
            their unrounded values for accepted history, matching replay's
            producer precision. T=1 retains BF16 state arithmetic. Both phases
            live only in registers and share one history reconstruction; verify
            uses the explicit arithmetic shared with unbuffered target verify.
            False consumes prepared FP32 inputs and writes FP32 outputs for
            reference/benchmark callers. This selects input representation,
            not a separate standard/speculative execution path.
        A_log/dt_bias: Contiguous FP32 [H]/[H*K] gate parameters when transforming;
            otherwise both must be None. No transformed-input workspace is built.
        history_inputs: Explicit tuple of FP32 replay-order K (after SiLU),
            value convolution accumulator (before SiLU), and log-decay, or None
            for inputs without separate replay producers. Pre-SiLU V preserves
            the original fused SiLU-minus-projection update. These are shared
            per-round workspace views, not persistent request state.
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
        Backing validation precedes recurrence in the same order for eager
        and CUDA graphs. Registration covers BF16/FP32 native Blackwell producers;
        direct prepared-FP32 calls remain the independent test contract.
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
    input_dtype = query.dtype if transform_inputs else torch.float32
    if input_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("native producers must be BF16 or FP32")
    output_dtype = torch.bfloat16 if transform_inputs else torch.float32
    fp32_producers = transform_inputs and input_dtype == torch.float32
    data = (
        (query, (batch, width, heads, key_dim), (input_dtype,)),
        (key, query.shape, (input_dtype,)),
        (value, (batch, width, heads, value_dim), (input_dtype,)),
        (
            decay,
            query.shape,
            (
                (torch.bfloat16, torch.float32)
                if transform_inputs and not fp32_producers
                else (torch.float32,)
            ),
        ),
        (
            beta,
            (batch, width, heads),
            (torch.bfloat16, torch.float32) if transform_inputs else (torch.float32,),
        ),
        (out, value.shape, (output_dtype,)),
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
    if history_inputs is not None:
        if not fp32_producers or width <= 1 or len(history_inputs) != 3:
            raise ValueError("history inputs require a multi-token native FP32 window")
        for tensor, shape in zip(
            history_inputs, (key.shape, value.shape, decay.shape), strict=True
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != torch.float32
                or any(s <= 0 for s in tensor.stride())
            ):
                raise ValueError(
                    "history K/value-accumulator/log-decay require matching FP32 views"
                )
            parameters.append(tensor)
    history_args = (None, None, None) if history_inputs is None else history_inputs
    history_strides = (
        (key.stride(), value.stride(), decay.stride())
        if history_inputs is None
        else tuple(t.stride() for t in history_inputs)
    )
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
    # Native 12-head shapes use measured static value tiles. With explicit
    # register-local history, L64 batches through B4 favor BV16 in the
    # rotating-layer working set. Larger batches keep BV32; this selection
    # never inspects per-request history lengths or changes the math.
    if (
        transform_inputs
        and width == 4
        and heads == 12
        and key_dim == value_dim == 128
        and (
            (capacity == 16 and batch <= 4)
            or (capacity == 32 and batch in (1, 4))
            or (capacity == 64 and batch <= 4)
        )
    ):
        value_tile = 16
    history_tile = min(
        4 if narrow_tile else 8, triton.next_power_of_2(capacity - width)
    )
    # Use the ordinary verify's measured launch geometry for native T4.
    # The two compile-time recurrence phases still share this single launch.
    align_verify = transform_inputs and width > 1
    num_warps = 1 if narrow_tile else 4
    num_stages = 1
    if align_verify and heads == 12 and key_dim == value_dim == 128:
        value_tile = 8 if batch < 16 else 16
        num_warps = 4 if batch <= 4 else 1
        num_stages = 3
    _buffered_recurrent[(batch, heads, triton.cdiv(value_dim, value_tile))](
        query,
        key,
        value,
        decay,
        beta,
        *history_args,
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
        history_strides,
        history_inputs is not None,
        transform_inputs,
        fp32_producers,
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
        num_warps=num_warps,
        num_stages=num_stages,
        enable_fp_fusion=align_verify,
    )
