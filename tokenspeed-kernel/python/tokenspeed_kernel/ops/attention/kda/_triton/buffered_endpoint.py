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

"""Accepted-endpoint materialization; no scheduler publication is issued here."""

import math

import torch
from tokenspeed_kernel._triton import gl, gluon, tl, triton
from tokenspeed_kernel.ops.attention.kda._triton.buffered import _reconstruct_history


@triton.jit
def _prepare_endpoint_commit(
    END,
    WIDTH,
    ACCEPTED,
    CHECKPOINT,
    FLUSH,
    OK,
    FORCE,
    MATERIALIZED,
    B: tl.constexpr,
    PREFIX: tl.constexpr,
    HANDOFF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < B
    end = tl.load(END + row, live, 0).to(tl.int64)
    width = tl.load(WIDTH + row, live, 0)
    accepted = tl.load(ACCEPTED + row, live, 0)
    checkpoint = tl.load(CHECKPOINT + row, live, 0)
    flush = tl.load(FLUSH + row, live, False)
    force = tl.load(FORCE + row, live, False)
    endpoint = end + accepted
    ok = (
        tl.load(OK + row, live, False)
        & (accepted >= 0)
        & (accepted <= width)
        & (endpoint <= 2**31 - 1)
    )
    source = tl.where(flush, end, checkpoint)
    if HANDOFF:
        ok &= (width == 0) & (~flush)
    needed = (
        live
        & ok
        & ((width > 0) | HANDOFF)
        & (endpoint > source)
        & (force | HANDOFF | (endpoint % PREFIX == 0))
    )
    tl.store(MATERIALIZED + row, needed, live)
    tl.store(OK + row, ok, live)


def prepare_endpoint_commit(
    end,
    width,
    accepted,
    checkpoint,
    flushed,
    ok,
    force,
    materialized,
    *,
    prefix_granularity,
    for_handoff,
):
    """Choose exact-endpoint writes after acceptance, without CPU readback.

    end/width/accepted are int32 [B], checkpoint int64 [B], and the remaining
    vectors bool [B], all contiguous on one GPU. force is a caller-owned handoff
    request, not provenance. Otherwise only an aligned accepted endpoint is
    selected, using prefix identity granularity rather than the state span.
    A capacity flush already materialized S_e; only history beyond that source
    is reconstructed. The materialized output means a write is needed, not that
    it has completed. Run materialize_endpoints before publishing its stamp.
    Invalid acceptance clears ok before any commit stores. Returns None.
    for_handoff selects every lagging endpoint, irrespective of prefix identity;
    width/acceptance and flushed must be zero after fresh handoff preparation.
    """
    batch = end.numel()
    if (
        isinstance(prefix_granularity, bool)
        or not isinstance(prefix_granularity, int)
        or prefix_granularity <= 0
    ):
        raise ValueError("positive prefix identity granularity required")
    for tensor, dtype in (
        (end, torch.int32),
        (width, torch.int32),
        (accepted, torch.int32),
        (checkpoint, torch.int64),
        (flushed, torch.bool),
        (ok, torch.bool),
        (force, torch.bool),
        (materialized, torch.bool),
    ):
        if (
            tensor.shape != (batch,)
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "endpoint commit requires contiguous vectors with declared dtypes"
            )
        if not tensor.is_cuda or tensor.device != end.device:
            raise ValueError("endpoint commit vectors must share one GPU")
    if batch:
        _prepare_endpoint_commit[(triton.cdiv(batch, 128),)](
            end,
            width,
            accepted,
            checkpoint,
            flushed,
            ok,
            force,
            materialized,
            batch,
            prefix_granularity,
            for_handoff,
            128,
            num_warps=4,
        )


@gluon.jit
def _materialize_endpoints(
    DESCRIPTORS,
    GROUPS,
    END,
    ACCEPTED,
    CHECKPOINT,
    FLUSH,
    OK,
    MATERIALIZED,
    B: gl.constexpr,
    LAYERS: gl.constexpr,
    GROUP_COUNT: gl.constexpr,
    FLAG_BLOCK: gl.constexpr,
    MAX_B: gl.constexpr,
    H: gl.constexpr,
    DK: gl.constexpr,
    DV: gl.constexpr,
    ROWS: gl.constexpr,
    STATE_GRAIN: gl.constexpr,
    TABLE_STRIDE: gl.constexpr,
    STATE_STRIDES: gl.constexpr,
    HK_STRIDES: gl.constexpr,
    HU_STRIDES: gl.constexpr,
    HD_STRIDES: gl.constexpr,
    BK: gl.constexpr,
    BV: gl.constexpr,
    BH: gl.constexpr,
):
    # Bound empty rounds to a small persistent grid, rather than scheduling
    # every layer/head/state tile only to discover no endpoint was requested.
    flag = gl.arange(
        0, FLAG_BLOCK, layout=gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    )
    flag_live = (flag < GROUP_COUNT * MAX_B) & (flag % MAX_B < B)
    any_work = gl.sum(
        (
            gl.load(MATERIALIZED + flag, flag_live, False)
            & gl.load(OK + flag, flag_live, False)
        ).to(gl.int32),
        0,
    )
    if any_work == 0:
        return
    for work in range(
        gl.program_id(0), LAYERS * H * B * gl.cdiv(DV, BV), gl.num_programs(0)
    ):
        tile = work % gl.cdiv(DV, BV)
        row = ((work // gl.cdiv(DV, BV)) % B).to(gl.int64)
        layer_head = work // (B * gl.cdiv(DV, BV))
        layer, head = layer_head // H, (layer_head % H).to(gl.int64)
        group = gl.load(GROUPS + layer).to(gl.int64)
        index = group * MAX_B + row
        if gl.load(MATERIALIZED + index) & gl.load(OK + index):
            end = gl.load(END + row).to(gl.int64)
            endpoint = end + gl.load(ACCEPTED + row)
            checkpoint = gl.where(
                gl.load(FLUSH + index), end, gl.load(CHECKPOINT + index)
            )
            desc = DESCRIPTORS + layer.to(gl.int64) * 6
            state_ptr = gl.load(desc).to(gl.pointer_type(gl.float32))
            hk = gl.load(desc + 1).to(gl.pointer_type(gl.float32))
            hu = gl.load(desc + 2).to(gl.pointer_type(gl.float32))
            hd = gl.load(desc + 3).to(gl.pointer_type(gl.float32))
            history_table = gl.load(desc + 4).to(gl.pointer_type(gl.int32))
            state_table = gl.load(desc + 5).to(gl.pointer_type(gl.int32))
            src = gl.load(
                state_table + row * TABLE_STRIDE + (checkpoint - 1) // STATE_GRAIN,
                checkpoint > 0,
                0,
            ).to(gl.int64)
            dst = gl.load(
                state_table + row * TABLE_STRIDE + (endpoint - 1) // STATE_GRAIN
            ).to(gl.int64)
            state_history_layout: gl.constexpr = gl.BlockedLayout(
                [1, 1, 4], [1, 1, 32], [gl.num_warps(), 1, 1], [2, 0, 1]
            )
            state_layout: gl.constexpr = gl.SliceLayout(1, state_history_layout)
            kk = gl.arange(0, BK, layout=gl.SliceLayout(0, state_layout)).to(gl.int64)
            vv = (
                tile * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, state_layout))
            ).to(gl.int64)
            mask = (vv[:, None] < DV) & (kk[None, :] < DK)
            feature = (
                head * STATE_STRIDES[1]
                + vv[:, None] * STATE_STRIDES[2]
                + kk[None, :] * STATE_STRIDES[3]
            )
            state = gl.load(
                state_ptr + src * STATE_STRIDES[0] + feature, mask & (checkpoint > 0), 0
            )
            state = _reconstruct_history(
                state,
                hk,
                hu,
                hd,
                history_table,
                row,
                head,
                checkpoint,
                endpoint - checkpoint,
                kk,
                vv,
                TABLE_STRIDE,
                ROWS,
                HK_STRIDES,
                HU_STRIDES,
                HD_STRIDES,
                DK,
                DV,
                BH,
                state_history_layout,
            )
            # Each program owns a disjoint state tile, safe even when src == dst.
            # The accepted range ends at endpoint; rejected K/U/D entries are unread.
            gl.store(state_ptr + dst * STATE_STRIDES[0] + feature, state, mask)


def materialize_endpoints(
    descriptors,
    groups,
    end,
    accepted,
    checkpoint,
    flushed,
    ok,
    materialized,
    *,
    heads,
    key_dim,
    value_dim,
    history_block_tokens,
    state_block_tokens,
    table_stride,
    state_strides,
    key_strides,
    correction_strides,
    decay_strides,
    max_programs,
):
    """Materialize selected accepted endpoints for all local layers in one launch.

    descriptors is int64 [layers,6]: FP32 state/K/U/D and int32 history/state
    raw-table pointers. groups is int32 [layers]; fields and tables have the
    explicit shared strides. end is int32 [max_bs], accepted int32 [live_bs];
    checkpoint is int64 [groups,max_bs], flushed/ok/materialized bool of that
    shape. All storage is caller-owned, GPU-resident and stable through commit.
    max_programs is a positive startup-fixed persistent-grid bound. Empty
    rounds return after checking live flags; active programs stride over all
    selected tiles without a host decision or a different graph.
    Persistent grids use a stride coprime to the request/value-tile cycle so
    programs are not pinned to inactive rows in a partially materialized batch.

    Preparation must validate the full history/state backing before data stores;
    prepare_endpoint_commit validates acceptance after all layer forwards, or
    zero acceptance after fresh quiescent-handoff preparation.
    Destinations must be request-writable, never published immutable snapshots.
    A following stamp commit, completion fence and owner protocol are required
    before any external handoff. Returns None; no allocations or host readback.
    """
    batch, max_bs = accepted.numel(), end.numel()
    if (
        isinstance(max_programs, bool)
        or not isinstance(max_programs, int)
        or max_programs <= 0
    ):
        raise ValueError("positive persistent endpoint grid bound required")
    if (
        descriptors.ndim != 2
        or descriptors.shape[1] != 6
        or descriptors.dtype != torch.int64
        or not descriptors.is_contiguous()
    ):
        raise ValueError("endpoint descriptors require contiguous int64 [layers,6]")
    layers = descriptors.shape[0]
    if (
        groups.shape != (layers,)
        or groups.dtype != torch.int32
        or not groups.is_contiguous()
    ):
        raise ValueError("endpoint group indices require contiguous int32 [layers]")
    if (
        batch > max_bs
        or end.shape != (max_bs,)
        or end.dtype != torch.int32
        or not end.is_contiguous()
        or accepted.shape != (batch,)
        or accepted.dtype != torch.int32
        or not accepted.is_contiguous()
    ):
        raise ValueError("endpoint counts require contiguous int32 batch vectors")
    if (
        checkpoint.ndim != 2
        or checkpoint.shape[1] != max_bs
        or checkpoint.dtype != torch.int64
        or not checkpoint.is_contiguous()
    ):
        raise ValueError(
            "endpoint checkpoints require contiguous int64 [groups,max_bs]"
        )
    for tensor in (flushed, ok, materialized):
        if (
            tensor.shape != checkpoint.shape
            or tensor.dtype != torch.bool
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "endpoint flags require contiguous bool group/batch matrices"
            )
    if any(
        not t.is_cuda or t.device != end.device
        for t in (
            descriptors,
            groups,
            end,
            accepted,
            checkpoint,
            flushed,
            ok,
            materialized,
        )
    ):
        raise ValueError("endpoint tensors must share one GPU")
    if min(
        heads,
        key_dim,
        value_dim,
        history_block_tokens,
        state_block_tokens,
        table_stride,
    ) <= 0 or any(
        len(s) != 4 or min(s) <= 0
        for s in (state_strides, key_strides, correction_strides, decay_strides)
    ):
        raise ValueError("positive endpoint field geometry required")
    if batch and layers:
        row_period = batch * triton.cdiv(value_dim, 32)
        total_work = layers * heads * row_period
        programs = min(max_programs, total_work)
        if programs < total_work:
            # The flattened work order cycles through each row's value tiles.
            # A common factor can keep programs on the same inactive rows for
            # every iteration. Adjust only the static stride, within the cap;
            # flags, arithmetic and grids with one tile per program are unchanged.
            while math.gcd(programs, row_period) != 1:
                programs -= 1
        _materialize_endpoints[(programs,)](
            descriptors,
            groups,
            end,
            accepted,
            checkpoint,
            flushed,
            ok,
            materialized,
            batch,
            layers,
            checkpoint.shape[0],
            triton.next_power_of_2(checkpoint.shape[0] * max_bs),
            max_bs,
            heads,
            key_dim,
            value_dim,
            history_block_tokens,
            state_block_tokens,
            table_stride,
            state_strides,
            key_strides,
            correction_strides,
            decay_strides,
            triton.next_power_of_2(key_dim),
            32,
            8,
            num_warps=4,
            enable_fp_fusion=False,
        )
