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


"""Unregistered, graph-safe position primitives for cache-owned KDA replay.

History stamps live in LCM pages, not request-slot arrays. These prototypes do
not materialize state or publish snapshots and are not used by serving yet.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _prepare_positions(
    STAMPS,
    TABLE,
    END,
    WIDTH,
    CHECKPOINT,
    LENGTH,
    FLUSH,
    OK,
    B: tl.constexpr,
    COLUMNS: tl.constexpr,
    PAGES: tl.constexpr,
    ROWS: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_WINDOW: tl.constexpr,
    HANDOFF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < B
    end = tl.load(END + row, live, 0).to(tl.int64)
    width = tl.load(WIDTH + row, live, 0)
    active = live & ((width > 0) | HANDOFF)
    previous = end - 1
    column = previous // ROWS
    in_table = active & (previous >= 0) & (column < COLUMNS)
    table_offset = row.to(tl.int64) * TABLE_STRIDE + column
    page = tl.load(TABLE + table_offset, in_table, 0).to(tl.int64)
    backed = in_table & (page > 0) & (page < PAGES)
    stamp = tl.load(
        STAMPS + page * PAGE_STRIDE + (previous % ROWS) * ROW_STRIDE,
        backed,
        0,
    )
    # A missing/zero previous row is the exact-endpoint prefill seed, not a
    # recovery path for lost committed history. Fresh pages must be zeroed.
    checkpoint = tl.where(stamp == 0, end, stamp - 1)
    length = end - checkpoint
    ok = (~active) | (
        (end >= 0)
        & (width <= MAX_WINDOW)
        & (stamp >= 0)
        & (checkpoint >= 0)
        & (length >= 0)
        & (length <= CAPACITY - MAX_WINDOW)
        & ((previous < 0) | (column < COLUMNS))
        & (page >= -1)
        & (page < PAGES)
    )
    ok = ok & (width >= 0)
    if HANDOFF:
        ok &= width == 0
    tl.store(CHECKPOINT + row, tl.where(active & ok, checkpoint, 0), live)
    tl.store(LENGTH + row, tl.where(active & ok, length, 0), live)
    # Handoff has no forward to perform a capacity flush: reconstruction must
    # start at the actual S_c, even when the next forward would have flushed.
    tl.store(
        FLUSH + row,
        active & ok & (length + 2 * MAX_WINDOW > CAPACITY) & (not HANDOFF),
        live,
    )
    tl.store(OK + row, ok, live)


@triton.jit
def _commit_positions(
    STAMPS,
    TABLE,
    END,
    WIDTH,
    ACCEPTED,
    CHECKPOINT,
    FLUSH,
    OK,
    MATERIALIZED,
    B: tl.constexpr,
    COLUMNS: tl.constexpr,
    PAGES: tl.constexpr,
    ROWS: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
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
    materialized = tl.load(MATERIALIZED + row, live, False)
    ok = tl.load(OK + row, live, False) & (accepted >= 0) & (accepted <= width)
    if HANDOFF:
        ok &= (width == 0) & (~flush)
    # Zero acceptance still advances c after a flush of old accepted history.
    write = (
        live & ok & ((width > 0) | HANDOFF) & ((accepted > 0) | flush | materialized)
    )
    token = end + accepted - 1
    column = token // ROWS
    in_table = write & (token >= 0) & (column < COLUMNS)
    table_offset = row.to(tl.int64) * TABLE_STRIDE + column
    page = tl.load(TABLE + table_offset, in_table, 0).to(tl.int64)
    backed = in_table & (page > 0) & (page < PAGES)
    ok = ok & ((~write) | backed)
    # Position/acceptance and block ownership are shared by the group's layers.
    # Store their stamps in one launch, after every layer's data stores finish.
    for layer in tl.static_range(len(STAMPS)):
        tl.store(
            STAMPS[layer] + page * PAGE_STRIDE + (token % ROWS) * ROW_STRIDE,
            tl.where(materialized, end + accepted, tl.where(flush, end, checkpoint))
            + 1,
            write & backed,
        )
    tl.store(OK + row, ok, live)


def _validate_positions(stamps, block_table, end, width, checkpoint, length, flush, ok):
    if stamps.ndim != 2 or stamps.dtype != torch.int64 or min(stamps.shape) < 1:
        raise ValueError("stamps must be int64 [pages, rows] with a null page")
    if (
        block_table.ndim != 2
        or block_table.dtype != torch.int32
        or block_table.stride(1) != 1
        or block_table.shape[1] < 1
    ):
        raise ValueError(
            "block_table must be int32 [batch, columns] with contiguous columns"
        )
    batch = block_table.shape[0]
    for tensor, dtype in (
        (end, torch.int32),
        (width, torch.int32),
        (checkpoint, torch.int64),
        (length, torch.int32),
        (flush, torch.bool),
        (ok, torch.bool),
    ):
        if (
            tensor.shape != (batch,)
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "position vectors must have contiguous batch shape and the declared dtype"
            )
    for tensor in (stamps, block_table, end, width, checkpoint, length, flush, ok):
        if not tensor.is_cuda or tensor.device != stamps.device:
            raise ValueError("all position tensors must be on the same GPU")
    if any(stride <= 0 for stride in stamps.stride()):
        raise ValueError("stamp strides must be positive")


def prepare_positions(
    stamps,
    block_table,
    end,
    width,
    checkpoint,
    length,
    flush,
    ok,
    *,
    capacity,
    max_window,
    for_handoff,
):
    """Fill caller-owned batch positions from the last committed history row.

    Args:
        stamps: Int64 [pages, rows] LCM field, with arbitrary positive strides.
            Zero means empty history; a committed row encodes checkpoint c+1.
        block_table: Int32 [batch, columns] raw absolute-token block table.
            Null/missing pages are 0/-1, never a cached physical request pointer.
        end: Int32 [batch] committed endpoints e at forward entry.
        width: Int32 [batch] valid input widths; zero is idle/padding.
        checkpoint: Int64 [batch] output materialized endpoints c.
        length: Int32 [batch] output committed history lengths e-c.
        flush: Bool [batch] output preemptive capacity-flush decisions.
        ok: Bool [batch] output validity; false must prevent forward/publication.
        capacity: Fixed logical history capacity, independent of block span.
        max_window: Fixed maximum valid width, for both ordinary/speculative use.
        for_handoff: All rows name quiescent endpoints, with zero input width.
            Read their stamps but never plan a capacity flush; there is no
            forward to execute it. Handoff batches contain no padding rows.

    Returns:
        None. Outputs are overwritten in place without allocation or readback.
        A zero stamp is legal only after exact-endpoint materialization and
        fresh-page zeroing; callers must not use it to recover missing history.
    """
    _validate_positions(stamps, block_table, end, width, checkpoint, length, flush, ok)
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int) or not 0 < v <= 2**31 - 1
            for v in (capacity, max_window)
        )
        or capacity < 2 * max_window
    ):
        raise ValueError("capacity must cover two positive int32 maximum windows")
    batch = end.numel()
    if batch:
        _prepare_positions[(triton.cdiv(batch, 128),)](
            stamps,
            block_table,
            end,
            width,
            checkpoint,
            length,
            flush,
            ok,
            batch,
            block_table.shape[1],
            stamps.shape[0],
            stamps.shape[1],
            *stamps.stride(),
            block_table.stride(0),
            capacity,
            max_window,
            for_handoff,
            128,
        )


def commit_positions(
    stamps,
    block_table,
    end,
    width,
    accepted,
    checkpoint,
    length,
    flush,
    ok,
    materialized,
    *,
    for_handoff,
):
    """Stamp all layers' last accepted input rows after data stores complete.

    ``stamps`` is a nonempty tuple of layer fields with identical shape, strides
    and device, all belonging to the same history group. The group's layers
    share positions; preparing from any one of these stamps is sufficient only
    when every layer completes before this common commit. Remaining arguments
    are the same buffers used by prepare_positions, plus int32
    [batch] accepted input counts (including the target input, no added one).
    A capacity flush must already have materialized state at e. With zero
    acceptance it restamps the previous committed row; otherwise only the last
    accepted row is stamped. Rejected candidates and idle rows are untouched.
    materialized is an explicit bool [batch] flag: when true, endpoint state
    at e+a must also have completed its stores, and that exact position is
    stamped even with zero acceptance. It does not itself fence those stores.
    for_handoff permits zero-width rows to stamp an endpoint reconstructed
    without a forward; all such rows are live and acceptance is zero.

    Returns None. Invalid acceptance or an unbacked destination clears the
    corresponding ok output and suppresses its store. The caller must consume
    ok before publication. This primitive is not an endpoint-publication gate.
    """
    if not isinstance(stamps, tuple) or not stamps:
        raise ValueError("commit requires a nonempty tuple of group stamp fields")
    first = stamps[0]
    _validate_positions(first, block_table, end, width, checkpoint, length, flush, ok)
    if (
        materialized.shape != width.shape
        or materialized.dtype != torch.bool
        or materialized.device != first.device
        or not materialized.is_contiguous()
    ):
        raise ValueError(
            "materialized must be a contiguous bool batch vector on the same GPU"
        )
    if any(
        t.shape != first.shape
        or t.stride() != first.stride()
        or t.dtype != first.dtype
        or t.device != first.device
        for t in stamps
    ):
        raise ValueError(
            "group stamp fields must share shape, strides, dtype and device"
        )
    if (
        accepted.shape != width.shape
        or accepted.dtype != torch.int32
        or accepted.device != first.device
        or not accepted.is_contiguous()
    ):
        raise ValueError(
            "accepted must be a contiguous int32 batch vector on the same GPU"
        )
    batch = end.numel()
    if batch:
        _commit_positions[(triton.cdiv(batch, 128),)](
            stamps,
            block_table,
            end,
            width,
            accepted,
            checkpoint,
            flush,
            ok,
            materialized,
            batch,
            block_table.shape[1],
            first.shape[0],
            first.shape[1],
            *first.stride(),
            block_table.stride(0),
            for_handoff,
            128,
        )


@triton.jit
def _refresh_decode_inputs(
    SEQ_LENS,
    END,
    WIDTH,
    B: tl.constexpr,
    LIVE,
    WINDOW: tl.constexpr,
    HANDOFF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = (row < B) & (row < LIVE)
    seq_len = tl.load(SEQ_LENS + row, active, WINDOW)
    tl.store(
        END + row, tl.where(active, seq_len - (0 if HANDOFF else WINDOW), 0), row < B
    )
    tl.store(WIDTH + row, tl.where(active, 0 if HANDOFF else WINDOW, 0), row < B)


def refresh_decode_inputs(seq_lens, end, width, *, actual_bs, max_window, for_handoff):
    """Refresh fixed-address int32 endpoint/width vectors for one decode batch.

    ``seq_lens`` includes the fixed target input window. Live rows therefore
    start at ``seq_lens - max_window``; padding is (end=0, width=0). Width one
    uses the same operation. All vectors are on one GPU; no allocation/readback.
    ``end``/``width`` have the padded batch shape and contiguous storage;
    ``seq_lens`` need only contain the live rows. Negative live endpoints remain
    invalid for position preparation rather than being silently clamped.
    for_handoff takes exact accepted endpoints instead of decode sequence
    lengths, sets width zero and requires an unpadded batch. No candidates are
    implied by this refresh.
    """
    batch = end.numel()
    if not 0 <= actual_bs <= batch or max_window <= 0:
        raise ValueError("invalid live batch or maximum window")
    if for_handoff and actual_bs != batch:
        raise ValueError("handoff rows must all be live")
    for tensor in (seq_lens, end, width):
        if (
            tensor.ndim != 1
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
            or not tensor.is_cuda
            or tensor.device != end.device
        ):
            raise ValueError("decode positions require contiguous int32 GPU vectors")
    if width.shape != end.shape or seq_lens.numel() < actual_bs:
        raise ValueError("decode position vectors do not cover the batch")
    if batch:
        _refresh_decode_inputs[(triton.cdiv(batch, 128),)](
            seq_lens,
            end,
            width,
            batch,
            actual_bs,
            max_window,
            for_handoff,
            128,
        )
