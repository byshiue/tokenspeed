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

"""Portable recurrence prototype for the buffered-replay semantic milestone.

Not registered for serving: LCM ownership/publication integration must precede
runtime dispatch. These primitives consume caller-owned storage and allocate
nothing. They intentionally use a serial recurrence before kernel tuning.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


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
    START,
    LENGTH,
    VALID,
    FLUSH,
    OUT,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    T: tl.constexpr,
    L: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    width = tl.load(VALID + row)
    start = tl.load(START + row)
    length = tl.load(LENGTH + row)
    flush = (width > 0) & (length + 2 * T > L)
    if head == 0 and tile == 0:
        tl.store(FLUSH + row, flush)
    if width == 0:
        return
    kk = tl.arange(0, BK)
    vv = tile * BV + tl.arange(0, BV)
    mask = (vv[:, None] < DV) & (kk[None, :] < DK)
    state_ptr = STATE + ((row * H + head) * DV + vv[:, None]) * DK + kk[None, :]
    state = tl.load(state_ptr, mask, 0).to(tl.float32)
    # Each history correction was computed using only its causal prefix.
    for offset in range(length):
        slot = (start + offset) % L
        hk = tl.load(HK + ((row * L + slot) * H + head) * DK + kk, kk < DK, 0)
        hd = tl.load(HD + ((row * L + slot) * H + head) * DK + kk, kk < DK, 0)
        hu = tl.load(HU + ((row * L + slot) * H + head) * DV + vv, vv < DV, 0)
        state = state * hd[None, :] + hu[:, None] * hk[None, :]
    if flush:
        # Only the accepted state at round entry is materialized.
        tl.store(state_ptr, state, mask)
    candidate_start = (start + length) % L
    for token in range(width):
        q = tl.load(Q + ((row * T + token) * H + head) * DK + kk, kk < DK, 0).to(
            tl.float32
        )
        k = tl.load(K + ((row * T + token) * H + head) * DK + kk, kk < DK, 0).to(
            tl.float32
        )
        v = tl.load(V + ((row * T + token) * H + head) * DV + vv, vv < DV, 0).to(
            tl.float32
        )
        d = tl.load(D + ((row * T + token) * H + head) * DK + kk, kk < DK, 0).to(
            tl.float32
        )
        beta = tl.load(BETA + (row * T + token) * H + head).to(tl.float32)
        state *= d[None, :]
        correction = beta * (v - tl.sum(state * k[None, :], 1))
        slot = (candidate_start + token) % L
        if tile == 0:
            tl.store(HK + ((row * L + slot) * H + head) * DK + kk, k, kk < DK)
            tl.store(HD + ((row * L + slot) * H + head) * DK + kk, d, kk < DK)
        tl.store(HU + ((row * L + slot) * H + head) * DV + vv, correction, vv < DV)
        state += correction[:, None] * k[None, :]
        out = tl.sum(state * q[None, :], 1) * (DK**-0.5)
        tl.store(OUT + ((row * T + token) * H + head) * DV + vv, out, vv < DV)


@triton.jit
def _buffered_commit(
    START,
    LENGTH,
    VALID,
    ACCEPTED,
    FLUSH,
    N: tl.constexpr,
    L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = row < N
    width = tl.load(VALID + row, live, 0)
    accepted = tl.load(ACCEPTED + row, live, 0)
    start = tl.load(START + row, live, 0)
    length = tl.load(LENGTH + row, live, 0)
    flush = tl.load(FLUSH + row, live, 0)
    # Invalid acceptance is a caller contract violation, not a clamp policy.
    tl.device_assert(
        (accepted >= 0) & (accepted <= width), "invalid accepted input count"
    )
    active = live & (width > 0)
    tl.store(START + row, tl.where(flush, (start + length) % L, start), active)
    tl.store(LENGTH + row, tl.where(flush, 0, length) + accepted, active)


def buffered_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    checkpoint: torch.Tensor,
    history_key: torch.Tensor,
    history_correction: torch.Tensor,
    history_decay: torch.Tensor,
    start: torch.Tensor,
    length: torch.Tensor,
    valid: torch.Tensor,
    flushed: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compute candidate outputs/history; write checkpoint only on capacity flush.

    query/key/decay: [B,T,H,K]; value/out: [B,T,H,V]; beta: [B,T,H].
    Q/K must already be L2-normalized; beta is sigmoid-transformed and decay
    is exp(log_decay). checkpoint is [B,H,V,K], history fields are
    [B,L,H,K/V/K]. All floating-point tensors are contiguous FP32.
    start/length/valid are contiguous INT32 [B] device metadata; flushed is a
    BOOL [B] output. T is the fixed maximum window, valid gives each row's
    active width (zero for padding). Buffers are caller-owned and exclusive
    for this operation; immutable published snapshots must never be passed as
    writable checkpoints. Call buffered_commit once after acceptance and before
    reusing any inputs or metadata. No CPU readback or allocation occurs here.
    """
    batch, width, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    capacity = history_key.shape[1]
    if width < 1 or capacity < 2 * width:
        raise ValueError("capacity must cover two positive execution windows")
    expected = (
        (query, (batch, width, heads, key_dim), torch.float32),
        (key, query.shape, torch.float32),
        (decay, query.shape, torch.float32),
        (value, (batch, width, heads, value_dim), torch.float32),
        (beta, (batch, width, heads), torch.float32),
        (out, value.shape, torch.float32),
        (checkpoint, (batch, heads, value_dim, key_dim), torch.float32),
        (history_key, (batch, capacity, heads, key_dim), torch.float32),
        (history_correction, (batch, capacity, heads, value_dim), torch.float32),
        (history_decay, history_key.shape, torch.float32),
        (start, (batch,), torch.int32),
        (length, (batch,), torch.int32),
        (valid, (batch,), torch.int32),
        (flushed, (batch,), torch.bool),
    )
    for tensor, shape, dtype in expected:
        if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(
                f"expected contiguous {dtype} tensor of shape {tuple(shape)}"
            )
        if not tensor.is_cuda or tensor.device != query.device:
            raise ValueError("all buffered recurrence tensors must share a GPU")
    if batch == 0:
        return
    _buffered_recurrent[(batch, heads, triton.cdiv(value_dim, 32))](
        query,
        key,
        value,
        decay,
        beta,
        checkpoint,
        history_key,
        history_correction,
        history_decay,
        start,
        length,
        valid,
        flushed,
        out,
        H=heads,
        DK=key_dim,
        DV=value_dim,
        T=width,
        L=capacity,
        BK=triton.next_power_of_2(key_dim),
        BV=32,
        num_warps=4,
        enable_fp_fusion=False,
    )


def buffered_commit(
    start: torch.Tensor,
    length: torch.Tensor,
    valid: torch.Tensor,
    accepted: torch.Tensor,
    flushed: torch.Tensor,
    capacity: int,
) -> None:
    """Publish accepted candidate counts using the forward's flush decisions.

    All metadata are caller-owned contiguous [B] GPU arrays: INT32 except
    BOOL flushed. accepted counts state-input tokens, without an implicit +1.
    Rejected history remains physically present but outside the committed range.
    Conv-window commit and absolute endpoint publication are integration work;
    this recurrence prototype changes only start/length, with no state replay.
    """
    for tensor, dtype in (
        (start, torch.int32),
        (length, torch.int32),
        (valid, torch.int32),
        (accepted, torch.int32),
        (flushed, torch.bool),
    ):
        if (
            tensor.shape != start.shape
            or tensor.ndim != 1
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
            or not tensor.is_cuda
            or tensor.device != start.device
        ):
            raise ValueError("incompatible buffered commit metadata")
    if capacity < 2:
        raise ValueError("capacity must be at least two")
    batch = start.numel()
    if batch:
        _buffered_commit[(triton.cdiv(batch, 128),)](
            start,
            length,
            valid,
            accepted,
            flushed,
            N=batch,
            L=capacity,
            BLOCK=128,
        )
