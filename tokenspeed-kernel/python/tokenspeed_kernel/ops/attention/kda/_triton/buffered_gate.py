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


import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
    _gate_tiling,
    _gate_tiling_dot,
)


@triton.jit(
    do_not_specialize_on_alignment=["f_a", "f_b", "A_log", "dt_bias", "gate_scratch"]
)
def _history_gate(
    f_a,
    f_b,
    A_log,
    dt_bias,
    gate_scratch,
    rows,
    stride_fa: tl.constexpr,
    stride_gate: tl.constexpr,
    lower_bound: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """Retain the descriptor gate's reduction and unknown pointer alignment."""
    i_lh, i_tb, i_kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_l, i_hv = i_lh // HV, i_lh % HV
    o_k = i_kb * BK + tl.arange(0, BK)
    o_fa = tl.arange(0, D_FA)
    mask_k = o_k < K
    gc = i_hv * K + o_k
    wfb = tl.load(
        f_b + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    ).to(tl.float32)
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_bias = tl.load(dt_bias + gc, mask=mask_k, other=0.0).to(tl.float32)
    for j in range(BT):
        row = i_tb * BT + j
        if row < rows:
            fa = tl.load(f_a + row * stride_fa + o_fa).to(tl.float32)
            gate = tl.sum(wfb * fa[None, :], axis=1) + b_bias
            if lower_bound is not None:
                gate = lower_bound * tl.sigmoid(tl.exp(b_A) * gate)
            else:
                gate = -tl.exp(b_A) * tl.where(
                    gate < 20.0, tl.log(1 + tl.exp(gate)), gate
                )
            tl.store(gate_scratch + row * stride_gate + gc, gate, mask=mask_k)


@triton.jit(
    do_not_specialize_on_alignment=["f_a", "f_b", "A_log", "dt_bias", "gate_scratch"]
)
def _history_gate_dot(
    f_a,
    f_b,
    A_log,
    dt_bias,
    gate_scratch,
    rows,
    stride_fa: tl.constexpr,
    stride_gate: tl.constexpr,
    lower_bound: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    D_FA: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """Tensor-core form of the batched KDA gate precompute."""
    i_lh, i_tb, i_kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_l, i_hv = i_lh // HV, i_lh % HV
    o_k = i_kb * BK + tl.arange(0, BK)
    o_fa = tl.arange(0, D_FA)
    o_t = tl.arange(0, BT)
    mask_k = o_k < K
    gc = i_hv * K + o_k
    row = i_tb * BT + o_t
    mask_t = row < rows
    # Left as bf16: tl.dot accumulates in fp32, and the product of two bf16
    # values is exact there, so upcasting first would only cost bandwidth.
    wfb = tl.load(
        f_b + gc[:, None] * D_FA + o_fa[None, :],
        mask=mask_k[:, None],
        other=0.0,
    )
    fa = tl.load(
        f_a + row[:, None] * stride_fa + o_fa[None, :],
        mask=mask_t[:, None],
        other=0.0,
    )
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_bias = tl.load(dt_bias + gc, mask=mask_k, other=0.0).to(tl.float32)
    gate = tl.dot(fa, tl.trans(wfb)) + b_bias[None, :]
    if lower_bound is not None:
        gate = lower_bound * tl.sigmoid(tl.exp(b_A) * gate)
    else:
        gate = -tl.exp(b_A) * tl.where(gate < 20.0, tl.log(1 + tl.exp(gate)), gate)
    tl.store(
        gate_scratch + row[:, None] * stride_gate + gc[None, :],
        gate,
        mask=mask_t[:, None] & mask_k[None, :],
    )


def buffered_history_gate(
    f_a, f_b, A_log, dt_bias, out, *, num_heads, head_dim, local_layers, lower_bound
):
    """Write FP32 log-decay with the original accepted-replay reduction.

    f_a is BF16 [tokens,rank], f_b BF16 [heads*head_dim,rank], A_log and
    dt_bias contiguous FP32 [heads] and [heads*head_dim]. out is caller-owned
    contiguous FP32 [tokens,heads*head_dim]. local_layers selects the original
    batched gate's static tile, without descriptor construction or readback.
    Returns None. Width/batch only choose the reduction's established geometry.
    """
    rows = f_a.shape[0]
    if (
        f_a.ndim != 2
        or f_b.shape != (num_heads * head_dim, f_a.shape[1])
        or out.shape != (rows, num_heads * head_dim)
        or head_dim != 128
        or local_layers < 1
        or f_a.dtype != torch.bfloat16
        or f_b.dtype != torch.bfloat16
        or out.dtype != torch.float32
        or not out.is_contiguous()
        or A_log.shape != (num_heads,)
        or dt_bias.shape != (num_heads * head_dim,)
        or A_log.dtype != torch.float32
        or dt_bias.dtype != torch.float32
        or not A_log.is_contiguous()
        or not dt_bias.is_contiguous()
        or f_a.stride(1) != 1
        or not f_b.is_contiguous()
    ):
        raise ValueError("invalid buffered history gate geometry/dtype")
    if any(
        not tensor.is_cuda or tensor.device != f_a.device
        for tensor in (f_a, f_b, A_log, dt_bias, out)
    ):
        raise ValueError("history gate tensors must share a GPU")
    if rows == 0:
        return
    if rows >= 16:
        block_t, block_k = _gate_tiling_dot(rows, head_dim)
        kernel, warps = _history_gate_dot, 1
    else:
        block_t, block_k = _gate_tiling(
            rows, num_heads, head_dim, out.device, layers=local_layers
        )
        kernel, warps = _history_gate, 1 if block_t >= 4 else 2
    kernel[(num_heads, triton.cdiv(rows, block_t), triton.cdiv(head_dim, block_k))](
        f_a,
        f_b,
        A_log,
        dt_bias,
        out,
        rows,
        stride_fa=f_a.stride(0),
        stride_gate=out.stride(0),
        lower_bound=lower_bound,
        HV=num_heads,
        K=head_dim,
        D_FA=f_a.shape[1],
        BK=block_k,
        BT=block_t,
        num_warps=warps,
    )
