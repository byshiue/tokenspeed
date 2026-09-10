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

"""Pack token-major KDA gate and beta inputs without changing their values."""

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _prepare_kda_prefill_gate_beta_kernel(
    gate,
    beta,
    gate_out,
    beta_out,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    gate_count: tl.constexpr,
    beta_count: tl.constexpr,
    gate_s0: tl.constexpr,
    gate_s1: tl.constexpr,
    gate_s2: tl.constexpr,
    gate_s3: tl.constexpr,
    beta_s0: tl.constexpr,
    beta_s1: tl.constexpr,
    beta_s2: tl.constexpr,
    WRITE_GATE: tl.constexpr,
    WRITE_BETA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if WRITE_GATE:
        row = offsets // dim
        source = (
            row // (tokens * heads) * gate_s0
            + row // heads % tokens * gate_s1
            + row % heads * gate_s2
            + offsets % dim * gate_s3
        )
        gate_values = tl.load(gate + source, offsets < gate_count, other=0).to(
            tl.float32
        )
        tl.store(gate_out + offsets, gate_values, offsets < gate_count)
    if WRITE_BETA:
        # Beta is much smaller than gate; only its own tiles do the gather.
        if tl.program_id(0) * BLOCK < beta_count:
            source = (
                offsets // (tokens * heads) * beta_s0
                + offsets // heads % tokens * beta_s1
                + offsets % heads * beta_s2
            )
            beta_values = tl.load(beta + source, offsets < beta_count, other=0)
            tl.store(beta_out + offsets, beta_values, offsets < beta_count)


def prepare_kda_prefill_gate_beta(
    gate: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare contiguous gate and beta operands for the token-major KDA ABI.

    Args:
        gate: Strided floating-point [batch, tokens, heads, key_dim] logits.
        beta: Strided floating-point [batch, tokens, heads] logits on the same
            device. Both inputs support float16, bfloat16, and float32.

    Returns:
        Contiguous FP32 gate and contiguous beta with its original dtype.
        Already prepared operands are returned unchanged. CUDA casts/packs
        the remaining operands in one launch, with no sigmoid, gate math or
        reduction; CPU uses the corresponding PyTorch operations. Inputs are
        read-only, and newly allocated outputs belong to this call.
    """
    if gate.ndim != 4 or beta.shape != gate.shape[:-1] or gate.shape[-1] == 0:
        raise ValueError("KDA gate must be [B, T, H, D] and beta [B, T, H]")
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if gate.dtype not in dtypes or beta.dtype not in dtypes:
        raise ValueError("KDA gate and beta must be float16, bfloat16 or float32")
    if gate.device != beta.device:
        raise ValueError("KDA gate and beta must be on the same device")
    if not gate.is_cuda:
        return gate.float().contiguous(), beta.contiguous()

    write_gate = gate.dtype != torch.float32 or not gate.is_contiguous()
    write_beta = not beta.is_contiguous()
    gate_out = (
        torch.empty(gate.shape, dtype=torch.float32, device=gate.device)
        if write_gate
        else gate
    )
    beta_out = (
        torch.empty(beta.shape, dtype=beta.dtype, device=beta.device)
        if write_beta
        else beta
    )
    count = max(gate.numel() if write_gate else 0, beta.numel() if write_beta else 0)
    if count:
        _prepare_kda_prefill_gate_beta_kernel[(triton.cdiv(count, 512),)](
            gate,
            beta,
            gate_out,
            beta_out,
            gate.shape[1],
            gate.shape[2],
            gate.shape[3],
            gate.numel(),
            beta.numel(),
            *gate.stride(),
            *beta.stride(),
            WRITE_GATE=write_gate,
            WRITE_BETA=write_beta,
            BLOCK=512,
            num_warps=4,
        )
    return gate_out, beta_out
