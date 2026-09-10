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

"""CuteDSL KDA adapter for the chunked KDA prefill scan.

States use the native ``[N, HV, V, K]`` convention, matching the NVIDIA
cache and the CuteDSL ABI. The dispatch facade adapts other cache layouts;
this wrapper must not transpose the state a second time. The native
token-major build reads ``[B, T, H, D]`` activations directly. Gate casting
and packing strided beta logits share one preparation kernel. Sigmoid(beta),
the safe gate, and QK L2 normalization run in-kernel, like the FLA and
FlashKDA paths. The safe-gate lower bound is baked into the CUBIN and
validated on every call.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.triton.kda_prefill_inputs import (
    prepare_kda_prefill_gate_beta,
)
from tokenspeed_kernel.thirdparty.cutedsl_kda import (
    DEFAULT_SCALE,
    cutedsl_kda_check_config,
    cutedsl_kda_forward,
    cutedsl_kda_supports_output_buffer,
    cutedsl_kda_workspace_size,
    is_cutedsl_kda_installed,
)

__all__ = ["cutedsl_kda_chunk_prefill", "is_cutedsl_kda_installed"]


def cutedsl_kda_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    lower_bound: float | None = None,
    beta_is_logit: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked prefill KDA scan through the CuteDSL KDA kernel (varlen native).

    Args:
        q: Query ``[B, T, H, K]`` (bfloat16; raw or pre-normalized — the
            kernel L2-normalizes, which is idempotent).
        k: Key, same shape/dtype rules as ``q``.
        v: Value ``[B, T, HV, V]`` bfloat16.
        g_raw: Raw per-channel decay logits ``[B, T, HV, K]``.
        beta: Raw beta logits ``[B, T, HV]``; sigmoid is applied in-kernel.
        A_log: Per-head FP32 decay parameter ``[HV]``.
        dt_bias: FP32 gate bias with ``HV * K`` elements.
        initial_state: Optional FP32 recurrent state per packed sequence in
            the native ``[N, HV, V, K]`` convention; ``None`` starts from
            zero.
        cu_seqlens: Cumulative sequence boundaries ``[N + 1]`` (``B`` must
            be 1); ``None`` treats each batch row as one sequence.
        cu_seqlens_cpu: Host int64 copy of ``cu_seqlens`` whose contents
            MUST equal it; REQUIRED whenever ``cu_seqlens`` is given. The
            kernel wrapper plans launch grids, routing, and workspace
            partitioning on the host from the boundary values; reading them
            back instead would be a stream-synchronizing D2H copy on every
            call. Runtime callers share a device int64 boundary tensor across
            layers, making the conversion below a no-op. Standalone callers
            may still supply int32 boundaries.
        lower_bound: Safe-gate lower bound; required, and must match the
            value baked into the CUBIN (validated).
        beta_is_logit: Must be True; the kernel always applies sigmoid.
        out: Contiguous destination with v's shape, dtype and device, or None
            to allocate. Must not overlap inputs. Older native wrappers use a
            copy fallback; output-buffer-capable wrappers write it directly.

    Returns:
        ``(o [B, T, HV, V], final_state [N, HV, V, K])`` in native layout.
    """
    if not beta_is_logit:
        raise ValueError("cutedsl_kda_chunk_prefill requires raw beta logits")
    if lower_bound is None:
        raise ValueError("cutedsl_kda_chunk_prefill requires a safe-gate bound")
    if dt_bias is None:
        raise ValueError("cutedsl_kda_chunk_prefill requires dt_bias")
    # The bound is a compile-time CUBIN constant; mismatches must fail
    # loudly rather than silently mis-gate.
    cutedsl_kda_check_config(float(lower_bound))
    batch, tokens, num_heads, key_dim = q.shape
    destination = out
    if out is not None and (
        out.shape != v.shape
        or out.dtype != v.dtype
        or out.device != v.device
        or not out.is_contiguous()
    ):
        raise ValueError("KDA out must be contiguous with v's shape, dtype and device")
    num_value_heads, value_dim = v.shape[2], v.shape[-1]
    if cu_seqlens is not None:
        num_sequences = cu_seqlens.numel() - 1
        boundaries = cu_seqlens.to(dtype=torch.int64)
        if cu_seqlens_cpu is None:
            raise ValueError(
                "cutedsl_kda_chunk_prefill requires cu_seqlens_cpu alongside "
                "cu_seqlens (host int64 copy with equal contents)"
            )
        if len(cu_seqlens_cpu) != num_sequences + 1:
            # A wrong copy would silently corrupt the host-side chunk plan;
            # a length mismatch means the caller wired the wrong tensor.
            raise ValueError(
                f"cu_seqlens_cpu has {len(cu_seqlens_cpu)} entries, "
                f"cu_seqlens has {num_sequences + 1}"
            )
    else:
        # The kernel is varlen-only with a unit batch dim; token-major
        # memory lets batch rows flatten to packed sequences as pure views.
        num_sequences = batch
        boundaries = torch.arange(
            0, (batch + 1) * tokens, tokens, device=q.device, dtype=torch.int64
        )
        cu_seqlens_cpu = torch.arange(
            0, (batch + 1) * tokens, tokens, dtype=torch.int64
        )
        q, k, v = (t.reshape(1, batch * tokens, -1, t.shape[-1]) for t in (q, k, v))
        g_raw = g_raw.reshape(1, batch * tokens, num_value_heads, key_dim)
        beta = beta.reshape(1, batch * tokens, num_value_heads)
    # Native token-major ABI: no head-major re-layout. Gate must be FP32
    # and beta may be a strided slice of the merged projection, so pack
    # both together into the contiguous memory the kernel descriptors index.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g_f32, beta = prepare_kda_prefill_gate_beta(g_raw, beta)
    dt_bias = dt_bias.reshape(num_value_heads, key_dim).contiguous()
    A_log = A_log.contiguous()
    # The dispatch layout trait already matches the native [N, HV, V, K]
    # ABI. A second transpose here would undo the dispatcher's conversion.
    if initial_state is not None:
        state_in = initial_state.contiguous()
    else:
        state_in = torch.zeros(
            num_sequences,
            num_value_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=q.device,
        )
    # Decomposition-route scratch (0 bytes on the engine route); preallocated
    # here so the wrapper does not allocate on the hot path.
    ws_bytes = cutedsl_kda_workspace_size(
        boundaries, num_value_heads, cu_seqlens_cpu=cu_seqlens_cpu
    )
    workspace = (
        torch.empty(ws_bytes, dtype=torch.uint8, device=q.device) if ws_bytes else None
    )
    output_kwargs = {}
    if out is not None and cutedsl_kda_supports_output_buffer():
        output_kwargs["out"] = out.reshape(v.shape)
    result, final_state = cutedsl_kda_forward(
        q,
        k,
        v,
        g_f32,
        A_log,
        dt_bias,
        beta,
        boundaries,
        state_in,
        scale=DEFAULT_SCALE,
        workspace=workspace,
        cu_seqlens_cpu=cu_seqlens_cpu,
        **output_kwargs,
    )
    result = result.view(batch, tokens, num_value_heads, value_dim)
    if destination is not None:
        if result.data_ptr() != destination.data_ptr():
            destination.copy_(result)
        result = destination
    return result, final_state
