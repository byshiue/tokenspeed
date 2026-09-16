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


"""Verify arithmetic is defined by logical keys, not compiler/launch layout."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)

from tokenspeed_kernel._triton import gl, gluon, tl, triton  # noqa: E402
from tokenspeed_kernel.ops.attention.kda._triton.buffered_gate import (  # noqa: E402
    buffered_history_gate,
)
from tokenspeed_kernel.ops.attention.kda._triton.verify_math import (  # noqa: E402
    verify_normalize,
    verify_recurrence,
)


@triton.jit
def _triton_step(S, Q, K, V, D, B, NEXT, OUT, CORRECTION, BV: tl.constexpr):
    kk = tl.arange(0, 128)
    vv = tl.program_id(0) * BV + tl.arange(0, BV)
    state = tl.load(S + vv[:, None] * 128 + kk[None, :])
    q, k = verify_normalize(tl.load(Q + kk), tl.load(K + kk), 128**-0.5, tl)
    state, out, correction = verify_recurrence(
        state, q, k, tl.load(V + vv), tl.load(D + kk), tl.load(B), tl
    )
    tl.store(NEXT + vv[:, None] * 128 + kk[None, :], state)
    tl.store(OUT + vv, out)
    tl.store(CORRECTION + vv, correction)


@gluon.jit
def _gluon_step(S, Q, K, V, D, B, NEXT, OUT, CORRECTION, BV: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [1, 32], [gl.num_warps(), 1], [1, 0]
    )
    kk = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    vv = gl.program_id(0) * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    state = gl.load(S + vv[:, None] * 128 + kk[None, :])
    q, k = verify_normalize(gl.load(Q + kk), gl.load(K + kk), 128**-0.5, gl)
    state, out, correction = verify_recurrence(
        state, q, k, gl.load(V + vv), gl.load(D + kk), gl.load(B), gl
    )
    gl.store(NEXT + vv[:, None] * 128 + kk[None, :], state)
    gl.store(OUT + vv, out)
    gl.store(CORRECTION + vv, correction)


def _pair_sum(value):
    while value.shape[-1] > 1:
        value = value[..., 0::2] + value[..., 1::2]
    return value.squeeze(-1)


@pytest.mark.parametrize("rows", [4, 16])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_history_gate_matches_reference(rows, lower_bound):
    torch.manual_seed(6902)
    fa = torch.randn((rows, 128), device="cuda", dtype=torch.bfloat16) * 0.1
    fb = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16) * 0.1
    a_log = torch.randn(2, device="cuda")
    bias = torch.randn(256, device="cuda")
    out = torch.empty((rows, 256), device="cuda")
    buffered_history_gate(
        fa,
        fb,
        a_log,
        bias,
        out,
        num_heads=2,
        head_dim=128,
        local_layers=69,
        lower_bound=lower_bound,
    )
    gate = (fa.float() @ fb.float().T + bias).view(rows, 2, 128)
    scale = a_log.exp()[None, :, None]
    expected = (
        -scale * torch.nn.functional.softplus(gate)
        if lower_bound is None
        else lower_bound * torch.sigmoid(scale * gate)
    )
    torch.testing.assert_close(out.view_as(expected), expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("tile,warps", [(8, 1), (8, 4), (16, 1), (32, 4)])
@pytest.mark.parametrize("fusion", [False, True])
def test_shared_verify_matches_explicit_reference(tile, warps, fusion):
    torch.manual_seed(6901)
    state = torch.randn((32, 128), device="cuda")
    q, k, decay = torch.randn((3, 128), device="cuda").unbind(0)
    decay = torch.sigmoid(decay)
    value = torch.randn(32, device="cuda")
    beta = torch.tensor(0.625, device="cuda")
    q_ref = (q / (_pair_sum(q * q) + 1e-6).sqrt()) * 128**-0.5
    k_ref = k / (_pair_sum(k * k) + 1e-6).sqrt()
    decayed = state * decay
    correction = (value - _pair_sum(decayed * k_ref)) * beta
    # FP64 forms the product/add before one FP32 rounding, modeling FMA.
    next_state = (
        correction.double()[:, None] * k_ref.double() + decayed.double()
    ).float()
    expected = (next_state, _pair_sum(next_state * q_ref), correction)
    for kernel in (_triton_step, _gluon_step):
        actual = tuple(torch.empty_like(tensor) for tensor in expected)
        kernel[(32 // tile,)](
            state,
            q,
            k,
            value,
            decay,
            beta,
            *actual,
            BV=tile,
            num_warps=warps,
            enable_fp_fusion=fusion,
        )
        for got, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, reference, atol=0, rtol=0)
