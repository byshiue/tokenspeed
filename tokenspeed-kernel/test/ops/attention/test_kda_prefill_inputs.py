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

"""Exact-value, strided-input and replay tests for KDA input packing."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.kda_prefill_inputs import (
    prepare_kda_prefill_gate_beta,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    return request.param


def _inputs(device, gate_dtype, beta_dtype, strided, tokens):
    if strided:
        gate = torch.randn(2, 12, tokens * 2, 256, device=device, dtype=gate_dtype)
        gate = gate[:, ::2, ::2, 1::2].transpose(1, 2)
        beta = torch.randn(2, tokens, 200, device=device, dtype=beta_dtype)[
            :, :, 3:15:2
        ]
    else:
        gate = torch.randn(2, tokens, 6, 128, device=device, dtype=gate_dtype)
        beta = torch.randn(2, tokens, 6, device=device, dtype=beta_dtype)
    return gate, beta


def _assert_exact(actual, expected):
    assert actual.is_contiguous()
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    # In addition to value equality, preserve signed zeros and all finite bits.
    integer = torch.int32 if actual.dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(integer), expected.view(integer))


@pytest.mark.parametrize("gate_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("beta_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_pack_exact(device, gate_dtype, beta_dtype, strided):
    gate, beta = _inputs(device, gate_dtype, beta_dtype, strided, 100)
    gate_before, beta_before = gate.clone(), beta.clone()
    actual_g, actual_b = prepare_kda_prefill_gate_beta(gate, beta)
    _assert_exact(actual_g, gate.float().contiguous())
    _assert_exact(actual_b, beta.contiguous())
    assert torch.equal(gate, gate_before)
    assert torch.equal(beta, beta_before)
    if gate.dtype == torch.float32 and gate.is_contiguous():
        assert actual_g is gate
    if beta.is_contiguous():
        assert actual_b is beta


@pytest.mark.parametrize("tokens", [0, 1, 768, 868])
def test_pack_token_shapes(device, tokens):
    gate, beta = _inputs(device, torch.bfloat16, torch.bfloat16, True, tokens)
    actual_g, actual_b = prepare_kda_prefill_gate_beta(gate, beta)
    _assert_exact(actual_g, gate.float().contiguous())
    _assert_exact(actual_b, beta.contiguous())


def test_pack_nonfinite_and_signed_zero(device):
    values = torch.tensor(
        [0.0, -0.0, float("inf"), -float("inf"), float("nan")],
        dtype=torch.bfloat16,
        device=device,
    )
    gate = values.view(1, 1, 1, 5).expand(1, 5, 1, 5)
    beta = values.view(1, 5, 1)
    actual_g, actual_b = prepare_kda_prefill_gate_beta(gate, beta)
    torch.testing.assert_close(actual_g, gate.float(), rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(actual_b, beta, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(actual_g[..., :2]), torch.signbit(gate[..., :2]))


def test_pack_rejects_invalid_inputs(device):
    gate, beta = _inputs(device, torch.bfloat16, torch.bfloat16, False, 2)
    with pytest.raises(ValueError, match="gate must"):
        prepare_kda_prefill_gate_beta(gate, beta[..., :1])
    with pytest.raises(ValueError, match="float16"):
        prepare_kda_prefill_gate_beta(gate.to(torch.int32), beta)
    with pytest.raises(ValueError, match="gate must"):
        prepare_kda_prefill_gate_beta(gate[..., :0], beta)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_pack_cuda_graph_replays_changed_inputs():
    gate, beta = _inputs("cuda", torch.bfloat16, torch.bfloat16, True, 768)
    for _ in range(3):
        prepare_kda_prefill_gate_beta(gate, beta)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_g, actual_b = prepare_kda_prefill_gate_beta(gate, beta)
    addresses = actual_g.data_ptr(), actual_b.data_ptr()
    for _ in range(3):
        gate.copy_(torch.randn_like(gate))
        beta.copy_(torch.randn_like(beta))
        graph.replay()
        _assert_exact(actual_g, gate.float().contiguous())
        _assert_exact(actual_b, beta.contiguous())
        assert addresses == (actual_g.data_ptr(), actual_b.data_ptr())
