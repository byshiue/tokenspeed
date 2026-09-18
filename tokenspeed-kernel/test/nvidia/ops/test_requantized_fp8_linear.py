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

import pytest
import torch
from tokenspeed_kernel import fp8_linear, prepare_requantized_deep_gemm_fp8_linear
from tokenspeed_kernel.ops.moe.deep_gemm.ue8m0 import (
    is_ue8m0,
    per_token_group_quant_fp8_ue8m0,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_blackwell_plus, reason="Requires Blackwell DeepGEMM"
)


@pytest.mark.parametrize("m", [1, 32, 128, 512])
def test_requantized_linear_and_graph(m):
    torch.manual_seed(42)
    n, k = 256, 512
    weight = (torch.randn(n, k, device="cuda") * 32).to(torch.float8_e4m3fn)
    scales = torch.full((n // 128, k // 128), 0.013, device="cuda")
    original = weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(
        128, 1
    )
    plan = prepare_requantized_deep_gemm_fp8_linear(weight, scales, (128, 128))
    assert is_ue8m0(scales)
    converted = weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(
        128, 1
    )
    assert (converted - original).norm() / original.norm() < 0.06
    saved_weight, saved_scales = weight.clone(), scales.clone()
    prepare_requantized_deep_gemm_fp8_linear(weight, scales, (128, 128))
    assert torch.equal(weight.float(), saved_weight.float())
    assert torch.equal(scales, saved_scales)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    def run():
        return fp8_linear(
            plan,
            x,
            weight,
            scales,
            input_scales=None,
            bias=None,
            out_dtype=torch.bfloat16,
        )

    for _ in range(3):
        eager = run()
    reference = x.float() @ converted.T
    assert (eager.float() - reference).norm() / reference.norm() < 0.06
    # Separate GEMM correctness from the deliberately changed quantization.
    quantized, input_scales = per_token_group_quant_fp8_ue8m0(x, 128)
    quantized_reference = (
        quantized.float() * input_scales.repeat_interleave(128, dim=1)
    ) @ converted.T
    assert (
        eager.float() - quantized_reference
    ).norm() / quantized_reference.norm() < 0.005
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for _ in range(3):
        x.normal_()
        expected = run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_requantization_rejects_invalid_scales_and_alignment():
    weight = torch.zeros((128, 256), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.ones((1, 2), device="cuda")
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        scales.fill_(invalid)
        with pytest.raises(ValueError, match="positive and finite"):
            prepare_requantized_deep_gemm_fp8_linear(weight, scales, (128, 128))
    scales.fill_(1)
    with pytest.raises(ValueError, match="aligned"):
        prepare_requantized_deep_gemm_fp8_linear(weight, scales, (64, 128))
    plan = prepare_requantized_deep_gemm_fp8_linear(weight, scales, (128, 128))
    x = torch.ones((1, 256), device="cuda", dtype=torch.bfloat16)
    output = fp8_linear(
        plan, x, weight, scales, input_scales=None, bias=None, out_dtype=torch.bfloat16
    )
    assert torch.count_nonzero(output) == 0
