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

"""Original-scale TRT-LLM CuTe-DSL FP8 correctness and CUDA graph coverage."""

import pytest
import torch
from tokenspeed_kernel import fp8_linear, prepare_trtllm_cutedsl_fp8_linear
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    flashinfer_fp8_blockscale_quantize_prepacked,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_blackwell, reason="Requires Blackwell CuTe-DSL"
)


@pytest.mark.parametrize("m", [1, 32, 64, 128, 256, 512])
def test_original_scales_and_graph(m):
    torch.manual_seed(42)
    n, k = 256, 512
    weight = (torch.randn(n, k, device="cuda") * 32).to(torch.float8_e4m3fn)
    scales = torch.rand(n // 128, k // 128, device="cuda") * 0.01 + 0.01
    original_weight, original_scales = weight.clone(), scales.clone()
    plan = prepare_trtllm_cutedsl_fp8_linear(weight, scales, (128, 128))
    assert torch.equal(weight.view(torch.uint8), original_weight.view(torch.uint8))
    assert torch.equal(scales, original_scales)
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

    eager = run()
    q, s = flashinfer_fp8_blockscale_quantize_prepacked(x, 128)
    dq = q[:m].float() * s[:, :m].T.repeat_interleave(128, dim=1)
    dw = weight.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
    reference = dq @ dw.T
    assert (eager.float() - reference).norm() / reference.norm() < 0.005

    # Capture after preparation, without warming this particular M again.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for _ in range(3):
        x.normal_()
        expected = run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_alignment_and_zero_weights():
    weight = torch.zeros((128, 256), device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.full((1, 2), 0.013, device="cuda")
    with pytest.raises(ValueError, match="aligned"):
        prepare_trtllm_cutedsl_fp8_linear(weight, scales, (64, 128))
    plan = prepare_trtllm_cutedsl_fp8_linear(weight, scales, (128, 128))
    x = torch.ones((1, 256), device="cuda", dtype=torch.bfloat16)
    output = fp8_linear(
        plan, x, weight, scales, input_scales=None, bias=None, out_dtype=torch.bfloat16
    )
    assert torch.count_nonzero(output) == 0
