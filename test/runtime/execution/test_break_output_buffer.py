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


"""Stable direct-write handoffs, including live KDA scan and graph padding."""

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid
from tokenspeed_kernel.ops.attention.cutedsl_kda import (
    cutedsl_kda_supports_output_buffer,
    is_cutedsl_kda_installed,
)

from tokenspeed.runtime.execution import breakable_cuda_graph as graph
from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend


@pytest.mark.parametrize(
    "kind", ["identity", "view", "prefix", "offset", "transpose", "dtype"]
)
def test_landing_skips_only_an_exact_leading_alias(monkeypatch, kind):
    dst = torch.empty(8, 8)
    result = {
        "identity": dst,
        "view": dst.view_as(dst),
        "prefix": dst[:3],
        "offset": dst[1:4],
        "transpose": dst.t(),
        "dtype": dst.view(torch.int32),
    }[kind]
    calls = []
    monkeypatch.setattr(
        torch.Tensor, "copy_", lambda target, source: calls.append((target, source))
    )
    graph._land_in(dst, result)
    assert bool(calls) == (kind in {"offset", "transpose", "dtype"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_direct_handoff_changes_inputs_buckets_and_preserves_padding(monkeypatch):
    live = {"rows": 8, "fail": False}
    seen = []

    class Op:
        @graph.break_point
        def forward(self, x):
            if live["fail"]:
                raise RuntimeError("test failure")
            dst = graph.current_break_output()
            seen.append(dst)
            x = x[: live["rows"]]
            if dst is None:
                return x * 2
            return torch.mul(x, 2, out=dst[: live["rows"]])

    op = Op()
    x = torch.randn(16, 8, device="cuda")
    captures = {}
    pool = None
    for bucket in (8, 16):
        live["rows"] = bucket

        def forward():
            h = x[:bucket] + 1
            for _ in range(3):
                h = op.forward(h) + 1
            return h

        for _ in range(3):
            forward()
        torch.cuda.synchronize()
        cap = graph.BreakableCapture(pool=pool, stream=None)
        with cap:
            out = forward()
        pool = cap.pool
        captures[bucket] = (cap, out)
    for bucket, rows in ((8, 3), (16, 11), (8, 8), (16, 1), (8, 1)):
        cap, out = captures[bucket]
        live["rows"] = rows
        x.normal_()
        expected = x[:rows] + 1
        for _ in range(3):
            expected = expected * 2 + 1

        def forbidden_copy(target, source, *args, **kwargs):
            raise AssertionError("direct handoff must not copy")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "copy_", forbidden_copy)
            cap.replay(valid_rows=rows)
        torch.testing.assert_close(out[:rows], expected, rtol=0, atol=0)
        assert torch.equal(out[rows:], torch.ones_like(out[rows:]))
        assert graph.current_break_output() is None
    live["fail"] = True
    with pytest.raises(RuntimeError, match="test failure"):
        captures[8][0].replay(valid_rows=1)
    assert graph.current_break_output() is None
    assert graph.current_valid_rows() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nested_break_cannot_borrow_its_parents_destination():
    x = torch.randn(8, 8, device="cuda")
    seen = []

    class Op:
        @graph.break_point
        def inner(self, value):
            assert graph.current_break_output() is None
            return value * 2

        @graph.break_point
        def outer(self, value):
            dst = graph.current_break_output()
            intermediate = self.inner(value)
            assert graph.current_break_output() is dst
            if dst is None:
                return intermediate + 3
            seen.append(dst.data_ptr())
            return torch.add(intermediate, 3, out=dst)

    op = Op()
    for _ in range(3):
        op.outer(x)
    torch.cuda.synchronize()
    cap = graph.BreakableCapture(pool=None, stream=None)
    with cap:
        out = op.outer(x + 1) + 1
    for _ in range(3):
        x.normal_()
        cap.replay(valid_rows=None)
        torch.testing.assert_close(out, ((x + 1) * 2 + 3) + 1, rtol=0, atol=0)
        assert graph.current_break_output() is None
    assert len(seen) == 3 and len(set(seen)) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kda_backend_direct_handoff_graph_replay(monkeypatch):
    if not is_cutedsl_kda_installed() or not cutedsl_kda_supports_output_buffer():
        pytest.skip("requires output-buffer-capable native KDA")
    torch.manual_seed(21)
    h, d = 12, 128
    q, k, v, g = [
        torch.randn(1, 768, h, d, device="cuda", dtype=torch.bfloat16) * 0.1
        for _ in range(4)
    ]
    beta = torch.randn(1, 768, h, device="cuda", dtype=torch.bfloat16)
    output_gate = torch.randn(768, h * d, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(d, device="cuda", dtype=torch.bfloat16)
    state = torch.randn(1, h, d, d, device="cuda") * 0.01
    a_log = torch.zeros(h, device="cuda")
    bias = torch.full((h, d), -4.0, device="cuda")
    backend = object.__new__(KdaAttnBackend)
    backend.kda_recurrent_layout = "v_major"
    backend.kda_backend = "cutedsl_kda"
    live = {"rows": 768}
    finals = []

    def set_rows(rows):
        live["rows"] = rows
        cpu = torch.tensor([0, rows], dtype=torch.int64)
        backend.forward_metadata = SimpleNamespace(
            query_start_loc_int64=cpu.to("cuda"),
            cu_extend_seq_lens_cpu=cpu,
        )

    class Op:
        @graph.break_point
        def forward(self, x):
            out, final = backend._prefill_scan(
                x,
                k[:, : x.shape[1]],
                v[:, : x.shape[1]],
                state,
                backend.forward_metadata.query_start_loc_int64,
                A_log=a_log,
                dt_bias=bias,
                a=None,
                b=None,
                g_raw=g[:, : x.shape[1]],
                f_a_out=None,
                f_b_weight=None,
                beta_raw=beta[:, : x.shape[1]],
                seq_len=x.shape[1],
                num_real_tokens=live["rows"],
                lower_bound=-5.0,
                cu_seqlens_cpu=backend.forward_metadata.cu_extend_seq_lens_cpu,
            )
            finals.append(final)
            return out

    op = Op()

    def forward(tokens):
        x = q[:, :tokens] + 0
        for _ in range(3):
            y = op.forward(x)
            y = rmsnorm_gated_sigmoid(
                y.reshape(tokens, h * d),
                output_gate[:tokens],
                weight,
                1e-6,
                h,
                d,
                enable_pdl=False,
            )
            x = y.view(1, tokens, h, d)
        return y

    captures, pool = {}, None
    for bucket in (112, 768):
        set_rows(bucket)
        for _ in range(3):
            forward(bucket)
        torch.cuda.synchronize()
        cap = graph.BreakableCapture(pool=pool, stream=None)
        with cap:
            out = forward(bucket)
        pool = cap.pool
        captures[bucket] = cap, out
    for bucket, rows in ((112, 100), (768, 768), (112, 16), (768, 768), (112, 1)):
        set_rows(rows)
        q.normal_(0, 0.1)
        state.normal_(0, 0.01)
        finals.clear()
        reference = forward(rows)
        reference_states = [s.clone() for s in finals]
        finals.clear()
        cap, out = captures[bucket]
        # No tensor copy is needed by the native scan or the landing path.
        with monkeypatch.context() as patch:
            patch.setattr(
                torch.Tensor,
                "copy_",
                lambda *a, **kw: pytest.fail("unexpected copy in direct KDA replay"),
            )
            cap.replay(valid_rows=rows)
        assert torch.equal(out[:rows], reference)
        assert torch.equal(out[rows:], torch.zeros_like(out[rows:]))
        assert all(
            torch.equal(a, b) for a, b in zip(finals, reference_states, strict=True)
        )
        assert graph.current_break_output() is None
