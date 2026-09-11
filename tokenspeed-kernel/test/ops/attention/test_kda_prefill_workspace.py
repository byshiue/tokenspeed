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

"""Forward scratch lifetime and bitwise native-scan equivalence."""

import pytest
import tokenspeed_kernel.ops.attention.cutedsl_kda as cutedsl
import torch
from tokenspeed_kernel.ops.attention import kda_paged_prefill
from tokenspeed_kernel.ops.attention.prefill_workspace import KdaPrefillWorkspace


@pytest.mark.parametrize("size", [0, 4096])
def test_query_once_per_head_count(size):
    bounds = torch.tensor([0, 768], dtype=torch.int64)
    workspace = KdaPrefillWorkspace(bounds, bounds)
    calls = []

    def query(boundaries, heads, *, cu_seqlens_cpu):
        calls.append((boundaries, heads, cu_seqlens_cpu))
        return size

    scratch = workspace.get(bounds, 12, query)
    for _ in range(68):
        assert workspace.get(bounds, 12, query) is scratch
    assert len(calls) == 1
    assert calls[0][2] == (0, 768)
    workspace.get(bounds, 24, query)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="different forward"):
        workspace.get(bounds.clone(), 12, query)


def test_new_forward_with_reused_inference_boundary_storage():
    calls = []

    def query(boundaries, heads, *, cu_seqlens_cpu):
        calls.append(cu_seqlens_cpu)
        return cu_seqlens_cpu[-1]

    with torch.inference_mode():
        bounds = torch.tensor([0, 768], dtype=torch.int64)
        first = KdaPrefillWorkspace(bounds, bounds)
        scratch = first.get(bounds, 12, query)
        # The old forward is over. Same tensor identity has new contents,
        # without a version counter: only a fresh owner is safe.
        bounds[1] = 100
        second = KdaPrefillWorkspace(bounds, bounds)
        next_scratch = second.get(bounds, 12, query)
    assert calls == [(0, 768), (0, 100)]
    assert scratch.numel() == 768 and next_scratch.numel() == 100
    assert scratch.data_ptr() != next_scratch.data_ptr()


@pytest.mark.parametrize("lengths", [[100], [768], [868], [33, 67], [16] * 16])
@pytest.mark.parametrize("prepared", [False, True])
def test_native_reuse_matches_standalone_bitwise(monkeypatch, lengths, prepared):
    if not torch.cuda.is_available() or not cutedsl.is_cutedsl_kda_installed():
        pytest.skip("requires native CUDA KDA")
    # Keep covering the old-wheel scratch-only fallback even with a new wheel.
    if prepared and not cutedsl.cutedsl_kda_supports_prefill_plan():
        pytest.skip("requires prepared-plan-capable native wrapper")
    monkeypatch.setattr(cutedsl, "cutedsl_kda_supports_prefill_plan", lambda: prepared)
    torch.manual_seed(71)
    tokens, heads, dim = sum(lengths), 12, 128
    q, k, v, gate = [
        torch.randn(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.1
        for _ in range(4)
    ]
    beta = torch.randn(1, tokens, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda") * 0.25
    bias = torch.randn(heads, dim, device="cuda") * 0.1 - 4
    state = torch.randn(len(lengths), heads, dim, dim, device="cuda") * 0.05
    cpu = torch.tensor([0, *lengths], dtype=torch.int64).cumsum(0)
    bounds = cpu.cuda()
    owner = KdaPrefillWorkspace(bounds, cpu)
    entry = "cutedsl_kda_prepare_prefill" if prepared else "cutedsl_kda_workspace_size"
    native_query = getattr(cutedsl, entry)
    calls = []

    def query(*args, **kwargs):
        calls.append(1)
        return native_query(*args, **kwargs)

    def run(workspace, initial):
        return kda_paged_prefill(
            q,
            k,
            v,
            gate,
            beta,
            a_log,
            bias,
            initial_state=initial,
            cu_seqlens=bounds,
            cu_seqlens_cpu=cpu,
            out=torch.empty_like(v),
            prefill_workspace=workspace,
            lower_bound=-5.0,
            override=None,
            solution="cutedsl_kda",
            recurrent_layout="v_major",
        )

    references = [run(None, initial) for initial in (state, state * 0.5, state * 0)]
    monkeypatch.setattr(cutedsl, entry, query)
    for initial, reference in zip((state, state * 0.5, state * 0), references):
        snapshot = initial.clone()
        actual = run(owner, initial)
        assert torch.equal(actual.out, reference.out)
        assert torch.equal(actual.final_state, reference.final_state)
        assert torch.equal(initial, snapshot)
    assert len(calls) == 1


def test_stream_isolation_and_capture_does_not_escape():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cpu = torch.tensor([0, 100], dtype=torch.int64)
    bounds = cpu.cuda()
    owner = KdaPrefillWorkspace(bounds, cpu)

    def query(boundaries, heads, *, cu_seqlens_cpu):
        return 4096

    first = owner.get(bounds, 12, query)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        other = owner.get(bounds, 12, query)
    assert first.data_ptr() != other.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = owner.get(bounds, 12, query)
        captured.fill_(7)
    with torch.cuda.stream(stream):
        assert owner.get(bounds, 12, query) is other
    assert captured.data_ptr() not in (first.data_ptr(), other.data_ptr())
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(captured == 7)


def test_opaque_plan_reuse_and_capture_bypass():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cpu = torch.tensor([0, 100], dtype=torch.int64)
    bounds = cpu.cuda()
    owner = KdaPrefillWorkspace(bounds, cpu)
    calls = []

    def prepare(boundaries, heads, *, cu_seqlens_cpu):
        calls.append(cu_seqlens_cpu)
        return object()

    first = owner.get_plan(bounds, 12, prepare)
    for _ in range(68):
        assert owner.get_plan(bounds, 12, prepare) is first
    assert len(calls) == 1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        other = owner.get_plan(bounds, 12, prepare)
        assert other is not first
    assert len(calls) == 2
    graph = torch.cuda.CUDAGraph()
    dummy = torch.empty(1, device="cuda")
    with torch.cuda.graph(graph, stream=stream):
        assert owner.get_plan(bounds, 12, prepare) is None
        dummy.fill_(1)
    assert len(calls) == 2
    with torch.cuda.stream(stream):
        assert owner.get_plan(bounds, 12, prepare) is other
    graph.replay()
    torch.cuda.synchronize()
