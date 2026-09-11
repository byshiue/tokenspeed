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


"""Caller-owned KDA outputs: native direct writes and legacy copy fallback."""

from types import SimpleNamespace

import pytest
import tokenspeed_kernel.ops.attention as attention
import tokenspeed_kernel.ops.attention.cutedsl_kda as cutedsl
import torch
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import SelectedKernel


def inputs(device, lengths):
    torch.manual_seed(123)
    t, h, d = sum(lengths), 12, 128
    q, k, v = [
        torch.randn(1, t, h, d, device=device, dtype=torch.bfloat16) * 0.1
        for _ in range(3)
    ]
    g = torch.randn_like(q)
    beta = torch.randn(1, t, h * 2, device=device, dtype=torch.bfloat16)[..., ::2]
    a_log = torch.randn(h, device=device) * 0.25
    bias = torch.randn(h, d, device=device) * 0.1 - 4.0
    state = torch.randn(len(lengths), h, d, d, device=device) * 0.05
    cpu = torch.tensor([0, *lengths], dtype=torch.int64).cumsum(0)
    return (q, k, v, g, beta, a_log, bias), state, cpu.to(device), cpu


def run(args, state, bounds, cpu, out):
    return attention.kda_paged_prefill(
        *args,
        initial_state=state,
        cu_seqlens=bounds,
        cu_seqlens_cpu=cpu,
        out=out,
        prefill_workspace=None,
        lower_bound=-5.0,
        override=None,
        solution="cutedsl_kda",
        recurrent_layout="v_major",
    )


@pytest.mark.parametrize("native_out", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_adapter_output_contract_and_legacy_fallback(monkeypatch, native_out, batch):
    args, state, bounds, cpu = inputs("cpu", [16] * batch)
    args = tuple(
        t.reshape(batch, 16, *t.shape[2:]) if i < 5 else t for i, t in enumerate(args)
    )
    target = torch.empty_like(args[2])
    seen = []

    def native(q, k, v, gate, a_log, bias, beta, boundaries, initial, **kwargs):
        dst = kwargs.get("out")
        seen.append(dst)
        if dst is None:
            return v + 1, initial
        torch.add(v, 1, out=dst)
        return dst, initial

    monkeypatch.setattr(cutedsl, "cutedsl_kda_check_config", lambda bound: None)
    monkeypatch.setattr(cutedsl, "cutedsl_kda_workspace_size", lambda *a, **k: 0)
    monkeypatch.setattr(
        cutedsl, "cutedsl_kda_supports_output_buffer", lambda: native_out
    )
    monkeypatch.setattr(cutedsl, "cutedsl_kda_forward", native)
    out, final = cutedsl.cutedsl_kda_chunk_prefill(
        *args,
        out=target,
        prefill_workspace=None,
        initial_state=state,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        lower_bound=-5.0,
        beta_is_logit=True,
    )
    assert out is target
    assert final is state
    assert torch.equal(out, args[2] + 1)
    assert (seen[0] is not None) == native_out
    if native_out:
        assert seen[0].data_ptr() == target.data_ptr()


@pytest.mark.parametrize("bad", ["shape", "dtype", "stride"])
def test_output_contract_rejects_invalid_destination(bad):
    args, state, bounds, cpu = inputs("cpu", [16])
    out = torch.empty_like(args[2])
    if bad == "shape":
        out = out[:, :-1]
    elif bad == "dtype":
        out = out.float()
    else:
        out = out.transpose(-1, -2)
    with pytest.raises(ValueError, match="KDA out"):
        run(args, state, bounds, cpu, out)


def test_dispatch_copy_fallback_without_output_trait(monkeypatch):
    args, state, bounds, cpu = inputs("cpu", [16])
    target = torch.empty_like(args[2])

    def kernel(**kwargs):
        assert "out" not in kwargs
        assert "prefill_workspace" not in kwargs
        return attention.KdaPrefillResult(args[2] + 1, state)

    selected = SelectedKernel("test_output_fallback", kernel)
    monkeypatch.setattr(attention, "select_kernel", lambda *a, **kw: selected)
    monkeypatch.setattr(
        KernelRegistry.get(), "get_by_name", lambda name: SimpleNamespace(traits={})
    )
    result = run(args, state, bounds, cpu, target)
    assert result.out is target
    assert result.final_state is state
    assert torch.equal(target, args[2] + 1)


@pytest.mark.parametrize("lengths", [[16], [100], [768], [868], [33, 67], [16] * 16])
@pytest.mark.parametrize("fresh", [False, True])
def test_native_direct_output_bitwise_and_guard_rows(lengths, fresh):
    if not torch.cuda.is_available() or not cutedsl.is_cutedsl_kda_installed():
        pytest.skip("requires native CUDA KDA")
    if not cutedsl.cutedsl_kda_supports_output_buffer():
        pytest.skip("requires output-buffer-capable native wrapper")
    args, state, bounds, cpu = inputs("cuda", lengths)
    if fresh:
        state.zero_()
    snapshots = [t.clone() for t in (*args, state)]
    storage = torch.full(
        (1, sum(lengths) + 12, 12, 128), torch.nan, device="cuda", dtype=torch.bfloat16
    )
    target = storage[:, : sum(lengths)]
    reference = run(args, state, bounds, cpu, None)
    actual = run(args, state, bounds, cpu, target)
    assert actual.out is target
    assert torch.equal(actual.out, reference.out)
    assert torch.equal(actual.final_state, reference.final_state)
    assert torch.isnan(storage[:, sum(lengths) :]).all()
    for actual_input, snapshot in zip((*args, state), snapshots, strict=True):
        assert torch.equal(actual_input, snapshot)


@pytest.mark.parametrize("tokens", [100, 768])
def test_native_legacy_wrapper_copy_fallback(monkeypatch, tokens):
    if not torch.cuda.is_available() or not cutedsl.is_cutedsl_kda_installed():
        pytest.skip("requires native CUDA KDA")
    args, state, bounds, cpu = inputs("cuda", [tokens])
    reference = run(args, state, bounds, cpu, None)
    target = torch.empty_like(args[2])
    monkeypatch.setattr(cutedsl, "cutedsl_kda_supports_output_buffer", lambda: False)
    actual = run(args, state, bounds, cpu, target)
    assert actual.out is target
    assert torch.equal(actual.out, reference.out)
    assert torch.equal(actual.final_state, reference.final_state)
