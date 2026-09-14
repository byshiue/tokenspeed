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

"""GPU numerical/graph gates for the unregistered buffered KDA prototype."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("requires a GPU", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda._triton.buffered import (  # noqa: E402
    buffered_commit,
    buffered_recurrent,
)


def direct(state, query, key, value, decay, beta):
    outputs = []
    state = state.clone()
    for token in range(query.shape[0]):
        state = state * decay[token, :, None, :]
        prediction = torch.einsum("hvk,hk->hv", state, key[token])
        correction = (value[token] - prediction) * beta[token, :, None]
        state = state + torch.einsum("hv,hk->hvk", correction, key[token])
        outputs.append(
            torch.einsum("hvk,hk->hv", state, query[token]) / query.shape[-1] ** 0.5
        )
    return torch.stack(outputs) if outputs else value[:0].clone(), state


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("extra_capacity", [0, 9])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_buffered_rounds_and_graph_match_sequential(width, extra_capacity, graph_mode):
    torch.manual_seed(93)
    batch, heads, dk, dv = 3, 12, 128, 128
    capacity = 2 * width + extra_capacity
    shape = (batch, width, heads, dk)
    query = torch.empty(shape, device="cuda")
    key = torch.empty_like(query)
    decay = torch.empty_like(query)
    value = torch.empty((batch, width, heads, dv), device="cuda")
    beta = torch.empty((batch, width, heads), device="cuda")
    state = torch.randn((batch, heads, dv, dk)) * 0.1
    checkpoint = state.cuda()
    history_key = torch.full((batch, capacity, heads, dk), float("nan"), device="cuda")
    history_decay = torch.full_like(history_key, float("nan"))
    history_correction = torch.full(
        (batch, capacity, heads, dv), float("nan"), device="cuda"
    )
    start = torch.zeros(batch, dtype=torch.int32, device="cuda")
    length = torch.zeros_like(start)
    valid = torch.zeros_like(start)
    accepted = torch.zeros_like(start)
    flushed = torch.zeros(batch, dtype=torch.bool, device="cuda")
    out = torch.full_like(value, float("nan"))

    def run():
        buffered_recurrent(
            query,
            key,
            value,
            decay,
            beta,
            checkpoint,
            history_key,
            history_correction,
            history_decay,
            start,
            length,
            valid,
            flushed,
            out,
        )
        buffered_commit(start, length, valid, accepted, flushed, capacity)

    # Compile on a non-default stream before graph capture, with all rows idle.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    if graph_mode:
        with torch.cuda.graph(graph):
            run()

    histories = [0] * batch
    starts = [0] * batch
    saw_flush = saw_no_flush = False
    for step in range(32):
        q = F.normalize(torch.randn(shape), dim=-1)
        k = F.normalize(torch.randn(shape), dim=-1)
        v = torch.randn(value.shape) * 0.2
        d = torch.exp(-torch.rand(shape) * 0.1)
        b = torch.rand(beta.shape)
        widths = [width, 1 + step % width, 0 if step % 3 else width]
        accepts = [width, step % (widths[1] + 1), widths[2]]
        before_checkpoint = checkpoint.cpu()
        for dst, src in ((query, q), (key, k), (value, v), (decay, d), (beta, b)):
            dst.copy_(src)
        valid.copy_(torch.tensor(widths, dtype=torch.int32))
        accepted.copy_(torch.tensor(accepts, dtype=torch.int32))
        out.fill_(float("nan"))
        if graph_mode:
            graph.replay()
        else:
            run()
        got = out.cpu()
        flags = flushed.cpu().tolist()
        next_checkpoint = checkpoint.cpu()
        hk, hu, hd = history_key.cpu(), history_correction.cpu(), history_decay.cpu()
        for row in range(batch):
            n, a = widths[row], accepts[row]
            expected_flush = n > 0 and histories[row] + 2 * width > capacity
            assert flags[row] == expected_flush
            expected, _ = direct(
                state[row], q[row, :n], k[row, :n], v[row, :n], d[row, :n], b[row, :n]
            )
            torch.testing.assert_close(got[row, :n], expected, atol=2e-5, rtol=2e-4)
            assert torch.isnan(got[row, n:]).all()
            if expected_flush:
                saw_flush = True
                torch.testing.assert_close(
                    next_checkpoint[row], state[row], atol=2e-5, rtol=2e-4
                )
                starts[row] = (starts[row] + histories[row]) % capacity
                histories[row] = 0
            else:
                saw_no_flush = True
                assert torch.equal(next_checkpoint[row], before_checkpoint[row])
            _, state[row] = direct(
                state[row], q[row, :a], k[row, :a], v[row, :a], d[row, :a], b[row, :a]
            )
            histories[row] += a
            materialized = next_checkpoint[row].clone()
            for offset in range(histories[row]):
                slot = (starts[row] + offset) % capacity
                materialized = (
                    materialized * hd[row, slot, :, None, :]
                    + hu[row, slot, :, :, None] * hk[row, slot, :, None, :]
                )
            torch.testing.assert_close(materialized, state[row], atol=2e-5, rtol=2e-4)
        assert length.cpu().tolist() == histories
        assert start.cpu().tolist() == starts
    assert saw_flush and saw_no_flush
