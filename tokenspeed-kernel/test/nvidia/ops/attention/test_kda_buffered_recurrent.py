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


"""Paged recurrence/commit gates; no serving scheduler or model is dispatched."""

from __future__ import annotations

import math
import random

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("requires a GPU", allow_module_level=True)

from tokenspeed_kernel._triton import tl, triton  # noqa: E402
from tokenspeed_kernel.ops.attention.kda._triton.buffered import (  # noqa: E402
    _input_offset,
    buffered_recurrent,
    validate_recurrent_blocks,
)
from tokenspeed_kernel.ops.attention.kda._triton.buffered_metadata import (  # noqa: E402
    commit_positions,
    prepare_positions,
)
from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (  # noqa: E402
    fused_recurrent_kda_pool,
)


def direct(state, query, key, value, decay, beta):
    outputs = []
    state = state.clone()
    for token in range(query.shape[0]):
        state = state * decay[token, :, None, :]
        correction = (
            value[token] - torch.einsum("hvk,hk->hv", state, key[token])
        ) * beta[token, :, None]
        state = state + torch.einsum("hv,hk->hvk", correction, key[token])
        outputs.append(
            torch.einsum("hvk,hk->hv", state, query[token]) / query.shape[-1] ** 0.5
        )
    return torch.stack(outputs) if outputs else value[:0].clone(), state


@triton.jit
def _wide_input_offsets(out):
    token = tl.arange(0, 4)
    zero = tl.full((4,), 0, tl.int64)
    offsets = _input_offset(
        zero, token, zero, token.to(tl.int64), (1, 2**30 + 3, 128, 1)
    )
    tl.store(out + token, offsets)


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("extra_capacity", [0, 9, 56])
@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("input_kind", ["prepared", "softplus", "bounded"])
def test_buffered_rounds_and_graph_match_sequential(
    width, extra_capacity, graph_mode, input_kind
):
    torch.manual_seed(93)
    rng = random.Random(93)
    requests, batch, heads, dk, dv = 3, 4, 12, 128, 128
    capacity = 2 * width + extra_capacity
    rows = 3 if extra_capacity else 8
    grain, context = 16 if width == 4 else 128, 1024 if extra_capacity == 56 else 512
    columns, state_columns = math.ceil(context / rows), math.ceil(context / grain)
    count = 1 + requests * (math.ceil(capacity / rows) + 4)
    # Different strides from dense tensors, including padded channel dimensions.
    hk = torch.full((count, rows, heads, dk + 3), float("nan"), device="cuda")[..., :dk]
    hu = torch.full((count, rows, heads, dv + 5), float("nan"), device="cuda")[..., :dv]
    hd = torch.full((count, rows, heads, dk + 7), float("nan"), device="cuda")[..., :dk]
    stamps = torch.zeros((count, rows, 3), dtype=torch.int64, device="cuda")[:, :, 1]
    state_count = 1 + requests * (state_columns + 1)
    pool = torch.full((state_count, heads, dv, dk + 1), float("nan"), device="cuda")[
        ..., :dk
    ]
    state_tables = torch.arange(
        1, 1 + requests * state_columns, dtype=torch.int32
    ).view(requests, state_columns)
    spare = list(range(1 + requests * state_columns, state_count))
    histories = torch.full((requests, columns), -1, dtype=torch.int32)
    free = list(range(1, count))
    rng.shuffle(free)
    ends = [125, 127, 0]
    checkpoints = list(ends)
    states = torch.randn(requests, heads, dv, dk) * 0.1
    states[2].zero_()  # c=0 uses implicit zero state, never null-block contents.
    for req in range(2):
        pool[state_tables[req, (ends[req] - 1) // grain]].copy_(states[req])
    native = input_kind != "prepared"
    dtype = torch.bfloat16 if native else torch.float32
    if native:
        # Zero-copy packed conv views with non-dense token/head strides.
        packed = torch.empty(
            (batch, width, 3, heads, dk + 3), device="cuda", dtype=dtype
        )
        q, k, v = packed[..., :dk].unbind(2)
    else:
        q = torch.empty((batch, width, heads, dk), device="cuda")
        k = torch.empty_like(q)
        v = torch.empty((batch, width, heads, dv), device="cuda")
    gate_dtype = torch.float32 if extra_capacity == 9 else dtype
    d = torch.empty((batch, width, heads, dk), device="cuda", dtype=gate_dtype)
    beta = torch.empty((batch, width, heads), device="cuda", dtype=gate_dtype)
    out = torch.empty((batch, width, heads, dv + 7), device="cuda", dtype=dtype)[
        ..., :dv
    ]
    a_cpu = torch.full((heads,), -9.0 if extra_capacity == 56 else -1.0)
    bias_cpu = torch.randn(heads * dk) * 0.2
    a_log, dt_bias = a_cpu.cuda(), bias_cpu.cuda()
    lower_bound = (
        (-1e-4 if extra_capacity == 56 else -0.3) if input_kind == "bounded" else None
    )
    ht = torch.full((batch, columns + 2), -1, dtype=torch.int32, device="cuda")[
        :, :columns
    ]
    st = torch.full((batch, state_columns + 2), -1, dtype=torch.int32, device="cuda")[
        :, :state_columns
    ]
    end = torch.zeros(batch, dtype=torch.int32, device="cuda")
    valid, accepted, length = (
        torch.zeros_like(end),
        torch.zeros_like(end),
        torch.empty_like(end),
    )
    cp = torch.empty(batch, dtype=torch.int64, device="cuda")
    flush = torch.empty(batch, dtype=torch.bool, device="cuda")
    ok = torch.empty_like(flush)
    endpoint_materialized = torch.zeros_like(flush)

    def run():
        prepare_positions(
            stamps,
            ht,
            end,
            valid,
            cp,
            length,
            flush,
            ok,
            capacity=capacity,
            max_window=width,
        )
        validate_recurrent_blocks(
            ht,
            st,
            end,
            cp,
            length,
            valid,
            flush,
            ok,
            history_blocks=hk.shape[0],
            state_blocks=pool.shape[0],
            history_block_tokens=hk.shape[1],
            state_block_tokens=grain,
            capacity=capacity,
            max_window=width,
        )
        buffered_recurrent(
            q,
            k,
            v,
            d,
            beta,
            pool,
            hk,
            hu,
            hd,
            ht,
            st,
            end,
            cp,
            length,
            valid,
            flush,
            ok,
            out,
            capacity=capacity,
            state_block_tokens=grain,
            transform_inputs=native,
            A_log=a_log if native else None,
            dt_bias=dt_bias if native else None,
            lower_bound=lower_bound,
        )
        commit_positions(
            (stamps,),
            ht,
            end,
            valid,
            accepted,
            cp,
            length,
            flush,
            ok,
            endpoint_materialized,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    if graph_mode:
        with torch.cuda.graph(graph):
            run()
    pointers = [t.data_ptr() for t in (pool, hk, hu, hd, stamps, ht, st, cp, out)]
    saw_flush = saw_no_flush = saw_zero_accept_flush = saw_block_reuse = False
    for step in range(256 if extra_capacity == 56 else 64):
        if step == 32:
            # Simulated cancel/reuse after readers complete, seeded at a new exact endpoint.
            req = 2
            for block in histories[req][histories[req] > 0].tolist():
                free.append(block)
            histories[req].fill_(-1)
            ends[req] = checkpoints[req] = 8
            states[req] = torch.randn_like(states[req]) * 0.1
            pool[state_tables[req, 0]].copy_(states[req])
        for req in range(requests):
            # A test allocator reuses physical blocks, but preserves absolute
            # table columns. This does not simulate scheduler overlap itself.
            first = max(0, min(checkpoints[req], ends[req] - 1)) // rows
            last = (ends[req] + width - 1) // rows
            for col in range(first):
                block = int(histories[req, col])
                if block > 0:
                    free.append(block)
                    histories[req, col] = -1
                    saw_block_reuse = True
            for col in range(first, last + 1):
                if histories[req, col] <= 0:
                    block = free.pop()
                    histories[req, col] = block
                    stamps[block].zero_()
                    for field in (hk, hu, hd):
                        field[block].fill_(float("nan"))
            if step and step % 11 == 0 and checkpoints[req] > 0:
                # A new physical address between forwards must be read through
                # the current table, never remembered as a request pointer.
                col = (checkpoints[req] - 1) // grain
                previous = int(state_tables[req, col])
                pool[spare[req]].copy_(pool[previous])
                state_tables[req, col], spare[req] = spare[req], previous
                pool[previous].fill_(float("nan"))
        order = [(step + i) % requests for i in range(requests)]
        widths = [width, 1 + step % width, 0 if step % 3 else width, 0]
        accepts = [widths[0], step % (widths[1] + 1), widths[2], 0]
        flags = [
            widths[row] > 0 and ends[req] - checkpoints[req] + 2 * width > capacity
            for row, req in enumerate(order)
        ]
        if not saw_zero_accept_flush:
            for row, flag in enumerate(flags):
                if flag:
                    accepts[row] = 0
        end.copy_(
            torch.tensor([ends[req] for req in order] + [2**31 - 1], dtype=torch.int32)
        )
        valid.copy_(torch.tensor(widths, dtype=torch.int32))
        accepted.copy_(torch.tensor(accepts, dtype=torch.int32))
        ht[:requests].copy_(histories[order])
        st[:requests].copy_(state_tables[order])
        if native:
            raw = (
                torch.randn(q.shape).to(dtype),
                torch.randn(k.shape).to(dtype),
                (torch.randn(v.shape) * 0.2).to(dtype),
                torch.randn(d.shape).to(gate_dtype),
                torch.randn(beta.shape).to(gate_dtype),
            )
            raw[0][0, 0].zero_()  # The normalization epsilon must handle zero Q.
            raw[3][..., :3] = torch.tensor([-100.0, 20.0, 100.0], dtype=dtype)
            q_cpu, k_cpu, v_cpu, gate_cpu, beta_cpu = (x.float() for x in raw)
            gate_cpu = gate_cpu + bias_cpu.view(heads, dk)
            log_decay = (
                lower_bound * torch.sigmoid(a_cpu.exp()[:, None] * gate_cpu)
                if lower_bound is not None
                else -a_cpu.exp()[:, None] * F.softplus(gate_cpu, threshold=20)
            )
            inputs = (
                q_cpu / torch.sqrt(q_cpu.square().sum(-1, keepdim=True) + 1e-6),
                k_cpu / torch.sqrt(k_cpu.square().sum(-1, keepdim=True) + 1e-6),
                v_cpu,
                log_decay.exp(),
                beta_cpu.sigmoid(),
            )
        else:
            inputs = (
                F.normalize(torch.randn(q.shape), dim=-1),
                F.normalize(torch.randn(k.shape), dim=-1),
                torch.randn(v.shape) * 0.2,
                torch.exp(
                    -torch.rand(d.shape) * (1e-4 if extra_capacity == 56 else 0.1)
                ),
                torch.rand(beta.shape),
            )
            raw = inputs
        if extra_capacity == 56 and not native:
            # Weak decay retains rounding error across tiles and flushes. Include
            # identity and zero decay: suffix reconstruction must not divide by D.
            inputs[3][..., 0] = 1
            if step % 13 == 0:
                inputs[3][..., 1] = 0
        for dest, src in zip((q, k, v, d, beta), raw, strict=True):
            dest.copy_(src)
        before = pool.clone()
        out.fill_(float("nan"))
        if graph_mode:
            graph.replay()
        else:
            run()
        assert ok.all()
        assert flush.tolist() == flags + [False]
        changed_blocks = set()
        for row, req in enumerate(order):
            n, a, e, c = widths[row], accepts[row], ends[req], checkpoints[req]
            expected, _ = direct(states[req], *(x[row, :n] for x in inputs))
            # Native outputs round once to BF16. Allow its half-ULP rounding
            # bound in addition to the established FP32 computation tolerance;
            # history and materialized-state checks retain the FP32 gate below.
            output_rtol = 2e-4 + (torch.finfo(torch.bfloat16).eps / 2 if native else 0)
            torch.testing.assert_close(
                out[row, :n].float().cpu(), expected, atol=2e-5, rtol=output_rtol
            )
            if native and step == 0 and n:
                # Pin native transform semantics to the existing GPU recurrence
                # as well as the independent CPU equations (before any history).
                original_pool = states[req].unsqueeze(0).cuda()
                indices = torch.zeros(1, dtype=torch.int32, device="cuda")
                original = fused_recurrent_kda_pool(
                    *(x[row : row + 1, :n].contiguous() for x in (q, k, v, d, beta)),
                    a_log,
                    dt_bias,
                    original_pool,
                    indices,
                    indices,
                    scale=dk**-0.5,
                    cu_seqlens=None,
                    lower_bound=lower_bound,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                )
                torch.testing.assert_close(
                    original[0].float().cpu(), expected, atol=2e-5, rtol=output_rtol
                )
                _, expected_state = direct(states[req], *(x[row, :n] for x in inputs))
                torch.testing.assert_close(
                    original_pool[0].cpu(), expected_state, atol=2e-5, rtol=2e-4
                )
            assert torch.isnan(out[row, n:]).all()
            if flags[row]:
                block = int(state_tables[req, (e - 1) // grain])
                changed_blocks.add(block)
                torch.testing.assert_close(
                    pool[block].cpu(), states[req], atol=2e-5, rtol=2e-4
                )
                checkpoints[req] = c = e
                saw_flush = True
                saw_zero_accept_flush |= a == 0
            else:
                saw_no_flush = True
            _, states[req] = direct(states[req], *(x[row, :a] for x in inputs))
            ends[req] += a
            materialized = (
                torch.zeros_like(states[req])
                if c == 0
                else pool[state_tables[req, (c - 1) // grain]].cpu()
            )
            for token in range(c, ends[req]):
                block, token_row = int(histories[req, token // rows]), token % rows
                materialized = (
                    materialized * hd[block, token_row].cpu()[:, None, :]
                    + hu[block, token_row].cpu()[:, :, None]
                    * hk[block, token_row].cpu()[:, None, :]
                )
            torch.testing.assert_close(materialized, states[req], atol=2e-5, rtol=2e-4)
            # Rejected data is poisoned after validation. Future reads must not
            # observe it until the corresponding token is recomputed.
            for token in range(e + a, e + n):
                block = int(histories[req, token // rows])
                for field in (hk, hu, hd):
                    field[block, token % rows].fill_(float("nan"))
        keep = [i for i in range(state_count) if i not in changed_blocks]
        torch.testing.assert_close(
            pool[keep], before[keep], rtol=0, atol=0, equal_nan=True
        )
        assert torch.isnan(out[-1]).all()
    assert saw_flush and saw_no_flush and saw_zero_accept_flush and saw_block_reuse
    assert [
        t.data_ptr() for t in (pool, hk, hu, hd, stamps, ht, st, cp, out)
    ] == pointers


@pytest.mark.parametrize("graph_mode", [False, True])
def test_invalid_backing_rejects_entire_row_before_stores(graph_mode):
    # Check overflow arithmetic without allocating a multi-gigabyte producer
    # tensor. The token-times-stride product must widen before multiplication.
    offsets = torch.empty(4, dtype=torch.int64, device="cuda")
    _wide_input_offsets[(1,)](offsets)
    if graph_mode:
        offset_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(offset_graph):
            _wide_input_offsets[(1,)](offsets)
        offset_graph.replay()
    assert offsets.tolist() == [i * (2**30 + 4) for i in range(4)]
    batch, width, heads, dim = 4, 4, 1, 16
    shape = (batch, width, heads, dim)
    q = torch.ones(shape, device="cuda") / 4
    v, d = torch.ones_like(q), torch.ones_like(q)
    beta = torch.ones((batch, width, heads), device="cuda") / 2
    pool = torch.ones((9, heads, dim, dim), device="cuda")
    hk = torch.ones((13, 2, heads, dim), device="cuda")
    hu, hd = torch.ones_like(hk), torch.ones_like(hk)
    ht = torch.tensor(
        [[1, 0, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        dtype=torch.int32,
        device="cuda",
    )
    st = torch.tensor(
        [[1, 2], [0, 4], [5, 0], [7, 8]], dtype=torch.int32, device="cuda"
    )
    end = torch.full((batch,), 2, dtype=torch.int32, device="cuda")
    cp = torch.ones(batch, dtype=torch.int64, device="cuda")
    length = torch.ones(batch, dtype=torch.int32, device="cuda")
    valid = torch.full_like(end, width)
    flush = torch.ones(batch, dtype=torch.bool, device="cuda")
    ok = torch.ones_like(flush)
    out = torch.full_like(q, float("nan"))
    before = [t.clone() for t in (pool, hk, hu, hd)]

    def run():
        validate_recurrent_blocks(
            ht,
            st,
            end,
            cp,
            length,
            valid,
            flush,
            ok,
            history_blocks=hk.shape[0],
            state_blocks=pool.shape[0],
            history_block_tokens=hk.shape[1],
            state_block_tokens=1,
            capacity=8,
            max_window=width,
        )
        buffered_recurrent(
            q,
            q,
            v,
            d,
            beta,
            pool,
            hk,
            hu,
            hd,
            ht,
            st,
            end,
            cp,
            length,
            valid,
            flush,
            ok,
            out,
            capacity=8,
            state_block_tokens=1,
            transform_inputs=False,
            A_log=None,
            dt_bias=None,
            lower_bound=None,
        )

    run()
    if graph_mode:
        ok.fill_(True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
    assert ok.tolist() == [False, False, False, True]
    assert torch.isnan(out[:3]).all()
    # Valid fourth row may write only state block 8 and history blocks 11/12.
    for tensor, original, allowed in zip(
        (pool, hk, hu, hd), before, ({8}, {11, 12}, {11, 12}, {11, 12}), strict=True
    ):
        keep = [i for i in range(tensor.shape[0]) if i not in allowed]
        torch.testing.assert_close(tensor[keep], original[keep], rtol=0, atol=0)
