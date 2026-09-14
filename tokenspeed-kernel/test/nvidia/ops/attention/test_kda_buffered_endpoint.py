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


"""Endpoint reconstruction boundaries, separate from scheduler handoff."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires a GPU", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda._triton.buffered_endpoint import (
    materialize_endpoints,
    prepare_endpoint_commit,
)


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("captured", [False, True])
def test_endpoint_identity_zero_seed_rejection_and_graph(width, captured):
    torch.manual_seed(713)
    layers, groups, batch, heads, dim, cols, rows, grain, prefix = (
        3,
        2,
        6,
        2,
        16,
        12,
        3,
        8,
        16,
    )
    pages = 1 + groups * batch * cols
    ht = torch.arange(1, pages, device="cuda", dtype=torch.int32).view(
        groups, batch, cols
    )
    st = ht.clone()
    state = torch.randn((layers, pages, heads, dim, dim + 3), device="cuda")[..., :dim]
    hk = (
        torch.randn((layers, pages, rows, heads, dim + 5), device="cuda")[..., :dim]
        * 0.1
    )
    hu = torch.randn_like(hk) * 0.02
    hd = torch.full_like(hk, 0.95)
    group_ids = [1, 0, 1]
    ids = torch.tensor(group_ids, dtype=torch.int32, device="cuda")
    descriptors = torch.tensor(
        [
            [
                state[l].data_ptr(),
                hk[l].data_ptr(),
                hu[l].data_ptr(),
                hd[l].data_ptr(),
                ht[g].data_ptr(),
                st[g].data_ptr(),
            ]
            for l, g in enumerate(group_ids)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    counts = [1, min(2, width), 0, 1, 1, 0]
    ends = [0, 8 - counts[1], 7, 15, 18, 2**31 - 1]
    checkpoints = [0, 5, 5, 13, 17, 0]
    flushed = torch.tensor(
        [[False, False, False, False, True, False]] * groups, device="cuda"
    )
    cp = torch.tensor([checkpoints] * groups, dtype=torch.int64, device="cuda")
    end = torch.tensor(ends, dtype=torch.int32, device="cuda")
    valid = torch.zeros(batch, dtype=torch.int32, device="cuda")
    accepted = torch.zeros_like(valid)
    force = torch.tensor([True, False, True, False, True, True], device="cuda")
    ok = torch.ones((groups, batch), dtype=torch.bool, device="cuda")
    needed = torch.zeros_like(ok)

    def run():
        for group in range(groups):
            prepare_endpoint_commit(
                end,
                valid,
                accepted,
                cp[group],
                flushed[group],
                ok[group],
                force,
                needed[group],
                prefix_granularity=prefix,
                for_handoff=False,
            )
        materialize_endpoints(
            descriptors,
            ids,
            end,
            accepted,
            cp,
            flushed,
            ok,
            needed,
            heads=heads,
            key_dim=dim,
            value_dim=dim,
            history_block_tokens=rows,
            state_block_tokens=grain,
            table_stride=cols,
            state_strides=state[0].stride(),
            key_strides=hk[0].stride(),
            correction_strides=hu[0].stride(),
            decay_strides=hd[0].stride(),
            max_programs=8,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    if captured:
        with torch.cuda.graph(graph):
            run()
    valid.copy_(torch.tensor([width] * 5 + [0]))
    accepted.copy_(torch.tensor(counts))
    expected = state.cpu().clone()
    # Only accepted history is readable. NaN candidates make an overlong
    # reconstruction fail even if its final store used the correct address.
    for layer, group in enumerate(group_ids):
        for row in range(5):
            for pos in range(ends[row] + counts[row], ends[row] + width):
                block = int(ht[group, row, pos // rows])
                for field in (hk, hu, hd):
                    field[layer, block, pos % rows].fill_(float("nan"))
    source_fields = [field.cpu() for field in (hk, hu, hd)]
    ht_cpu, st_cpu = ht.cpu(), st.cpu()
    for layer, group in enumerate(group_ids):
        for row in (0, 2, 3, 4):
            source = ends[row] if row == 4 else checkpoints[row]
            endpoint = ends[row] + counts[row]
            if source:
                value = expected[
                    layer, st_cpu[group, row, (source - 1) // grain]
                ].clone()
            else:
                value = torch.zeros((heads, dim, dim))
            for pos in range(source, endpoint):
                block = ht_cpu[group, row, pos // rows]
                k, u, d = [field[layer, block, pos % rows] for field in source_fields]
                value = value * d[:, None, :] + u[:, :, None] * k[:, None, :]
            expected[layer, st_cpu[group, row, (endpoint - 1) // grain]] = value
    if captured:
        graph.replay()
    else:
        run()
    assert needed.tolist() == [[True, False, True, True, True, False]] * groups
    assert ok.all()
    torch.testing.assert_close(state.cpu(), expected, atol=2e-5, rtol=2e-4)
    # Endpoint 8 is aligned to the state span, not to prefix identity 16.
    # Its unchanged state and false flag above catch using the wrong granularity.
