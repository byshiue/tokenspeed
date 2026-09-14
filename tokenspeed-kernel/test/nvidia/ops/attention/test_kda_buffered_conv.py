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

"""Conv production/capture and grouped accepted-window commit, eager and graph."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires a GPU", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda._triton.buffered_conv import (
    buffered_conv,
    commit_conv_windows,
    prepare_conv_blocks,
)
from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
    fused_kda_verify_conv_update,
)


@pytest.mark.parametrize(
    "width,heads,dim", [(1, 2, 16), (4, 2, 16), (1, 12, 128), (4, 12, 128)]
)
@pytest.mark.parametrize("captured", [False, True])
def test_buffered_conv_reordering_acceptance_and_graph(width, heads, dim, captured):
    torch.manual_seed(710)
    layers, groups, requests, batch, cols, grain = 5, 2, 3, 4, 32, 8
    channels = 3 * heads * dim
    blocks = 1 + groups * requests * cols
    slab = torch.full(
        (layers, blocks, channels + 5, 3),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    states = tuple(slab[layer, :, :channels] for layer in range(layers))
    group_ids = [1, 0, 1, 0, 1]
    group_tensor = torch.tensor(group_ids, dtype=torch.int32, device="cuda")
    pointers = torch.tensor(
        [s.data_ptr() for s in states], dtype=torch.int64, device="cuda"
    )
    raw_storage = torch.empty(
        (layers, batch, width, channels + 7), dtype=torch.bfloat16, device="cuda"
    )
    raw = raw_storage[..., :channels]
    payload_storage = torch.full(
        (layers, batch + 1, width, channels + 3),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    payload = payload_storage[:, :batch, :, :channels]
    out = torch.empty_like(raw)
    weights = (
        torch.randn((layers, channels, 4), device="cuda", dtype=torch.bfloat16) * 0.2
    )
    tables = torch.zeros((groups, batch, cols), device="cuda", dtype=torch.int32)
    end = torch.zeros(batch, device="cuda", dtype=torch.int32)
    valid = torch.zeros_like(end)
    accepted = torch.zeros_like(end)
    read = torch.empty((groups, batch), device="cuda", dtype=torch.int32)
    writes = torch.empty((groups, batch, width), device="cuda", dtype=torch.int32)
    ok = torch.ones((groups, batch), device="cuda", dtype=torch.bool)
    ends = [0, 7, 9]
    windows = torch.randn((layers, requests, channels, 3), dtype=torch.bfloat16) * 0.2
    windows[:, 0].zero_()

    def page(group, req, column):
        return 1 + (group * requests + req) * cols + column

    for layer, state in enumerate(states):
        for req, e in enumerate(ends):
            if e:
                state[page(group_ids[layer], req, (e - 1) // grain)].copy_(
                    windows[layer, req]
                )

    def prepare():
        for group in range(groups):
            prepare_conv_blocks(
                tables[group],
                end,
                valid,
                ok[group],
                read[group],
                writes[group],
                blocks=blocks,
                grain=grain,
            )

    def run():
        prepare()
        for layer, state in enumerate(states):
            group = group_ids[layer]
            buffered_conv(
                raw[layer],
                weights[layer],
                state,
                read[group],
                valid,
                ok[group],
                out[layer],
                payload[layer],
            )
        commit_conv_windows(
            payload,
            pointers,
            group_tensor,
            read,
            writes,
            valid,
            ok,
            accepted,
            conv_strides=states[0].stride(),
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
    for step in range(32):
        order = [(step + i) % requests for i in range(requests)]
        widths = [width if (step + i) % 7 else 0 for i in range(requests)] + [0]
        counts = [(step + i) % (w + 1) for i, w in enumerate(widths)]
        end.copy_(
            torch.tensor([ends[r] for r in order] + [2**31 - 1], dtype=torch.int32)
        )
        valid.copy_(torch.tensor(widths, dtype=torch.int32))
        accepted.copy_(torch.tensor(counts, dtype=torch.int32))
        ok.fill_(True)
        for group in range(groups):
            tables[group, :requests].copy_(
                torch.tensor(
                    [[page(group, req, col) for col in range(cols)] for req in order],
                    dtype=torch.int32,
                )
            )
        raw_cpu = torch.randn(raw.shape, dtype=torch.bfloat16) * 0.3
        raw.copy_(raw_cpu)
        out.fill_(float("nan"))
        payload.fill_(float("nan"))
        prepare()
        # Existing GPU conv is the rounding oracle: compare BF16 output exactly,
        # including its FP32 FMA accumulation and SiLU order.
        expected = [
            fused_kda_verify_conv_update(
                raw[layer].view(batch * width, channels),
                weights[layer],
                states[layer],
                read[group_ids[layer]],
                num_heads=heads,
                head_dim=dim,
                draft_token_num=width,
                out=None,
                block_c=256,
                num_warps=4,
            ).view(batch, width, channels)
            for layer in range(layers)
        ]
        if captured:
            graph.replay()
        else:
            run()
        assert ok.all()
        for layer, state in enumerate(states):
            group = group_ids[layer]
            for row, req in enumerate(order):
                w, a = widths[row], counts[row]
                torch.testing.assert_close(
                    out[layer, row, :w], expected[layer][row, :w], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    payload[layer, row, :w], raw[layer, row, :w], atol=0, rtol=0
                )
                assert out[layer, row, w:].isnan().all()
                if a:
                    full = torch.cat(
                        (windows[layer, req], raw_cpu[layer, row, :a].T), dim=-1
                    )
                    windows[layer, req] = full[:, -3:]
                e = ends[req] + a
                if e:
                    actual = state[page(group, req, (e - 1) // grain)]
                    torch.testing.assert_close(
                        actual.cpu(), windows[layer, req], atol=0, rtol=0
                    )
            assert out[layer, -1].isnan().all()
        for row, req in enumerate(order):
            ends[req] += counts[row]
    assert slab[:, :, channels:].isnan().all()
    assert payload_storage[:, batch:].isnan().all()
    assert payload_storage[..., channels:].isnan().all()

    # A missing possible destination invalidates the whole group's row before
    # capture or state stores, including layers at noncontiguous descriptor rows.
    ok.fill_(True)
    valid.fill_(width)
    end.zero_()
    tables[0, 0, 0] = 0
    before = slab.clone()
    accepted.fill_(width + 1)  # Every other row has invalid acceptance: no stores.
    out.fill_(float("nan"))
    if captured:
        graph.replay()
    else:
        run()
    assert not ok[0, 0]
    for layer, group in enumerate(group_ids):
        if group == 0:
            assert out[layer, 0].isnan().all()
    torch.testing.assert_close(slab, before, atol=0, rtol=0, equal_nan=True)
