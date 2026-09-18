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

"""Column projection shape policy and token/channel ownership contracts."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention import column_proj
from tokenspeed.runtime.layers.attention.o_proj import projection_mapping


def test_column_projection_padding_and_settings(monkeypatch):
    for width, expected in (
        (36864, 36864),
        (49376, 49664),
        (14400, 14848),
        (18432, 18432),
    ):
        padded = column_proj.column_projection_width(width, 4, 128)
        assert padded == expected and padded // 4 % 128 == 0
        assert width <= padded < width + 512
    for values in ((0, 4, 128), (1, 0, 128), (1, 4, 0)):
        with pytest.raises(ValueError):
            column_proj.column_projection_width(*values)
    parallel = projection_mapping(0, 4, 4)
    monkeypatch.setattr(column_proj.pg_manager, "get_process_group", lambda *args: None)
    for ag, a2a in (("invalid", "nccl"), ("nccl", "invalid")):
        with pytest.raises(ValueError):
            column_proj.DistributedColumnProjection(
                parallel,
                128,
                7,
                8,
                3,
                torch.bfloat16,
                torch.device("cpu"),
                ag,
                a2a,
            )


def test_column_projection_restores_uneven_rank_and_channel_order(monkeypatch):
    parallel = projection_mapping(5, 8, 4)
    monkeypatch.setattr(column_proj.pg_manager, "get_process_group", lambda *args: None)
    module = column_proj.DistributedColumnProjection(
        parallel,
        3,
        7,
        8,
        3,
        torch.bfloat16,
        torch.device("cpu"),
        "nccl",
        "nccl",
    )
    counts = [0, 0, 0, 0, 2, 3, 0, 1]
    inputs = [
        torch.arange(count * 3, dtype=torch.bfloat16).reshape(count, 3) + rank * 16
        for rank, count in enumerate(counts)
    ]
    weight = torch.arange(24, dtype=torch.bfloat16).reshape(8, 3)
    gathered = torch.cat(
        [
            torch.cat(
                (inputs[rank], torch.zeros(3 - counts[rank], 3, dtype=torch.bfloat16))
            )
            for rank in parallel.tp_group
        ]
    )
    calls = []

    def gather(out, source, group, backend):
        assert group == parallel.tp_group and backend is None
        torch.testing.assert_close(source, inputs[5])
        out.copy_(gathered)
        calls.append("gather")

    def exchange(out, source, group, backend):
        assert group == parallel.tp_group and backend is None
        torch.testing.assert_close(source, gathered @ weight[2:4].T)
        out.copy_(
            torch.cat([inputs[5] @ weight[2 * r : 2 * r + 2].T for r in range(4)])
        )
        calls.append("a2a")

    monkeypatch.setattr(column_proj, "all_gather_single", gather)
    monkeypatch.setattr(column_proj, "all_to_all_single", exchange)

    class Linear:
        gather_output = False
        bias = None
        tp_group = parallel.tp_group
        tp_rank = 1
        tp_size = 4
        input_size = 3
        output_size = 8

        def __call__(self, x):
            torch.testing.assert_close(x, gathered)
            return x @ weight[2:4].T, None

    linear = Linear()
    actual = module.forward(inputs[5], linear, counts)
    torch.testing.assert_close(actual, inputs[5] @ weight[:7].T)
    assert calls == ["gather", "a2a"]
    retained = actual.clone()
    module.received.zero_()
    torch.testing.assert_close(actual, retained)

    # With one row the transpose is already contiguous; contiguous() alone
    # would alias reusable receive storage instead of returning owned output.
    def single_row_exchange(out, source, group, backend):
        out.copy_(torch.arange(8, dtype=torch.bfloat16).view(4, 2))

    monkeypatch.setattr(column_proj, "all_to_all_single", single_row_exchange)
    single = module.restore_outputs(torch.empty(4, 2, dtype=torch.bfloat16), 1)
    module.received.zero_()
    torch.testing.assert_close(single, torch.arange(8, dtype=torch.bfloat16).view(1, 8))
    empty = module.forward(inputs[5][:0], linear, [0] * 8)
    assert empty.shape == (0, 7) and len(calls) == 2
    with pytest.raises(ValueError):
        module.forward(inputs[5], linear, [0] * 8)
    bad = SimpleNamespace(
        **{
            key: getattr(linear, key)
            for key in (
                "gather_output",
                "bias",
                "tp_group",
                "tp_rank",
                "tp_size",
                "input_size",
                "output_size",
            )
        }
    )
    bad.gather_output = True
    with pytest.raises(ValueError):
        module.forward(inputs[5], bad, counts)
