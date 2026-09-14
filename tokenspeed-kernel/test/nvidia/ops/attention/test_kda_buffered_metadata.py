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


"""Position lifecycle gates for the unregistered paged replay prototype."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires a GPU", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda._triton.buffered_metadata import (  # noqa: E402
    commit_positions,
    prepare_positions,
)


@pytest.mark.parametrize("max_window,capacity", [(1, 8), (4, 8), (4, 37)])
@pytest.mark.parametrize("captured", [False, True])
def test_paged_positions_reordering_acceptance_and_graph(
    max_window, capacity, captured
):
    batch, requests, rows, columns = 5, 4, 16, 32
    # Neither page stride nor row stride is a dense checkpoint matrix.
    storage = torch.zeros(
        (1 + requests * columns, rows, 5), dtype=torch.int64, device="cuda"
    )
    stamps = storage[:, :, 2]
    expected = torch.zeros_like(stamps, device="cpu")
    table = torch.full((batch, columns), -1, dtype=torch.int32, device="cuda")
    end = torch.zeros(batch, dtype=torch.int32, device="cuda")
    width = torch.zeros_like(end)
    accepted = torch.zeros_like(end)
    checkpoint = torch.empty(batch, dtype=torch.int64, device="cuda")
    length = torch.empty_like(end)
    flush = torch.empty(batch, dtype=torch.bool, device="cuda")
    ok = torch.empty_like(flush)

    def run():
        prepare_positions(
            stamps,
            table,
            end,
            width,
            checkpoint,
            length,
            flush,
            ok,
            capacity=capacity,
            max_window=max_window,
        )
        commit_positions(
            (stamps,), table, end, width, accepted, checkpoint, length, flush, ok
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
    pointers = [t.data_ptr() for t in (stamps, table, checkpoint, length, flush, ok)]
    ends = [15, 16, 31, 0]
    checkpoints = list(ends)
    saw_flush = saw_zero_accept_flush = False
    for step in range(64):
        order = [(step + i) % requests for i in range(requests)]
        widths = [max_window if (step + i) % 7 else 0 for i in range(requests)] + [0]
        accepts = [(step + i) % (w + 1) for i, w in enumerate(widths)]
        # Deliberately exercise zero acceptance at the first flush; the
        # deterministic acceptance pattern alone need not hit that case.
        if not saw_zero_accept_flush:
            for row, req in enumerate(order):
                if (
                    widths[row]
                    and ends[req] - checkpoints[req] + 2 * max_window > capacity
                ):
                    accepts[row] = 0
        tables = [[1 + req * columns + col for col in range(columns)] for req in order]
        tables.append([-1] * columns)
        # Padding deliberately carries an endpoint outside the allocated table.
        end.copy_(
            torch.tensor([ends[req] for req in order] + [2**31 - 1], dtype=torch.int32)
        )
        table.copy_(torch.tensor(tables, dtype=torch.int32))
        width.copy_(torch.tensor(widths, dtype=torch.int32))
        accepted.copy_(torch.tensor(accepts, dtype=torch.int32))
        exp_c, exp_h, exp_f = [], [], []
        for row, req in enumerate(order):
            e, c, w, a = ends[req], checkpoints[req], widths[row], accepts[row]
            flushing = w > 0 and e - c + 2 * max_window > capacity
            exp_c.append(c if w else 0)
            exp_h.append(e - c if w else 0)
            exp_f.append(flushing)
            if flushing:
                c = e
                saw_flush = True
                saw_zero_accept_flush |= a == 0
            if w and (a or flushing):
                token = e + a - 1
                expected[tables[row][token // rows], token % rows] = c + 1
            checkpoints[req] = c
            ends[req] += a
        if captured:
            graph.replay()
        else:
            run()
        assert checkpoint.tolist() == exp_c + [0]
        assert length.tolist() == exp_h + [0]
        assert flush.tolist() == exp_f + [False]
        assert ok.all()
        torch.testing.assert_close(stamps.cpu(), expected, rtol=0, atol=0)
        assert not torch.count_nonzero(storage[:, :, :2])
        assert not torch.count_nonzero(storage[:, :, 3:])
    assert saw_flush and saw_zero_accept_flush
    assert [
        t.data_ptr() for t in (stamps, table, checkpoint, length, flush, ok)
    ] == pointers


def test_invalid_rows_do_not_write_and_zeroed_page_reseeds():
    stamps = torch.zeros((9, 16), dtype=torch.int64, device="cuda")
    table = torch.tensor(
        [[1, 2], [3, 4], [5, 0], [7, 8]], dtype=torch.int32, device="cuda"
    )
    end = torch.tensor([5, 5, 16, 5], dtype=torch.int32, device="cuda")
    width = torch.full((4,), 4, dtype=torch.int32, device="cuda")
    accepted = torch.tensor([1, 5, 1, 1], dtype=torch.int32, device="cuda")
    checkpoint = torch.empty(4, dtype=torch.int64, device="cuda")
    length = torch.empty(4, dtype=torch.int32, device="cuda")
    flush = torch.empty(4, dtype=torch.bool, device="cuda")
    ok = torch.empty_like(flush)
    stamps[1, 4] = 999  # Checkpoint beyond the accepted endpoint is corrupt.
    initial = stamps.clone()
    prepare_positions(
        stamps,
        table,
        end,
        width,
        checkpoint,
        length,
        flush,
        ok,
        capacity=8,
        max_window=4,
    )
    assert ok.tolist() == [False, True, True, True]
    commit_positions(
        (stamps,), table, end, width, accepted, checkpoint, length, flush, ok
    )
    assert ok.tolist() == [False, False, False, True]
    initial[7, 5] = 6
    torch.testing.assert_close(stamps, initial, rtol=0, atol=0)
    # Page reuse is legal only after allocator zeroing and endpoint materialization.
    stamps[1].zero_()
    prepare_positions(
        stamps,
        table,
        end,
        width,
        checkpoint,
        length,
        flush,
        ok,
        capacity=8,
        max_window=4,
    )
    assert checkpoint[0] == 5 and length[0] == 0 and ok[0]
    with pytest.raises(ValueError, match="maximum windows"):
        prepare_positions(
            stamps,
            table,
            end,
            width,
            checkpoint,
            length,
            flush,
            ok,
            capacity=7,
            max_window=4,
        )
    with pytest.raises(ValueError, match="accepted"):
        commit_positions(
            (stamps,), table, end, width, accepted.long(), checkpoint, length, flush, ok
        )
