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

"""TP4 peer reduction correctness, input ownership, and graph replay."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication.triton import (
    ProjectionPeerState,
    triton_projection_reduce_scatter,
)


def _reference(partial, rows, rank):
    peers = [torch.empty_like(partial) for _ in range(4)]
    dist.all_gather(peers, partial)
    result = torch.zeros(
        (rows, partial.shape[1]), device=partial.device, dtype=torch.float32
    )
    for peer in peers:
        result.add_(peer[rank * rows : (rank + 1) * rows].float())
    return result.to(partial.dtype)


def _fill(partial, rank, generation):
    values = torch.arange(partial.numel(), device=partial.device, dtype=torch.float32)
    values = values.view_as(partial).remainder_(31)
    values.mul_(0.015625).add_((rank + 1) * 0.125 + generation * 0.03125)
    partial.copy_(values.to(partial.dtype))


def _worker(rank, rendezvous):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=600),
        device_id=device,
    )
    state = ProjectionPeerState(dist.group.WORLD, 129, 128, device)

    for rows in (1, 17, 128, 129):
        direct = state.input_buffer(rows)
        _fill(direct, rank, rows)
        expected = _reference(direct, rows, rank)
        actual = triton_projection_reduce_scatter(state, direct, rows)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        copied = direct.clone()
        expected = _reference(copied, rows, rank)
        actual = triton_projection_reduce_scatter(state, copied, rows)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    rows = 32
    direct = state.input_buffer(rows)
    _fill(direct, rank, 0)
    for _ in range(2):
        triton_projection_reduce_scatter(state, direct, rows)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        retained = triton_projection_reduce_scatter(state, direct, rows)
    for generation in range(1, 5):
        _fill(direct, rank, generation)
        expected = _reference(direct, rows, rank)
        graph.replay()
        torch.testing.assert_close(retained, expected, rtol=0, atol=0)

    torch.cuda.synchronize()
    dist.barrier()
    del graph, retained, state
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="Four NVLink CUDA GPUs required"
)
def test_projection_reduce_scatter(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=4, join=True)
