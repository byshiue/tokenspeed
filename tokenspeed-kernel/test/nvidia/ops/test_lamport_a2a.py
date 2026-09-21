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

"""Run with torchrun --standalone --nproc-per-node=4; no model is loaded."""

import argparse
import datetime
import json
import os
import statistics
from functools import partial

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.cuda import (
    CudaLamportA2AState,
    cuda_lamport_a2a,
)


def reference(x, inverse):
    if inverse:
        rows, width = x.shape[0] // 4, x.shape[1]
        packed = x.view(4, rows, width).contiguous()
        received = torch.empty_like(packed)
        dist.all_to_all_single(received, packed)
        return received.transpose(0, 1).contiguous().view(rows, 4 * width)
    rows, channels = x.shape
    packed = x.view(rows, 4, channels // 4).transpose(0, 1).contiguous()
    received = torch.empty_like(packed)
    dist.all_to_all_single(received, packed)
    return received.view(4 * rows, channels // 4)


def check_bits(output, expected):
    torch.testing.assert_close(
        output.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0
    )


def measure(call):
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            call()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        dist.barrier()
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        value = torch.tensor(
            begin.elapsed_time(end), dtype=torch.float64, device="cuda"
        )
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        samples.append(value.item())
    return samples


def run_cases(args, benchmark):
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(1103 + rank)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group(
            "nccl", device_id=device, timeout=datetime.timedelta(seconds=90)
        )
    for blocks in args.blocks:
        state = CudaLamportA2AState(
            dist.group.WORLD, max(args.rows), args.channels, device, blocks
        )
        # Alternating shapes/directions/inputs on the SAME scratch catches stale
        # packets and ring reuse. Random bits cover signed zeros, infinities,
        # NaNs and subnormals; compare integers, never NaN-tolerant floats.
        for inverse in (False, True, False):
            for rows in args.rows:
                shape = (
                    (4 * rows, args.channels // 4) if inverse else (rows, args.channels)
                )
                bits = torch.randint(
                    -32768, 32768, shape, dtype=torch.int16, device=device
                )
                bits.flatten()[:4] = torch.tensor(
                    [-32768, 0, 32704, -64], dtype=torch.int16, device=device
                )
                x = bits.view(torch.bfloat16)
                expected = reference(x, inverse)
                call = partial(cuda_lamport_a2a, state, x, inverse)
                check_bits(call(), expected)
                # Consumers are captured too: validating only the final replay
                # would miss output corruption during earlier graph iterations.
                sources = [
                    torch.randint(
                        -32768, 32768, shape, dtype=torch.int16, device=device
                    ).view(torch.bfloat16)
                    for _ in range(9)
                ]
                if rank == 0:
                    for source in sources:
                        source.zero_()  # Empty logical owner, still participating.
                references = [reference(source, inverse) for source in sources]
                snapshots = [torch.empty_like(expected) for _ in sources]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for source, snapshot in zip(sources, snapshots):
                        x.copy_(source)
                        # Artificial peer skew tests ring lifetime, not just
                        # lockstep throughput with identical repeated payloads.
                        if rank == 1:
                            torch.cuda._sleep(10000)
                        snapshot.copy_(call())
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                for snapshot, target in zip(snapshots, references):
                    check_bits(snapshot, target)
                del graph, snapshots, sources, references
        # Unsigned generation IDs must also work across the signed-int32
        # boundary. All ranks re-seed after synchronizing, only for this test.
        torch.cuda.synchronize()
        dist.barrier()
        state.control[0] = 2**31 - 2
        if state.chunk_control is not None:
            state.chunk_control[0] = 2**31 - 2
        for rows in (args.rows[0], max(args.rows)):
            x = torch.randn((rows, args.channels), dtype=torch.bfloat16, device=device)
            expected = reference(x, False)
            for _ in range(5):
                check_bits(cuda_lamport_a2a(state, x, False), expected)
        for rows in args.rows if benchmark else ():
            x = torch.randn((rows, args.channels), dtype=torch.bfloat16, device=device)
            expected = reference(x, False)
            call = partial(cuda_lamport_a2a, state, x, False)
            samples = measure(call)
            check_bits(call(), expected)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "backend": "lamport_packet",
                            "rows": rows,
                            "channels": args.channels,
                            "blocks": blocks,
                            "bytes": x.numel() * 2,
                            "median_us": statistics.median(samples),
                            "samples_us": samples,
                            "correctness": "PASS",
                        }
                    ),
                    flush=True,
                )
        dist.barrier()
        del call, state
    if owns_group:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def distributed_group():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group(
            "nccl", device_id=device, timeout=datetime.timedelta(seconds=90)
        )
    yield dist.group.WORLD
    if owns_group:
        dist.destroy_process_group()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires torchrun --nproc-per-node=4",
)
@pytest.mark.parametrize("channels", [8, 40, 12288, 16384])
def test_lamport_a2a(channels, distributed_group):
    assert distributed_group.size() == 4
    run_cases(
        argparse.Namespace(rows=[1, 3, 32, 64, 128], channels=channels, blocks=[128]),
        benchmark=False,
    )


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4,
    reason="requires torchrun --nproc-per-node=4",
)
@pytest.mark.parametrize("channels,threshold", [(32, 4096), (12288, 8 * 2**20)])
def test_chunk_and_packet_transitions(
    channels, threshold, distributed_group, monkeypatch
):
    original = CudaLamportA2AState

    def prepare(group, max_rows, channels, device, blocks):
        state = original(group, max_rows, channels, device, blocks)
        state.prepare_chunk_exchange(threshold_bytes=threshold)
        return state

    assert distributed_group.size() == 4
    monkeypatch.setitem(globals(), "CudaLamportA2AState", prepare)
    run_cases(
        argparse.Namespace(
            rows=[1, 3, 128, 512, 1, 512, 3], channels=channels, blocks=[128]
        ),
        benchmark=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", required=True)
    run_cases(parser.parse_args(), benchmark=True)


if __name__ == "__main__":
    main()
