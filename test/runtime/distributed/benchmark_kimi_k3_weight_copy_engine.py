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

"""Compare NCCL and copy-engine weight prefetch beside production KDA decode.

This is a standalone experiment, not a serving backend. The immutable variant
publishes its weight shard once and never modifies or frees it until every rank
has finished. This permits independent peer reads without per-forward barriers.
"""

import argparse
import ctypes
import gc
import json
import os
import statistics
from pathlib import Path
from test.runtime.distributed.benchmark_kimi_k3_o_proj_weight_prefetch import (
    KdaDecode,
    WeightPrefetch,
    capture,
    measure,
)
from test.runtime.distributed.kimi_k3_o_proj_helpers import (
    dep_mapping,
    load_projection,
    make_linears,
)

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)


class CopyEngineWeightPrefetch(WeightPrefetch):
    """Pull persistent peer shards into local scratch, retaining normal GEMM.

    Args:
        group: Collective group defining shard order.
        weight_cpu: Complete checkpoint FP8 matrix, used only to select a shard.
        scales_cpu: Complete checkpoint scale matrix, sharded identically.
        synchronization: ``barrier`` for general synchronized reads or
            ``immutable`` for shards published once and held for the run.
    """

    def __init__(self, group, weight_cpu, scales_cpu, synchronization):
        super().__init__(group, weight_cpu, scales_cpu)
        assert synchronization in ("barrier", "immutable")
        self.synchronization = synchronization
        self.aux = torch.cuda.Stream(priority=-1)
        source = self.payload
        self.payload = symm_mem.empty(
            source.shape, dtype=source.dtype, device=source.device
        )
        self.payload.copy_(source)
        self.handle = symm_mem.rendezvous(self.payload, group)
        self.sources = [
            self.handle.get_buffer(
                peer, self.payload.shape, self.payload.dtype, storage_offset=0
            )
            for peer in range(self.size)
        ]
        self.destinations = self.gathered.chunk(self.size)
        # Publication is outside capture. Remote reads see initialized shards,
        # which remain alive and unmodified until a collective shutdown fence.
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def gather(self):
        if self.synchronization == "barrier":
            self.handle.barrier(channel=0)
        # Match PyTorch's low-contention pull order. Contiguous byte copies use
        # the CUDA memcpy path; the NSYS trace verifies actual CE execution.
        for step in range(self.size):
            peer = (self.rank - step) % self.size
            self.destinations[peer].copy_(self.sources[peer], non_blocking=True)
        if self.synchronization == "barrier":
            self.handle.barrier(channel=0)


class DirectCopyEngineWeightPrefetch(CopyEngineWeightPrefetch):
    """Read immutable column shards directly into the full GEMM matrix.

    CUDA 2D copies insert each peer's K columns into row-major [N, K] storage.
    Prepared scales are already contiguous in peer order. Neither transfer
    needs the rank-major temporary or an SM restoration kernel. This is a
    benchmark-only CUDA-runtime binding; graph capture retains stable buffers.
    """

    def __init__(self, group, weight_cpu, scales_cpu):
        super().__init__(group, weight_cpu, scales_cpu, "immutable")
        # Resolve CUDA runtime symbols through PyTorch's loaded dependency tree.
        # Extensions may load another libcudart too; do not choose an arbitrary
        # matching path or load a new runtime into the shared environment.
        self.cudart = ctypes.CDLL(
            str(Path(torch.__file__).parent / "lib" / "libtorch_cuda.so"),
            mode=os.RTLD_NOLOAD | os.RTLD_LOCAL,
        )
        self.copy_2d = self.cudart.cudaMemcpy2DAsync
        self.copy_2d.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.copy_2d.restype = ctypes.c_int
        self.copy_bytes = self.cudart.cudaMemcpyAsync
        self.copy_bytes.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.copy_bytes.restype = ctypes.c_int
        self.shard_k = self.k // self.size
        self.weight_shard_bytes = self.n * self.shard_k
        self.scale_shard_bytes = self.scales.numel() * 4 // self.size
        self.destinations = []
        self.gathered = torch.empty(0, device="cuda", dtype=torch.uint8)
        self.scratch_bytes = self.weight.numel() + self.scales.numel() * 4

    def gather(self):
        stream = torch.cuda.current_stream().cuda_stream
        for step in range(self.size):
            peer = (self.rank - step) % self.size
            source = self.sources[peer].data_ptr()
            status = self.copy_2d(
                self.weight.data_ptr() + peer * self.shard_k,
                self.k,
                source,
                self.shard_k,
                self.shard_k,
                self.n,
                3,  # cudaMemcpyDeviceToDevice, including mapped peer memory.
                stream,
            )
            if status != 0:
                raise RuntimeError(f"cudaMemcpy2DAsync failed with CUDA error {status}")
            status = self.copy_bytes(
                self.scales.data_ptr() + peer * self.scale_shard_bytes,
                source + self.weight_shard_bytes,
                self.scale_shard_bytes,
                3,
                stream,
            )
            if status != 0:
                raise RuntimeError(f"cudaMemcpyAsync failed with CUDA error {status}")

    def restore(self):
        # The copies produce the GEMM layout directly.
        pass


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--replays", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--direct-layout", action="store_true")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world % args.tp_size == 0
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_default_dtype(torch.bfloat16)
    mapping = dep_mapping(rank, world)
    pg_manager.init_distributed(
        mapping,
        distributed_init_method="env://",
        backend="nccl",
        timeout=600,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    group = None
    for first in range(0, world, args.tp_size):
        options = dist.ProcessGroupNCCL.Options()
        options.is_high_priority_stream = True
        candidate = dist.new_group(
            list(range(first, first + args.tp_size)), backend="nccl", pg_options=options
        )
        if first <= rank < first + args.tp_size:
            group = candidate
    assert group is not None
    weight, scales, quant = load_projection(str(args.model), 0)
    (baseline, _), unused_compute = make_linears(mapping, weight, scales, quant)
    del unused_compute
    states = {"nccl_high": WeightPrefetch(group, weight, scales)}
    states["nccl_high"].aux = torch.cuda.Stream(priority=-1)
    for synchronization in ("barrier", "immutable"):
        states[f"ce_{synchronization}"] = CopyEngineWeightPrefetch(
            group, weight, scales, synchronization
        )
    if args.direct_layout:
        states["ce_direct"] = DirectCopyEngineWeightPrefetch(group, weight, scales)
    for state in states.values():
        state.prepare()
        torch.testing.assert_close(
            state.weight.view(torch.uint8).cpu(),
            weight.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state.scales.cpu(), scales.t().contiguous(), rtol=0, atol=0
        )
    result = {
        "world_size": world,
        "weight_tp_size": args.tp_size,
        "weight_shape_NK": list(weight.shape),
        "weight_dtype": str(weight.dtype),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "nccl": torch.cuda.nccl.version(),
        "gpu": torch.cuda.get_device_name(),
        "symm_mem_backend": symm_mem.get_backend(torch.device("cuda")),
        "steps": args.steps,
        "samples": args.samples,
        "replays": args.replays,
        "allgather_bytes_per_rank": states["nccl_high"].gathered.numel(),
        "memory": {
            name: {
                "shard_bytes": state.shard_bytes,
                "scratch_bytes": state.scratch_bytes,
            }
            for name, state in states.items()
        },
        "scope": "Real-weight KDA attention and O projection only; not full-model E2E",
        "cases": [],
    }
    for rows in args.rows:
        attention = KdaDecode(args.model, rows, rank)
        x = attention()
        reference = baseline(x)[0].clone()
        calls = {
            "attention": attention,
            "dep16_projection": lambda: baseline(x)[0],
            "dep16_pipeline": lambda: baseline(attention())[0],
        }
        for name, state in states.items():
            calls[f"{name}_gather"] = state.gather
            if state.gathered.numel():
                calls[f"{name}_restore"] = state.restore
            calls[f"{name}_prepare"] = state.prepare
            calls[f"{name}_pipeline"] = lambda s=state: s.overlap(attention)
            torch.testing.assert_close(
                state.overlap(attention), reference, rtol=0, atol=0
            )
        graphs, outputs = {}, {}
        for name, call in calls.items():
            graphs[name], outputs[name] = capture(call, args.steps)
        # Deliberately poison both gathered and restored scratch and change
        # activations: each replay must really fetch weights and obey joins.
        for iteration in range(3):
            attention.qkv.add_(0.03125)
            reference = baseline(attention())[0].clone()
            for name, state in states.items():
                state.gathered.zero_()
                state.weight.zero_()
                state.scales.fill_(float("nan"))
                # Immutable sources permit asymmetric consumers: delaying one
                # reader must not change any peer's values or buffer lifetime.
                if rank % args.tp_size == iteration % args.tp_size:
                    torch.cuda._sleep(300000)
                graphs[f"{name}_pipeline"].replay()
                torch.testing.assert_close(
                    outputs[f"{name}_pipeline"], reference, rtol=0, atol=0
                )
        timings = measure(graphs, args.steps, args.samples, args.replays)
        entry = {
            "rows_per_rank": rows,
            "global_requests": rows * world,
            "correctness": "bit-exact reconstruction and eager/changed-input graph outputs; delayed peers",
            "median_us_max_rank": {
                name: statistics.median(values) for name, values in timings.items()
            },
            "samples_us_max_rank": timings,
        }
        result["cases"].append(entry)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(entry), flush=True)
        if args.profile:
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            for name in ["dep16_pipeline"] + [f"{name}_pipeline" for name in states]:
                with torch.cuda.nvtx.range(f"{name}_TP{args.tp_size}_C{rows}"):
                    graphs[name].replay()
                    torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        # Destroy graph-held NCCL references before destroying the group.
        for graph in graphs.values():
            graph.reset()
        del graphs, outputs, calls, attention, x, reference
        gc.collect()
    torch.cuda.synchronize()
    dist.barrier()
    # All remote reads have now finished; persistent symmetric weights may be
    # released during process teardown, never at an individual local GEMM end.
    dist.destroy_process_group()
    if rank == 0:
        print("COPY_ENGINE_WEIGHT_PREFETCH_PASSED", flush=True)


if __name__ == "__main__":
    main()
