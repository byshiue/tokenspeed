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

"""Isolate small scale-copy scheduling beside production KDA decode.

Weight-only and scale-only modes intentionally retain the omitted operand.
They are diagnostic controls, not equivalent full weight-prefetch backends.
All copied operands and final outputs are nevertheless checked bit-exactly.
"""

import argparse
import json
import os
import statistics
from functools import partial
from pathlib import Path
from test.runtime.distributed.benchmark_kimi_k3_o_proj_weight_prefetch import (
    KdaDecode,
    WeightPrefetch,
    measure,
)
from test.runtime.distributed.benchmark_kimi_k3_weight_copy_engine import (
    CopyEngineWeightPrefetch,
    DirectCopyEngineWeightPrefetch,
)
from test.runtime.distributed.kimi_k3_o_proj_helpers import (
    dep_mapping,
    load_projection,
    make_linears,
)

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_blockscale
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    flashinfer_fp8_blockscale_quantize_prepacked,
)

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)


@triton.jit
def copy_schedule_marker(output):
    tl.store(output, 0)


@triton.jit
def gather_weight_scales(
    sources,
    output,
    WEIGHT_WORDS: tl.constexpr,
    SCALE_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy one immutable peer's scales per CTA into the prepared GEMM layout."""
    peer = tl.program_id(0)
    source = tl.load(sources + peer).to(tl.pointer_type(tl.int32))
    offsets = tl.arange(0, BLOCK)
    bits = tl.load(source + WEIGHT_WORDS + offsets, offsets < SCALE_WORDS, other=0)
    tl.store(output + peer * SCALE_WORDS + offsets, bits, offsets < SCALE_WORDS)


@triton.jit
def copy_local_weight_and_gather_scales(
    sources,
    weight,
    scales,
    RANK: tl.constexpr,
    PEERS: tl.constexpr,
    K_WORDS: tl.constexpr,
    WEIGHT_WORDS: tl.constexpr,
    SCALE_WORDS: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    VECTOR_STORE: tl.constexpr,
    VECTOR_LOAD: tl.constexpr,
    ASM_LOAD: tl.constexpr,
    STREAMING: tl.constexpr,
):
    """Replace the late local CUDA 2D copy with a bounded ordinary CTA grid.

    Remote weight copies remain on copy engines. This kernel touches only the
    local weight columns and all peers' tiny scale slices, bit-for-bit.
    """
    pid = tl.program_id(0)
    source_address = tl.load(sources + RANK)
    if VECTOR_LOAD:
        # Symmetric allocations are aligned; pointer-table loads otherwise lose
        # this information and generate scalar loads even with vector stores.
        source_address = tl.multiple_of(source_address, 16)
    source = source_address.to(tl.pointer_type(tl.int32))
    shard_k = K_WORDS // PEERS
    for start in range(pid * BLOCK, WEIGHT_WORDS, tl.num_programs(0) * BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        if ASM_LOAD:
            # This experiment admits only whole tiles. Four adjacent int32
            # words form one aligned 128-bit load; no tail may be over-read.
            tl.static_assert(WEIGHT_WORDS % BLOCK == 0)
            bits = tl.inline_asm_elementwise(
                asm="ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
                constraints="=r,=r,=r,=r,l,l,l,l",
                args=[source + offsets],
                dtype=tl.int32,
                is_pure=True,
                pack=4,
            )
        else:
            bits = tl.load(
                source + offsets,
                offsets < WEIGHT_WORDS,
                other=0,
                cache_modifier=".cg" if STREAMING else "",
            )
        destinations = offsets // shard_k * K_WORDS + RANK * shard_k + offsets % shard_k
        if VECTOR_STORE:
            # Both row pitches and loop offsets are multiples of four words:
            # every aligned group of four destinations is contiguous.
            destinations = tl.max_contiguous(destinations, 4)
        tl.store(
            weight + destinations,
            bits,
            offsets < WEIGHT_WORDS,
            cache_modifier=".cs" if STREAMING else "",
        )
    if pid < PEERS:
        peer_address = tl.load(sources + pid)
        if VECTOR_LOAD:
            peer_address = tl.multiple_of(peer_address, 16)
        peer_source = peer_address.to(tl.pointer_type(tl.int32))
        offsets = tl.arange(0, SCALE_BLOCK)
        bits = tl.load(
            peer_source + WEIGHT_WORDS + offsets, offsets < SCALE_WORDS, other=0
        )
        tl.store(scales + pid * SCALE_WORDS + offsets, bits, offsets < SCALE_WORDS)


# Explicit experimental choices: remote-first, fused scales, late join,
# attention-first graph construction. Every mode refreshes all operands.
SCHEDULE_MODES = {
    "late_join": (False, False, True, False),
    "remote_first": (True, False, False, False),
    "remote_first_late_join": (True, False, True, False),
    "fused_scales": (False, True, False, False),
    "fused_scales_late_join": (False, True, True, False),
    "remote_first_fused_scales": (True, True, True, False),
    "attention_first_fused_scales": (False, True, True, True),
    "attention_first_remote_fused_scales": (True, True, True, True),
}
LOCAL_COPY_MODES = {f"local_kernel_{ctas}": ctas for ctas in (4, 16, 32, 64, 128, 152)}
LOCAL_COPY_MODES["local_kernel_sm"] = -1
# CTA count, word tile, vector store, streaming cache policy, local-first,
# early stream join, vector load. All fields are explicit.
LOCAL_TUNING_MODES = {
    "local_vector_sm": (-1, 1024, True, False, False, False, False),
    "local_vector_4096_sm": (-1, 4096, True, False, False, False, False),
    "local_vector_4096_32": (32, 4096, True, False, False, False, False),
    "local_streaming_sm": (-1, 1024, True, True, False, False, False),
    "local_vector_first": (-1, 1024, True, False, True, False, False),
    "local_vector_early_join": (-1, 1024, True, False, False, True, False),
    "local_vector_first_early_join": (-1, 1024, True, False, True, True, False),
    "local_vec_load_sm": (-1, 1024, True, False, False, False, True),
    "local_vec_load_32": (32, 1024, True, False, False, False, True),
    "local_vec_load_4096_sm": (-1, 4096, True, False, False, False, True),
    "local_vec_load_first": (-1, 1024, True, False, True, False, True),
    "local_asm_load_sm": (-1, 1024, True, False, False, False, True),
    "local_asm_load_32": (32, 1024, True, False, False, False, True),
    "local_asm_load_4096_sm": (-1, 4096, True, False, False, False, True),
}
ASM_LOAD_MODES = {"local_asm_load_sm", "local_asm_load_32", "local_asm_load_4096_sm"}
REMOTE_STREAM_MODES = {
    f"remote_streams_{count}": (count, 1024, False) for count in (1, 2, 3)
}
REMOTE_STREAM_MODES["remote_streams_3_asm"] = (3, 4096, True)


class ScaleCopyDiagnostic(DirectCopyEngineWeightPrefetch):
    """Control copy order or omit one immutable operand to isolate its cost."""

    def __init__(self, group, weight_cpu, scales_cpu, mode):
        super().__init__(group, weight_cpu, scales_cpu)
        self.mode = mode
        self.copy_mode = mode.removesuffix("_normal").removesuffix("_marker")
        self.use_marker = mode.removesuffix("_normal").endswith("_marker")
        self.marker = torch.empty(1, device="cuda", dtype=torch.int32)
        if mode.endswith("_normal"):
            self.aux = torch.cuda.Stream(priority=0)
        assert self.copy_mode in (
            "interleaved",
            "weights_only",
            "scales_only",
            "weights_then_scales",
            "scales_then_weights",
            "skip_local_scale",
        )
        # Establish retained operands before capture, never in a timed call.
        self.weight.copy_(weight_cpu)
        self.scales.copy_(scales_cpu.t().contiguous())

    def copy_weight(self, peer, stream):
        status = self.copy_2d(
            self.weight.data_ptr() + peer * self.shard_k,
            self.k,
            self.sources[peer].data_ptr(),
            self.shard_k,
            self.shard_k,
            self.n,
            3,
            stream,
        )
        if status:
            raise RuntimeError(f"cudaMemcpy2DAsync failed: {status}")

    def copy_scale(self, peer, stream):
        status = self.copy_bytes(
            self.scales.data_ptr() + peer * self.scale_shard_bytes,
            self.sources[peer].data_ptr() + self.weight_shard_bytes,
            self.scale_shard_bytes,
            3,
            stream,
        )
        if status:
            raise RuntimeError(f"cudaMemcpyAsync failed: {status}")

    def gather(self):
        stream = torch.cuda.current_stream().cuda_stream
        peers = [(self.rank - step) % self.size for step in range(self.size)]
        if self.copy_mode in ("interleaved", "skip_local_scale"):
            for peer in peers:
                self.copy_weight(peer, stream)
                if self.copy_mode != "skip_local_scale" or peer != self.rank:
                    self.copy_scale(peer, stream)
        else:
            operations = {
                "weights_only": (self.copy_weight,),
                "scales_only": (self.copy_scale,),
                "weights_then_scales": (self.copy_weight, self.copy_scale),
                "scales_then_weights": (self.copy_scale, self.copy_weight),
            }[self.copy_mode]
            for operation in operations:
                for peer in peers:
                    operation(peer, stream)
        if self.use_marker:
            # A one-store compute node tests graph scheduling at the selected priority.
            # It does not replace any copy or synchronization dependency.
            copy_schedule_marker[(1,)](self.marker, num_warps=1)

    def poison_copied_operands(self):
        if self.copy_mode != "scales_only":
            self.weight.zero_()
        if self.copy_mode != "weights_only":
            for peer, chunk in enumerate(self.scales.chunk(self.size)):
                if self.copy_mode != "skip_local_scale" or peer != self.rank:
                    chunk.fill_(float("nan"))


class PrefetchScheduleOptimization(ScaleCopyDiagnostic):
    """Keep the complete transfer contract while changing copy/join scheduling.

    Scales are refreshed by either CUDA copies plus a marker, or a four-CTA
    gather that also supplies the compute node. Activation quantization needs
    only attention output, so the weight-ready wait may move immediately before
    GEMM. Scratch still cannot be overwritten until the preceding GEMM ends.
    """

    def __init__(self, group, weight_cpu, scales_cpu, mode):
        super().__init__(group, weight_cpu, scales_cpu, "weights_then_scales_marker")
        self.optimization_mode = mode
        self.local_copy_ctas = LOCAL_COPY_MODES.get(mode, 0)
        self.local_copy_block = 1024
        self.vector_store = False
        self.vector_load = False
        self.asm_load = mode in ASM_LOAD_MODES
        self.streaming = False
        self.local_first = False
        early_join = False
        self.peer_streams = []
        self.local_copy_kernel = None
        if mode in REMOTE_STREAM_MODES:
            count, self.local_copy_block, self.asm_load = REMOTE_STREAM_MODES[mode]
            self.peer_streams = [torch.cuda.Stream(priority=-1) for _ in range(count)]
            self.local_copy_ctas = -1
            self.vector_load = self.asm_load
            self.vector_store = self.asm_load
        if mode in LOCAL_TUNING_MODES:
            (
                self.local_copy_ctas,
                self.local_copy_block,
                self.vector_store,
                self.streaming,
                self.local_first,
                early_join,
                self.vector_load,
            ) = LOCAL_TUNING_MODES[mode]
        if self.vector_load:
            assert all(source.data_ptr() % 16 == 0 for source in self.sources)
            assert self.weight_shard_bytes % 16 == 0
        if self.local_copy_ctas == -1:
            self.local_copy_ctas = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        (
            self.remote_first,
            self.fused_scales,
            self.late_join,
            self.attention_first,
        ) = (
            (True, True, True, False) if self.local_copy_ctas else SCHEDULE_MODES[mode]
        )
        if early_join:
            self.late_join = False
        self.source_pointers = torch.tensor(
            [source.data_ptr() for source in self.sources],
            device="cuda",
            dtype=torch.int64,
        )

    def copy_local(self):
        self.local_copy_kernel = copy_local_weight_and_gather_scales[
            (self.local_copy_ctas,)
        ](
            self.source_pointers,
            self.weight.view(torch.int32),
            self.scales.view(torch.int32),
            self.rank,
            self.size,
            self.k // 4,
            self.weight_shard_bytes // 4,
            self.scale_shard_bytes // 4,
            triton.next_power_of_2(self.scale_shard_bytes // 4),
            self.local_copy_block,
            self.vector_store,
            self.vector_load,
            self.asm_load,
            self.streaming,
            num_warps=4,
        )

    def gather(self):
        stream = torch.cuda.current_stream().cuda_stream
        peers = [(self.rank - step) % self.size for step in range(self.size)]
        if self.peer_streams:
            owner_stream = torch.cuda.current_stream()
            # All branches inherit the previous GEMM's scratch-release edge.
            # Destinations are disjoint, and peer sources are immutable.
            for copy_stream in self.peer_streams:
                copy_stream.wait_stream(owner_stream)
            for index, peer in enumerate(peers[1:]):
                copy_stream = self.peer_streams[index % len(self.peer_streams)]
                with torch.cuda.stream(copy_stream):
                    self.copy_weight(peer, copy_stream.cuda_stream)
            self.copy_local()
            for copy_stream in self.peer_streams:
                owner_stream.wait_stream(copy_stream)
            return
        if self.remote_first:
            peers = peers[1:] + peers[:1]
        if self.local_first:
            self.copy_local()
        for peer in peers:
            if not self.local_copy_ctas or peer != self.rank:
                self.copy_weight(peer, stream)
        if self.local_copy_ctas:
            if not self.local_first:
                self.copy_local()
        elif self.fused_scales:
            gather_weight_scales[(self.size,)](
                self.source_pointers,
                self.scales.view(torch.int32),
                self.weight_shard_bytes // 4,
                self.scale_shard_bytes // 4,
                triton.next_power_of_2(self.scale_shard_bytes // 4),
                num_warps=4,
            )
        else:
            for peer in peers:
                self.copy_scale(peer, stream)
            copy_schedule_marker[(1,)](self.marker, num_warps=1)

    def overlap(self, attention):
        main = torch.cuda.current_stream()
        # Fork before attention: neither branch depends on the other's work.
        # Recording attention first changes node order, not required edges.
        self.aux.wait_stream(main)
        if self.attention_first:
            x = attention()
        with torch.cuda.stream(self.aux):
            self.prepare()
        if not self.attention_first:
            x = attention()
        if not self.late_join:
            main.wait_stream(self.aux)
            return self.project(x)
        values, scales = flashinfer_fp8_blockscale_quantize_prepacked(x, 128)
        main.wait_stream(self.aux)
        return flashinfer_mm_fp8_blockscale(
            values,
            self.weight,
            scales,
            self.scales,
            torch.bfloat16,
            alpha=None,
            block_size=[128, 128],
            out=None,
            prepacked_scales=True,
            original_m=x.shape[0],
        )


class RawContiguousCopy(DirectCopyEngineWeightPrefetch):
    """Use the direct CUDA binding with the original contiguous copy layout."""

    def __init__(self, group, weight_cpu, scales_cpu, copy_kind):
        super().__init__(group, weight_cpu, scales_cpu)
        self.copy_kind = copy_kind
        self.gathered = torch.empty(
            self.size * self.payload.numel(), device="cuda", dtype=torch.uint8
        )
        self.destinations = self.gathered.chunk(self.size)
        self.scratch_bytes += self.gathered.numel()

    def gather(self):
        stream = torch.cuda.current_stream().cuda_stream
        for step in range(self.size):
            peer = (self.rank - step) % self.size
            status = self.copy_bytes(
                self.destinations[peer].data_ptr(),
                self.sources[peer].data_ptr(),
                self.payload.numel(),
                self.copy_kind,
                stream,
            )
            if status:
                raise RuntimeError(f"cudaMemcpyAsync failed: {status}")

    def restore(self):
        WeightPrefetch.restore(self)


def capture_debug(call, steps, dot_path):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        for _ in range(steps):
            output = call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    if dist.get_rank() == 0:
        graph.debug_dump(str(dot_path))
    return graph, output


def check_shared_scratch(state, baseline, attention, mapping, group, model, dot_path):
    """Alternate two real layers through one scratch set, including graph replay.

    Repeating one immutable layer alone cannot detect accidentally retaining a
    previous layer's local weight or scales in the reusable full-weight buffer.
    """
    weight, scales, quant = load_projection(str(model), 1)
    (other_baseline, _), unused = make_linears(mapping, weight, scales, quant)
    del unused
    other = PrefetchScheduleOptimization(group, weight, scales, state.optimization_mode)
    other.weight = state.weight
    other.scales = state.scales

    def pair():
        return state.overlap(attention), other.overlap(attention)

    graph, output = capture_debug(pair, 2, dot_path)
    for _ in range(3):
        attention.qkv.add_(0.015625)
        expected = (
            baseline(attention())[0].clone(),
            other_baseline(attention())[0].clone(),
        )
        state.poison_copied_operands()
        eager = pair()
        for actual, reference in zip(eager, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        state.poison_copied_operands()
        graph.replay()
        for actual, reference in zip(output, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            state.weight.view(torch.uint8).cpu(),
            weight.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(state.scales.cpu(), scales.t(), rtol=0, atol=0)
    graph.reset()
    torch.cuda.synchronize()
    dist.barrier()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--replays", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world >= 4 and world % 4 == 0
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
    for first in range(0, world, 4):
        candidate = dist.new_group(list(range(first, first + 4)), backend="nccl")
        if first <= rank < first + 4:
            group = candidate
    assert group is not None
    weight, scales, quant = load_projection(str(args.model), 0)
    (baseline, _), unused = make_linears(mapping, weight, scales, quant)
    del unused
    attention = KdaDecode(args.model, args.rows, rank)
    states = {}
    for mode in args.modes:
        if (
            mode in SCHEDULE_MODES
            or mode in LOCAL_COPY_MODES
            or mode in LOCAL_TUNING_MODES
            or mode in REMOTE_STREAM_MODES
        ):
            states[mode] = PrefetchScheduleOptimization(group, weight, scales, mode)
        elif mode == "contiguous":
            states[mode] = CopyEngineWeightPrefetch(group, weight, scales, "immutable")
        elif mode in ("contiguous_raw_d2d", "contiguous_raw_default"):
            states[mode] = RawContiguousCopy(
                group, weight, scales, 3 if mode == "contiguous_raw_d2d" else 4
            )
        else:
            states[mode] = ScaleCopyDiagnostic(group, weight, scales, mode)
    reference = baseline(attention())[0].clone()
    for state in states.values():
        state.prepare()
        torch.testing.assert_close(
            state.weight.view(torch.uint8).cpu(),
            weight.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(state.scales.cpu(), scales.t(), rtol=0, atol=0)
        torch.testing.assert_close(state.overlap(attention), reference, rtol=0, atol=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graphs, outputs = {}, {}
    calls = {"replicated": lambda: baseline(attention())[0]}
    calls.update(
        {name: partial(state.overlap, attention) for name, state in states.items()}
    )
    for name, call in calls.items():
        graphs[name], outputs[name] = capture_debug(
            call, args.steps, args.output.with_name(f"{args.output.stem}-{name}.dot")
        )
        if rank == 0:
            print(f"Captured {name}", flush=True)
    for iteration in range(3):
        attention.qkv.add_(0.03125)
        reference = baseline(attention())[0].clone()
        for name, state in states.items():
            if isinstance(state, ScaleCopyDiagnostic):
                state.poison_copied_operands()
            else:
                state.gathered.zero_()
                state.weight.zero_()
                state.scales.fill_(float("nan"))
            if rank % 4 == iteration:
                torch.cuda._sleep(300000)
            graphs[name].replay()
            torch.testing.assert_close(outputs[name], reference, rtol=0, atol=0)
    shared_scratch_checks = []
    for name, state in states.items():
        if (
            not isinstance(state, PrefetchScheduleOptimization)
            or not state.local_copy_ctas
        ):
            continue
        check_shared_scratch(
            state,
            baseline,
            attention,
            mapping,
            group,
            args.model,
            args.output.with_name(f"{args.output.stem}-{name}-shared-scratch.dot"),
        )
        shared_scratch_checks.append(name)
        if rank == 0:
            args.output.with_name(f"{args.output.stem}-{name}.ptx").write_text(
                state.local_copy_kernel.asm["ptx"]
            )
    timings = measure(graphs, args.steps, args.samples, args.replays)
    result = {
        "world_size": world,
        "rows_per_rank": args.rows,
        "weight_tp_size": 4,
        "steps": args.steps,
        "samples": args.samples,
        "replays": args.replays,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "sm_count": torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count,
        "scope": "KDA attention and O projection; omitted operands retained for diagnostics",
        "correctness": "Exact reconstruction, eager output, changing-input/poisoned-scratch graph replay and delayed peers passed",
        "two_real_layers_shared_scratch": (
            "passed" if shared_scratch_checks else "not run"
        ),
        "shared_scratch_checked_modes": shared_scratch_checks,
        "median_us_max_rank": {k: statistics.median(v) for k, v in timings.items()},
        "samples_us_max_rank": timings,
    }
    if rank == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if args.profile:
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        dist.barrier()
        torch.cuda.synchronize()
        for name, graph in graphs.items():
            with torch.cuda.nvtx.range(f"{name}_TP4_C{args.rows}"):
                graph.replay()
                torch.cuda.synchronize()
        # The first rank stopping capture must not truncate a slower peer.
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    for graph in graphs.values():
        graph.reset()
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("WEIGHT_SCALE_COPY_DIAGNOSTIC_PASSED", flush=True)


if __name__ == "__main__":
    main()
