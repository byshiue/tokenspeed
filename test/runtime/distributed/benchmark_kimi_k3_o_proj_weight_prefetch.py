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

"""Real-weight KDA O-projection benchmark: replicated, compute TP, weight-only TP.

This experiment does not change serving. Weight-only TP gathers FP8 codes and
scales on an auxiliary stream, restores the full matrix, then uses the same
quantizer/GEMM as the replicated path. One reusable full-weight workspace
models scratch shared by sequential layers, not persistent weights per layer.
"""

import argparse
import gc
import json
import os
import statistics
from pathlib import Path
from test.runtime.distributed.kimi_k3_o_proj_helpers import (
    dep_mapping,
    load_projection,
    make_linears,
)

import torch
import torch.distributed as dist
from safetensors import safe_open
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.kda import try_kda_fused_paged_decode
from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_blockscale
from tokenspeed_kernel.ops.gemm.fp8_utils import (
    flashinfer_fp8_blockscale_quantize_prepacked,
)

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.dp_row_parallel_linear import (
    ProjectionWorkspace,
    initialize_projection_group,
    projection_mapping,
)


@triton.jit
def restore_weight_and_scales(
    packed,
    weight,
    scales,
    N: tl.constexpr,
    K: tl.constexpr,
    P: tl.constexpr,
    PAYLOAD_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy uint32 packets bit-exactly, undoing rank-major K-shard gathering."""
    pid = tl.program_id(0)
    weight_words = N * K // 4
    weight_ctas = tl.cdiv(weight_words, BLOCK)
    i = tl.arange(0, BLOCK)
    if pid < weight_ctas:
        dst = pid * BLOCK + i
        n = dst // (K // 4)
        k = dst % (K // 4)
        peer = k // (K // P // 4)
        src = peer * PAYLOAD_WORDS + n * (K // P // 4) + k % (K // P // 4)
        bits = tl.load(packed + src, dst < weight_words, other=0)
        tl.store(weight + dst, bits, dst < weight_words)
    else:
        # Each K shard stores scales already in the GEMM's MN-major layout.
        dst = (pid - weight_ctas) * BLOCK + i
        shard_scales = (K // P // 128) * (N // 128)
        peer = dst // shard_scales
        src = peer * PAYLOAD_WORDS + N * (K // P) // 4 + dst % shard_scales
        bits = tl.load(packed + src, dst < (K // 128) * (N // 128), other=0)
        tl.store(scales + dst, bits, dst < (K // 128) * (N // 128))


class WeightPrefetch:
    """Shard at load time; gather and release a full-weight scratch lease per call."""

    def __init__(self, group, weight_cpu, scales_cpu):
        self.group = group
        self.size = group.size()
        self.rank = group.rank()
        self.n, self.k = weight_cpu.shape
        assert self.k % (self.size * 128) == 0
        shard_k = self.k // self.size
        # CPU slicing means this variant never uploads a full persistent weight.
        w = weight_cpu[:, self.rank * shard_k : (self.rank + 1) * shard_k].contiguous()
        s = scales_cpu[:, self.rank * shard_k // 128 : (self.rank + 1) * shard_k // 128]
        s = s.t().contiguous()
        self.payload = torch.cat(
            (w.view(torch.uint8).flatten(), s.view(torch.uint8).flatten())
        ).cuda()
        self.gathered = torch.empty(
            self.size * self.payload.numel(), device="cuda", dtype=torch.uint8
        )
        self.weight = torch.empty(
            (self.n, self.k), device="cuda", dtype=weight_cpu.dtype
        )
        self.scales = torch.empty(
            (self.k // 128, self.n // 128), device="cuda", dtype=torch.float32
        )
        self.aux = torch.cuda.Stream()
        self.shard_bytes = self.payload.numel()
        self.scratch_bytes = (
            self.gathered.numel() + self.weight.numel() + self.scales.numel() * 4
        )

    def gather(self):
        dist.all_gather_into_tensor(
            self.gathered, self.payload, group=self.group, async_op=False
        )

    def restore(self):
        count = triton.cdiv(self.n * self.k // 4, 1024) + triton.cdiv(
            self.scales.numel(), 1024
        )
        restore_weight_and_scales[(count,)](
            self.gathered.view(torch.int32),
            self.weight.view(torch.int32),
            self.scales.view(torch.int32),
            self.n,
            self.k,
            self.size,
            self.payload.numel() // 4,
            1024,
        )

    def prepare(self):
        self.gather()
        self.restore()

    def project(self, x):
        values, scales = flashinfer_fp8_blockscale_quantize_prepacked(x, 128)
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

    def serial(self, x):
        self.prepare()
        return self.project(x)

    def overlap(self, attention):
        main = torch.cuda.current_stream()
        # This dependency also releases the previous call's scratch lease:
        # its GEMM must finish reading before a new gather overwrites storage.
        self.aux.wait_stream(main)
        with torch.cuda.stream(self.aux):
            self.prepare()
        x = attention()
        main.wait_stream(self.aux)
        return self.project(x)


class KdaDecode:
    """The production conv/gate/recurrent/norm fusion with real layer-0 weights.

    Read pages and write pages are distinct so every timed call has identical
    inputs while still performing production state reads and writes.
    """

    def __init__(self, model, rows, rank):
        index = json.loads((model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        prefix = "language_model.model.layers.0.self_attn."

        def load(leaf):
            name = prefix + leaf
            with safe_open(model / index[name], framework="pt", device="cpu") as f:
                value = f.get_tensor(name)
            if value.dtype == torch.float8_e4m3fn:
                scale_name = name.removesuffix(".weight") + ".weight_scale"
                with safe_open(
                    model / index[scale_name], framework="pt", device="cpu"
                ) as f:
                    scale = f.get_tensor(scale_name).reshape(
                        triton.cdiv(value.shape[0], 128),
                        triton.cdiv(value.shape[1], 128),
                    )
                value = value.float() * scale.repeat_interleave(
                    128, 0
                ).repeat_interleave(128, 1)
            return value

        config = json.loads((model / "config.json").read_text())["text_config"]
        linear_config = config["linear_attn_config"]
        self.rows = rows
        self.heads, self.dim = linear_config["num_heads"], linear_config["head_dim"]
        self.lower_bound = linear_config["gate_lower_bound"]
        self.norm_eps = config["rms_norm_eps"]
        k = self.heads * self.dim
        torch.manual_seed(413 + rank)
        self.conv = torch.cat(
            [load(f"{x}_conv1d.weight").squeeze(1) for x in ("q", "k", "v")]
        ).to("cuda", torch.bfloat16)
        self.fb = load("f_b_proj.weight").to("cuda", torch.bfloat16)
        self.a_log = load("A_log").flatten()[: self.heads].to("cuda", torch.float32)
        self.dt = load("dt_bias").flatten().to("cuda", torch.float32)
        self.norm = load("o_norm.weight").flatten().to("cuda", torch.bfloat16)
        self.qkv = torch.randn(rows, 3 * k, device="cuda", dtype=torch.bfloat16)
        self.fa = (
            torch.randn(rows, self.fb.shape[1], device="cuda", dtype=torch.bfloat16)
            * 0.1
        )
        self.beta = torch.randn(rows, self.heads, device="cuda", dtype=torch.bfloat16)
        self.gate = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        self.conv_pool = (
            torch.randn(2 * rows, 3 * k, 3, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        self.state = (
            torch.randn(
                2 * rows,
                self.heads,
                self.dim,
                self.dim,
                device="cuda",
                dtype=torch.float32,
            )
            * 0.01
        )
        self.read = torch.arange(rows, device="cuda", dtype=torch.int32)
        self.write = self.read + rows
        self.cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32)

    def __call__(self):
        result = try_kda_fused_paged_decode(
            self.qkv,
            self.conv,
            self.conv_pool,
            self.fa,
            self.fb,
            self.beta,
            self.a_log,
            self.dt,
            state_pool=self.state,
            read_indices=self.read,
            write_indices=self.write,
            num_heads=self.heads,
            head_dim=self.dim,
            cu_seqlens=self.cu,
            lower_bound=self.lower_bound,
            output_gate=self.gate,
            norm_weight=self.norm,
            norm_eps=self.norm_eps,
            recurrent_layout="v_major",
            override="triton_nvidia_kda_fused_paged_decode",
            solution=None,
        )
        assert result is not None and result.output_norm_applied
        return result.out.reshape(self.rows, self.heads * self.dim)


def capture(call, steps):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(steps):
            output = call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph, output


def measure(graphs, steps, samples, replays):
    timings = {key: [] for key in graphs}
    # Alternate order to avoid giving every cold/warm or clock effect to one mode.
    for sample in range(samples):
        order = list(graphs) if sample % 2 == 0 else list(reversed(graphs))
        for name in order:
            dist.barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            latency = torch.tensor(
                start.elapsed_time(end) * 1000 / (steps * replays),
                device="cuda",
                dtype=torch.float32,
            )
            dist.all_reduce(latency, op=dist.ReduceOp.MAX)
            timings[name].append(latency.item())
    return timings


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--replays", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == 16 and all(r > 0 and r % 4 == 0 for r in args.rows)
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
    parallel = projection_mapping(rank, world, 4)
    initialize_projection_group(parallel)
    group4 = pg_manager.get_process_group("nccl", parallel.tp_group)
    weight_cpu, scales_cpu, quant = load_projection(str(args.model), 0)
    n, k = weight_cpu.shape
    assert (n, k) == (7168, 12288)
    (baseline, _), (linear4, exchange4) = make_linears(
        mapping, weight_cpu, scales_cpu, quant
    )
    workspace = ProjectionWorkspace(
        max(args.rows), k, torch.bfloat16, torch.device("cuda")
    )
    workspace.initialize_a2a(parallel, [k], backend="tokenspeed_a2a_lamport")
    workspace.initialize_reduce_scatter(parallel, [n], backend="trtllm_lamport")
    exchange4.workspace = workspace
    prefetch = {
        "tp4_weight": WeightPrefetch(group4, weight_cpu, scales_cpu),
        "tp16_weight": WeightPrefetch(dist.group.WORLD, weight_cpu, scales_cpu),
    }
    # Validate restoration, including the last shard, scales and consecutive
    # overwrites. No full-weight cache may bypass a later gather.
    for state in prefetch.values():
        state.prepare()
        torch.testing.assert_close(
            state.weight.view(torch.uint8).cpu(),
            weight_cpu.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            state.scales.cpu(), scales_cpu.t().contiguous(), rtol=0, atol=0
        )
    full_bytes = weight_cpu.numel() + scales_cpu.numel() * 4
    result = {
        "world_size": world,
        "weight_shape_NK": [n, k],
        "weight_dtype": str(weight_cpu.dtype),
        "weight_scale_dtype": str(scales_cpu.dtype),
        "torch": torch.__version__,
        "nccl": torch.cuda.nccl.version(),
        "gpu": torch.cuda.get_device_name(),
        "nccl_launch_env": {
            name: os.environ.get(name)
            for name in ("NCCL_NTHREADS", "NCCL_LL128_NTHREADS", "NCCL_MAX_CTAS")
        },
        "scope": "one real-weight KDA layer component, seeded activations/states; not full-model E2E",
        "memory": {
            "replicated_weight_and_scales_bytes_per_layer": full_bytes,
            "tp4_compute_weight_and_scales_bytes_per_layer": full_bytes // 4,
            **{
                name: {
                    "persistent_shard_bytes_per_layer": state.shard_bytes,
                    "reusable_scratch_bytes": state.scratch_bytes,
                    "allgather_full_payload_bytes_per_rank": state.gathered.numel(),
                    "remote_payload_bytes_per_rank": state.shard_bytes
                    * (state.size - 1),
                }
                for name, state in prefetch.items()
            },
            "note": "Logical tensor accounting, not full-model peak. Full-weight scratch is reused by serial layers and is not cudaFree'd during graph replay; benchmark retains all variants for paired timing. NCCL private workspace and allocator reservations are excluded.",
        },
        "cases": [],
    }
    for rows in args.rows:
        attention = KdaDecode(args.model, rows, rank)
        x = attention()
        counts = [rows] * world
        reference = baseline(x)[0]
        assert torch.isfinite(reference).all()
        errors = {}
        actual4 = exchange4.forward(x, linear4, counts)
        error = (actual4.float() - reference.float()).norm() / reference.float().norm()
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        assert error < 0.015, error
        errors["tp4_compute_relative_l2"] = error.item()
        for name, state in prefetch.items():
            state.weight.zero_()
            state.scales.fill_(float("nan"))
            torch.testing.assert_close(state.serial(x), reference, rtol=0, atol=0)
            torch.testing.assert_close(
                state.overlap(attention), reference, rtol=0, atol=0
            )
            errors[name + "_bit_exact"] = True
        xq, xs = flashinfer_fp8_blockscale_quantize_prepacked(x, 128)
        state4 = prefetch["tp4_weight"]
        calls = {
            "attention": attention,
            "dep16_projection": lambda: baseline(x)[0],
            "tp4_compute_projection": lambda: exchange4.forward(x, linear4, counts),
            "dep16_attention_projection": lambda: baseline(attention())[0],
            "tp4_compute_attention_projection": lambda: exchange4.forward(
                attention(), linear4, counts
            ),
            "quantize": lambda: flashinfer_fp8_blockscale_quantize_prepacked(x, 128)[0],
            "full_gemm": lambda: flashinfer_mm_fp8_blockscale(
                xq,
                state4.weight,
                xs,
                state4.scales,
                torch.bfloat16,
                alpha=None,
                block_size=[128, 128],
                out=None,
                prepacked_scales=True,
                original_m=rows,
            ),
        }
        for name, state in prefetch.items():
            calls[name + "_allgather"] = state.gather
            calls[name + "_restore"] = state.restore
            calls[name + "_prepare"] = state.prepare
            calls[name + "_projection_serial"] = lambda s=state: s.serial(x)
            calls[name + "_attention_projection_serial"] = lambda s=state: s.serial(
                attention()
            )
            calls[name + "_attention_projection_prefetch"] = lambda s=state: s.overlap(
                attention
            )
        graphs, outputs = {}, {}
        for name, call in calls.items():
            graphs[name], outputs[name] = capture(call, args.steps)
        # Capture/replay correctness must observe changing inputs and overwrite
        # poisoned weights, not simply validate an already-ready startup buffer.
        for _ in range(3):
            attention.qkv.add_(0.03125)
            changed_reference = baseline(attention())[0].clone()
            for name, state in prefetch.items():
                state.weight.zero_()
                state.scales.fill_(float("nan"))
                graphs[name + "_attention_projection_prefetch"].replay()
                torch.testing.assert_close(
                    outputs[name + "_attention_projection_prefetch"],
                    changed_reference,
                    rtol=0,
                    atol=0,
                )
        # All projection-only inputs remain the fixed x; pipeline calls consume
        # the same mutated attention inputs for every variant.
        timings = measure(graphs, args.steps, args.samples, args.replays)
        entry = {
            "rows_per_rank": rows,
            "global_requests": rows * world,
            "gemm_MNK_dep16_and_weight_only": [rows, n, k],
            "gemm_MNK_tp4_compute": [4 * rows, n, k // 4],
            "correctness": errors,
            "samples_us_max_rank": timings,
            "median_us_max_rank": {
                key: statistics.median(values) for key, values in timings.items()
            },
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
            for name in (
                "dep16_attention_projection",
                "tp4_compute_attention_projection",
                "tp4_weight_attention_projection_prefetch",
                "tp16_weight_attention_projection_prefetch",
            ):
                with torch.cuda.nvtx.range(f"{name}_C{rows}perRank_{args.steps}steps"):
                    graphs[name].replay()
                    torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()
        del (
            graphs,
            outputs,
            calls,
            attention,
            x,
            xq,
            xs,
            reference,
            actual4,
            changed_reference,
        )
        gc.collect()
    torch.cuda.synchronize()
    workspace.close()
    dist.destroy_process_group()
    if rank == 0:
        print("WEIGHT_PREFETCH_COMPARISON_PASSED", flush=True)


if __name__ == "__main__":
    main()
