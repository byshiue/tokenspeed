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

"""One-kernel quantize/exchange, FP8 blockwise GEMM, and inverse A2A experiment."""

import argparse
import json
import os
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass.cute.runtime import from_dlpack
from experiment import FullExperiment
from support import initialize, measure, module_from_path, reference
from tokenspeed_kernel.ops.quantization import quantize_fp8


def relative_error(actual, expected):
    value = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-10)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--exchange", choices=("pull", "push", "ilp", "ilp8", "tma"), required=True
    )
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--baseline-kernel", type=Path, required=True)
    parser.add_argument("--tile-m", type=int, required=True)
    args = parser.parse_args()
    rank, world, device, models = initialize(args.model)
    run_case(args, rank, world, device, models)
    for model in models:
        reference.release_dp_linear_communication(model)
    dist.destroy_process_group()


@torch.no_grad()
def run_case(args, rank, world, device, models):
    baseline, linear = [m.model.layers[0].self_attn.qkvgb_proj for m in models]
    torch.manual_seed(51 + rank)
    x = torch.randn((128, linear.weight.shape[1]), device=device)
    out = torch.empty((128, linear.output_size), device=device)
    workspace, backend = linear.projection_workspace, linear.comm_backend
    ctx = reference.context(rank, [128] * world)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if rank == 0:
        print(f"Compiling all-stage kernel with {sms} CTAs", flush=True)
    experiment = FullExperiment(
        x,
        linear.weight,
        linear.weight_scale_inv,
        out,
        rank,
        world,
        sms,
        args.exchange,
        args.kernel,
        args.tile_m,
    )
    if rank == 0:
        print("Executing all-stage kernel", flush=True)
    experiment(x, out)
    torch.cuda.synchronize()
    expected = linear(x, ctx=ctx)[0]
    error = relative_error(out[:, : expected.shape[1]], expected)
    gathered, _ = backend.projection_all_gather(x, 128, False, workspace)
    expected_a, expected_sa = quantize_fp8(
        gathered, granularity="token_group", group_size=128, scale_encoding="float32"
    )
    quant_error = {
        "fp8_mismatches": int(
            (experiment.a.view(torch.uint8) != expected_a.view(torch.uint8)).sum()
        ),
        "scale_relative_max": float(
            ((experiment.sa - expected_sa).abs() / expected_sa).max()
        ),
    }
    if rank == 0:
        print(f"Initial relative L2 {error}; quantization {quant_error}", flush=True)
    assert error < 0.015, error
    torch.testing.assert_close(
        experiment.a.view(torch.uint8), expected_a.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(experiment.sa, expected_sa, rtol=0, atol=0)
    torch.testing.assert_close(out[:, : expected.shape[1]], expected, rtol=0, atol=0)

    def full():
        result = torch.empty_like(out)
        return experiment(x, result)

    for _ in range(3):
        full()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        retained = full()
        second = full()
    replay_errors = []
    for seed in range(16):
        torch.manual_seed(170 + rank + seed * world)
        x.normal_()
        if seed == 0:
            x.zero_()
        if seed == 1:
            x[:, ::2] = 0
        if seed == 12:
            x[:, ::2] = -0.0
            x[:, 1::2] = 0.0
        if seed == 13:
            x.mul_(2.0**-50)
        if seed == 14:
            x.mul_(2.0**10)
        if seed == 15:
            # Repeated BF16 mantissas exercise FP8 rounding at varied maxima.
            values = (
                torch.arange(x.shape[1], device=device, dtype=torch.float32) % 255 - 127
            ) / 128
            x.copy_(values.to(x.dtype).expand_as(x))
        # Deliberate host launch skew; no timed samples include this delay.
        if rank == seed % world:
            import time

            time.sleep(0.002)
        graph.replay()
        actual_expected = linear(x, ctx=ctx)[0]
        gathered, _ = backend.projection_all_gather(x, 128, False, workspace)
        expected_a, expected_sa = quantize_fp8(
            gathered,
            granularity="token_group",
            group_size=128,
            scale_encoding="float32",
        )
        if not torch.equal(
            experiment.a.view(torch.uint8), expected_a.view(torch.uint8)
        ):
            mismatch = experiment.a.view(torch.uint8) != expected_a.view(torch.uint8)
            print(
                {
                    "rank": rank,
                    "fp8_replay": seed,
                    "actual_bits": experiment.a.view(torch.uint8)[mismatch][
                        :8
                    ].tolist(),
                    "expected_bits": expected_a.view(torch.uint8)[mismatch][
                        :8
                    ].tolist(),
                    "source_bits": x.view(torch.int16)[0, :8].tolist(),
                    "gathered_bits": gathered.view(torch.int16)[0, :8].tolist(),
                },
                flush=True,
            )
        torch.testing.assert_close(
            experiment.a.view(torch.uint8), expected_a.view(torch.uint8), rtol=0, atol=0
        )
        if not torch.equal(experiment.sa, expected_sa):
            mismatch = experiment.sa != expected_sa
            print(
                {
                    "rank": rank,
                    "scale_replay": seed,
                    "actual_scales": experiment.sa[mismatch][:8].tolist(),
                    "expected_scales": expected_sa[mismatch][:8].tolist(),
                    "actual_bits": experiment.sa.view(torch.int32)[mismatch][
                        :8
                    ].tolist(),
                    "expected_bits": expected_sa.view(torch.int32)[mismatch][
                        :8
                    ].tolist(),
                },
                flush=True,
            )
        torch.testing.assert_close(experiment.sa, expected_sa, rtol=0, atol=0)
        logical = actual_expected.shape[1]
        for tensor in (retained, second):
            current_error = relative_error(tensor[:, :logical], actual_expected)
            assert current_error < 0.015, (seed, current_error)
            torch.testing.assert_close(
                tensor[:, :logical], actual_expected, rtol=0, atol=0
            )
            replay_errors.append(current_error)
        saved = retained.clone()
        full()
        torch.testing.assert_close(retained, saved, rtol=0, atol=0)
    # Device-resident 64-bit epochs must keep working across the 32-bit boundary
    # without recapture or host-provided epoch arguments.
    # Restore the previous timing input so edge cases do not change the workload.
    torch.manual_seed(170 + rank + 11 * world)
    x.normal_()
    torch.cuda.synchronize()
    dist.barrier()
    for flag in (experiment.ready, experiment.ack, experiment.input_flags):
        flag.fill_(2**32 - 4)
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(20):
        graph.replay()
    expected = linear(x, ctx=ctx)[0]
    for tensor in (retained, second):
        torch.testing.assert_close(
            tensor[:, : expected.shape[1]], expected, rtol=0, atol=0
        )
    assert int(experiment.ready[0]) > 2**32
    original = FullExperiment(
        x,
        linear.weight,
        linear.weight_scale_inv,
        out,
        rank,
        world,
        sms,
        "tma",
        args.baseline_kernel,
        128,
    )

    def original_full():
        return original(x, torch.empty_like(out))

    calls = {
        "previous_fused": original_full,
        "tp1": lambda: baseline(x, ctx=ctx),
        "tp4": lambda: linear(x, ctx=ctx),
        "one_kernel": full,
    }
    timings = measure(calls, 5, 20, 10)
    torch.cuda.synchronize()
    # Slot zero is the rank arrival, slots 16 onward are CTA arrivals.
    # Reserved slot one may hold a phase-specific hierarchical completion.
    for flags in (experiment.ack, experiment.input_flags):
        assert torch.equal(experiment.ready[:1], flags[:1])
        assert torch.equal(experiment.ready[16 : 16 + sms], flags[16 : 16 + sms])
    result = {
        "active_clusters": sms,
        "kernel": str(args.kernel),
        "exchange": args.exchange,
        "relative_l2": error,
        "quantization": quant_error,
        "changed_input_graph_replays": 16,
        "replay_errors": replay_errors,
        "quantization_checked_each_replay": True,
        "edge_inputs": [
            "signed_zero",
            "small_magnitude",
            "large_magnitude",
            "bf16_mantissa_pattern",
        ],
        "retained_output_check": True,
        "rank_skew_check": True,
        "epoch_32bit_boundary_check": True,
        "epoch_after_benchmark": int(experiment.ready[0]),
        "used_epochs_agree": True,
        "timing": timings,
        "resources": experiment.resources,
        "baseline_resources": original.resources,
    }
    if args.trace:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(3):
                full()
            torch.cuda.synchronize()
        trace_path = args.output.with_name(f"full-trace-rank{rank}.json")
        prof.export_chrome_trace(str(trace_path))
        if rank == 0:
            trace = json.loads(trace_path.read_text())
            result["profiled_kernel_names"] = [
                e["name"] for e in trace["traceEvents"] if e.get("cat") == "kernel"
            ]
            assert len(result["profiled_kernel_names"]) == 3
        kernel_count = sum(
            e.get("cat") == "kernel"
            for e in json.loads(trace_path.read_text())["traceEvents"]
        )
        assert kernel_count == 3, (rank, kernel_count)
    if rank == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    torch.cuda.synchronize()
    dist.barrier()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        text = traceback.format_exc()
        print(text, flush=True)
        raise
