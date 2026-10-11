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

"""Compare explicit fused TP4, automatic dispatch, ordinary TP4 and TP1."""

import argparse
import json
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
from support import initialize, measure, reference
from tokenspeed_kernel.ops.communication.cute_dsl import (
    cute_dsl_fused_tp4_projection,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8

from tokenspeed.runtime.layers.linear import LinearBase
from tokenspeed.runtime.utils.env import envs


def _fused_projection(linear, inputs, counts):
    """Benchmark the registered kernel even when automatic dispatch skips it."""
    out = inputs.new_empty((inputs.shape[0], linear.output_size))
    cute_dsl_fused_tp4_projection(
        linear.projection_workspace.fused_column,
        inputs,
        linear.weight,
        linear.weight_scale_inv,
        counts,
        out,
    )
    return out[:, : linear.logical_output_size]


def _owner_rows(value):
    counts = tuple(int(part) for part in value.split(","))
    if len(counts) != 4 or min(counts) < 0 or max(counts) == 0:
        raise argparse.ArgumentTypeError("expected four nonnegative row counts")
    return counts


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner-rows", type=_owner_rows, nargs="+", required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with envs.TOKENSPEED_FUSED_TP4_PROJECTION.override(False):
        rank, world, device, models = initialize(args.model)
    base_model = models[1]
    capacity = max(max(counts) for counts in args.owner_rows)
    reference.release_dp_linear_communication(base_model)
    with envs.TOKENSPEED_FUSED_TP4_PROJECTION.override(False):
        reference.prepare_dp_linear_communication(
            base_model, capacity, torch.bfloat16, reference.AutoBackend()
        )
    parallel, _ = reference.validate_projection_settings(base_model.mapping, 4, 1)
    fused_model = reference.ProjectionCheckpoint(
        base_model.config, base_model.mapping, base_model.quant_config, parallel, device
    )
    fused_model.load_weights(reference.checkpoint(args.model, device))
    for module in fused_model.modules():
        if isinstance(module, LinearBase) and module.quant_method is not None:
            module.quant_method.process_weights_after_loading(module)
    with envs.TOKENSPEED_FUSED_TP4_PROJECTION.override(True):
        reference.prepare_dp_linear_communication(
            fused_model, max(64, capacity), torch.bfloat16, reference.AutoBackend()
        )
    models.append(fused_model)
    tp1, tp4, fused = [model.model.layers[0].self_attn.qkvgb_proj for model in models]
    state = fused.projection_workspace.fused_column
    assert state is not None and tp4.projection_workspace.fused_column is None
    torch.testing.assert_close(
        fused.weight.view(torch.uint8), tp4.weight.view(torch.uint8), rtol=0, atol=0
    )
    result = {
        "scope": "Real checkpoint weights, isolated projection; not full-model E2E",
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "world": world,
        "gpu": torch.cuda.get_device_name(device),
        "sampling": {
            "alternating_samples": 5,
            "calls_per_graph": 20,
            "replays_per_sample": 10,
            "reduction": "maximum rank then median sample",
        },
        "results": [],
    }
    for counts in args.owner_rows:
        rows = max(counts)
        bucket = max(32, 1 << (rows - 1).bit_length())
        label = "-".join(str(n) for n in counts)
        ctx = reference.context(rank, list(counts))
        x = torch.empty((counts[rank], 7168), device=device, dtype=torch.bfloat16)
        errors = []
        dispatches = set()
        for seed in range(16):
            torch.manual_seed(170 + rank + seed * world)
            x.normal_()
            if seed == 0:
                x.zero_()
            elif seed == 1:
                x[:, ::2] = 0
            elif seed == 12:
                x[:, ::2] = -0.0
                x[:, 1::2] = 0.0
            elif seed == 13:
                x.mul_(2.0**-50)
            elif seed == 14:
                x.mul_(2.0**10)
            elif seed == 15:
                values = (
                    torch.arange(7168, device=device, dtype=torch.float32) % 255 - 127
                ) / 128
                x.copy_(values.to(x.dtype).expand_as(x))
            expected = tp4(x, ctx=ctx)[0]
            epoch = int(state.ready[0])
            actual = fused(x, ctx=ctx)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            dispatch = int(state.ready[0]) - epoch
            assert dispatch in (0, 1)
            dispatches.add(dispatch)
            actual = _fused_projection(fused, x, counts)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            gathered, _ = tp4.comm_backend.projection_all_gather(
                x, rows, False, tp4.projection_workspace
            )
            padded = gathered.new_zeros((4, bucket, 7168))
            padded[:, :rows].copy_(gathered.view(4, rows, 7168))
            values, scales = quantize_fp8(
                padded.view(4 * bucket, 7168),
                granularity="token_group",
                group_size=128,
                scale_encoding="float32",
            )
            torch.testing.assert_close(
                state.a[: bucket * 4].view(torch.uint8),
                values.view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(state.sa[: bucket * 4], scales, rtol=0, atol=0)
            errors.append(0.0)
        assert len(dispatches) == 1
        auto_fused = bool(dispatches.pop())
        torch.manual_seed(43 + rank)
        x.normal_()
        left, right = tp1(x, ctx=ctx)[0], tp4(x, ctx=ctx)[0]
        left = left[:, : right.shape[1]]
        relative_l2 = (
            left.float() - right.float()
        ).norm() / left.float().norm().clamp_min(1e-10)
        dist.all_reduce(relative_l2, op=dist.ReduceOp.MAX)
        assert float(relative_l2) < 0.015
        timings = measure(
            {
                "tp1": lambda: tp1(x, ctx=ctx),
                "tp4": lambda: tp4(x, ctx=ctx),
                "fused_tp4": lambda: _fused_projection(fused, x, counts),
                "auto_tp4": lambda: fused(x, ctx=ctx),
            },
            5,
            20,
            10,
        )
        record = {
            "owner_rows": counts,
            "bucket_rows": bucket,
            "automatic_dispatch": "fused" if auto_fused else "unfused",
            "exact_output_and_quantization_cases": len(errors),
            "tp4_vs_tp1_relative_l2": float(relative_l2),
            "timing": timings,
        }
        if args.trace:
            launches = {}
            for name, call in {
                "fused_tp4": lambda: _fused_projection(fused, x, counts),
                "auto_tp4": lambda: fused(x, ctx=ctx),
            }.items():
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for _ in range(3):
                        call()
                    torch.cuda.synchronize()
                trace = args.output.with_name(f"{name}-rows{label}-rank{rank}.json")
                prof.export_chrome_trace(str(trace))
                names = [
                    event["name"]
                    for event in json.loads(trace.read_text())["traceEvents"]
                    if event.get("cat") == "kernel"
                ]
                fused_count = sum("FusedBlockwiseGemmKernel" in n for n in names)
                if name == "fused_tp4" or auto_fused:
                    assert fused_count == len(names) == 3, names
                else:
                    assert names and fused_count == 0, names
                launches[name] = {"total": len(names), "fused": fused_count}
            record["profiled_kernels_per_rank"] = launches
        result["results"].append(record)
        if rank == 0:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    for flags in (state.ack, state.input_flags):
        torch.testing.assert_close(state.ready[16:], flags[16:], rtol=0, atol=0)
    for model in models:
        reference.release_dp_linear_communication(model)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
