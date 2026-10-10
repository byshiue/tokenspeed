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

"""Capture one warmed collective launch without kernel-local replay."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from experiment import FullExperiment, initialize, reference


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--tile-m", type=int, required=True)
    args = parser.parse_args()
    rank, world, device, models = initialize(args.model)
    linear = models[1].model.layers[0].self_attn.qkvgb_proj
    torch.manual_seed(51 + rank)
    x = torch.randn((128, linear.weight.shape[1]), device=device)
    out = torch.empty((128, linear.output_size), device=device)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if rank == 0:
        print("MODEL_READY", flush=True)
    experiment = FullExperiment(
        x,
        linear.weight,
        linear.weight_scale_inv,
        out,
        rank,
        world,
        sms,
        "tma",
        args.kernel,
        args.tile_m,
    )
    if rank == 0:
        print("COMPILE_READY", flush=True)
    ctx = reference.context(rank, [128] * world)
    expected = linear(x, ctx=ctx)[0].clone()
    for _ in range(args.warmups):
        experiment(x, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            experiment(x, out)
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print("PROFILE_START", flush=True)
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("fused")
    graph.replay()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    torch.testing.assert_close(out[:, : expected.shape[1]], expected, rtol=0, atol=0)
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        args.output.write_text(
            json.dumps(
                {
                    "capture": "three fused graph nodes per rank; use final node for timelines",
                    "kernel": str(args.kernel),
                    "warmups": args.warmups,
                    "exact_output": True,
                    "ranks": world,
                    "sms": sms,
                },
                indent=2,
            )
            + "\n"
        )
        print("PROFILE_DONE exact outputs on all ranks", flush=True)
    del graph
    del experiment
    for model in models:
        reference.release_dp_linear_communication(model)
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
