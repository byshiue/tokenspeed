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

"""Validate and compare selected candidates using one loaded checkpoint."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import benchmark
import torch
import torch.distributed as dist


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, action="append", required=True)
    parser.add_argument("--baseline-kernel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--tile-m", type=int, required=True)
    args = parser.parse_args()
    rank, world, device, models = benchmark.initialize(args.model)
    for kernel in args.kernel:
        directory = args.output_dir / kernel.stem
        directory.mkdir(parents=True, exist_ok=True)
        case = SimpleNamespace(
            kernel=kernel,
            baseline_kernel=args.baseline_kernel,
            output=directory / "result.json",
            exchange="tma",
            trace=args.trace,
            tile_m=args.tile_m,
        )
        if rank == 0:
            print("CANDIDATE", str(kernel), flush=True)
        benchmark.run_case(case, rank, world, device, models)
    for model in models:
        benchmark.reference.release_dp_linear_communication(model)
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
