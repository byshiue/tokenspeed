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

"""Capture baseline/candidate Nsight or IKET profiles sequentially."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tool", choices=("ncu", "iket"), required=True)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parent
ncu = shutil.which("ncu")
if args.tool == "ncu" and ncu is None:
    parser.error("Nsight Compute must be available as ncu on PATH")
metrics = ["gpu__time_duration.sum", "smsp__pcsamp_sample_count"]
metrics += [
    "smsp__pcsamp_warps_issue_stalled_" + name
    for name in (
        "barrier",
        "long_scoreboard",
        "short_scoreboard",
        "wait",
        "membar",
        "not_selected",
    )
]
for label, kernel in (("baseline", args.baseline), ("optimized", args.candidate)):
    print(args.tool, label, kernel, flush=True)
    common = [
        "--model",
        str(args.model),
        "--kernel",
        str(kernel),
        "--output",
        str(args.output_dir / f"{args.tool}-{label}-correctness.json"),
        "--warmups",
        "3",
        "--tile-m",
        "128",
    ]
    if args.tool == "ncu":
        command = [
            ncu,
            "--target-processes",
            "application-only",
            "--profile-from-start",
            "off",
            "--replay-mode",
            "application",
            "--launch-skip",
            "2",
            "--launch-count",
            "1",
            "--clock-control",
            "none",
            "--cache-control",
            "none",
            "--section",
            "LaunchStats",
            "--metrics",
            ",".join(metrics),
            "--export",
            str(args.output_dir / f"ncu-{label}"),
            sys.executable,
            str(root / "ncu_single_rank.py"),
            *common,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "iket.cli.main",
            "--output-dir",
            str(args.output_dir / f"iket-coarse-{label}"),
            "--log-level",
            "warn",
            "profile",
            "--postprocess",
            "all",
            "--keep",
            "--max-ts-cnt-per-warp",
            "64",
            "--",
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(root / "capture_graph.py"),
            *common,
        ]
    with (args.output_dir / f"{args.tool}-{label}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
