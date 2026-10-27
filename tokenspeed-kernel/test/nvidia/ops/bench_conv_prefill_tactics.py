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

"""Validate conservative selection against both tactics and the original kernel."""

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.conv._triton import prefill_candidate
from tokenspeed_kernel.ops.conv._triton.prefill_candidate import (
    PrefillCandidateConfig,
    PrefillTacticPolicy,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_conv_prefill_candidate import (
    BASELINE,
    BASELINE_PREFILL_SHA256,
    MAIN_LENGTHS,
    MeasuredGraph,
    baseline_prefill_sha256,
    case_for,
    clock_snapshot,
    summarize,
)

ROLLING = PrefillCandidateConfig("gluon_serial", 8, 256, 4)
PARALLEL = PrefillCandidateConfig("gluon_window_r64", 8, 512, 4)


def validation_shapes(policy):
    """Include both sides of each bound, balanced work and skewed work."""
    minimum = policy.min_parallel_tokens
    maximum = policy.max_parallel_tokens
    average = policy.min_tokens_per_request
    shapes = [
        [t]
        for t in (
            1,
            2,
            3,
            7,
            8,
            9,
            16,
            32,
            64,
            128,
            256,
            512,
            1023,
            1024,
            1025,
            1330,
            minimum - 1,
            minimum,
            minimum + 1,
            maximum - 1,
            maximum,
            maximum + 1,
        )
    ]
    shapes += [
        [1] * 128,
        [1] * 1024,
        [2] * 128,
        [8] * 128,
        [16] * 64,
        [32] * 32,
        [64] * 16,
        [128] * 8,
        [512] + [1] * 128,
        [1024] + [1] * 32,
        [32] * 31 + [31],
    ]
    for total in sorted({minimum, (minimum + maximum) // 2, maximum}):
        limit = max(1, total // average)
        for requests in sorted({max(1, limit - 1), limit, limit + 1}):
            quotient, extra = divmod(total, requests)
            shapes.append([quotient + 1] * extra + [quotient] * (requests - extra))
        shapes.append([total - limit + 1] + [1] * (limit - 1))
        shapes.append([total] + [0] * (limit - 1))
    shapes += MAIN_LENGTHS
    return list(dict.fromkeys(tuple(shape) for shape in shapes if sum(shape) > 0))


def measure_choices(case, policy, *, scope, pdl):
    configs = {
        "original_kernel": BASELINE,
        "rolling_tactic": ROLLING,
        "parallel_tactic": PARALLEL,
        "automatic": policy,
    }
    graphs = {
        name: MeasuredGraph(case, config, scope=scope, pdl=pdl)
        for name, config in configs.items()
    }
    for graph in graphs.values():
        graph.check_replay()
    if scope == "chain":
        for graph in graphs.values():
            graph.source.copy_(case.x)
    selected = policy.select(case.x.shape[1], case.slots.numel(), exact_workload=True)
    fixed = "parallel_tactic" if selected == PARALLEL else "rolling_tactic"
    binaries = {
        name: hashlib.sha256(graph.binary.asm["cubin"]).hexdigest()
        for name, graph in graphs.items()
    }
    assert binaries["automatic"] == binaries[fixed]
    values = {name: [] for name in graphs}
    rounds = {name: [] for name in graphs}
    names = list(graphs)
    for iteration in range(5):
        # Rotate the first graph and reverse alternate rounds to limit drift.
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        if iteration % 2:
            order = list(reversed(order))
        for name in order:
            measured = graphs[name].samples()
            values[name].extend(measured)
            rounds[name].append(statistics.median(measured))
    timings = {name: summarize(samples) for name, samples in values.items()}
    return {
        "scope": scope,
        "pdl": pdl,
        "selected": asdict(selected),
        "binary_matches_fixed_tactic": True,
        "binary_sha256": binaries,
        "timings": timings,
        "round_medians_us": rounds,
        "automatic_vs_rolling": timings["rolling_tactic"]["median_us"]
        / timings["automatic"]["median_us"],
        "automatic_vs_original": timings["original_kernel"]["median_us"]
        / timings["automatic"]["median_us"],
        "parallel_vs_rolling": timings["rolling_tactic"]["median_us"]
        / timings["parallel_tactic"]["median_us"],
        "round_speedups_vs_rolling": [
            a / b
            for a, b in zip(rounds["rolling_tactic"], rounds["automatic"], strict=True)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.min_speedup <= 1:
        parser.error("--min-speedup must reserve a positive margin above 1")
    policy = PrefillTacticPolicy(**json.loads(args.policy.read_text()))
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("the policy validation requires B300 or GB300 (SM103)")
    baseline_hash = baseline_prefill_sha256()
    if baseline_hash != BASELINE_PREFILL_SHA256:
        raise RuntimeError("the frozen baseline changed")
    report = {
        "status": "running",
        "policy": asdict(policy),
        "minimum_selected_speedup": args.min_speedup,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "baseline_prefill_sha256": baseline_hash,
        "selector_sha256": hashlib.sha256(
            Path(prefill_candidate.__file__).read_bytes()
        ).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_clocks": clock_snapshot(),
        "records": [],
        "pdl_chain": [],
        "timing": "5x100 interleaved CUDA graph event samples; reset outside timing; no L2 flush",
        "acceptance": "selected parallel: every round meets margin over rolling; fallback: identical CUBIN",
        "selection": "exact packed work only; capacity-only graphs fall back to rolling",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def measure(shape, scope, pdl):
        case = case_for(list(shape), 36864, 713)
        result = measure_choices(case, policy, scope=scope, pdl=pdl)
        result.update(lengths=list(shape), tokens=sum(shape), requests=len(shape))
        parallel = result["selected"]["implementation"] == PARALLEL.implementation
        # Identical fallback code can still show event timing jitter. Prove
        # equivalence by binary identity instead of interpreting noise as gain.
        result["passed"] = (
            (
                min(result["round_speedups_vs_rolling"]) >= args.min_speedup
                and result["automatic_vs_rolling"] >= args.min_speedup
            )
            if parallel
            else result["binary_matches_fixed_tactic"]
        )
        print(
            sum(shape),
            len(shape),
            scope,
            result["selected"]["implementation"],
            f"vs_rolling={result['automatic_vs_rolling']:.6f}",
            f"vs_original={result['automatic_vs_original']:.6f}",
            f"passed={result['passed']}",
            flush=True,
        )
        return result

    save()
    for shape in validation_shapes(policy):
        for scope in (
            ("kernel", "metadata") if list(shape) in MAIN_LENGTHS else ("kernel",)
        ):
            report["records"].append(measure(shape, scope, False))
            save()
    for shape in ([1], [8192]):
        report["pdl_chain"].append(measure(shape, "chain", True))
        save()
    report["final_clocks"] = clock_snapshot()
    report["status"] = (
        "complete"
        if all(row["passed"] for row in report["records"] + report["pdl_chain"])
        else "failed"
    )
    save()
    if report["status"] != "complete":
        raise RuntimeError("selected tactic did not pass the measured gain margin")


if __name__ == "__main__":
    main()
