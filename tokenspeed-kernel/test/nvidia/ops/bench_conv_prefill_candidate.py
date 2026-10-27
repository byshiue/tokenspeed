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

"""B300/GB300 local experiment; never changes serving kernel selection.

Run from the repository root with PYTHONPATH=tokenspeed-kernel/python.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import math
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.conv._triton import causal_conv1d
from tokenspeed_kernel.ops.conv._triton.prefill_candidate import PrefillCandidateConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from conv_prefill_utils import (
    BASELINE,
    MAIN_LENGTHS,
    assert_bitwise,
    check_case,
    make_case,
)

BASELINE_PREFILL_SHA256 = (
    "746237a6402c8064af7048fe92e9231da5a8fadec1d7614cf2d428a3e79cae27"
)


def baseline_prefill_sha256():
    # Freeze the tested prefill function and its decorators. Changes to the
    # separate decode kernel in the same source file do not change this baseline.
    source = inspect.getsource(causal_conv1d._causal_conv1d_fwd_kernel.fn)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@triton.jit(do_not_specialize=["count"])
def _projection_copy(
    Source, Dest, count, ENABLE_PDL: tl.constexpr, BLOCK: tl.constexpr
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        Dest + offsets,
        tl.load(Source + offsets, offsets < count, other=0),
        offsets < count,
    )


def tag(config):
    return f"{config.implementation}-m{config.block_m}-n{config.block_n}-w{config.num_warps}"


def case_for(lengths, channels, seed):
    torch.manual_seed(seed)
    case = make_case(
        lengths,
        channels,
        4,
        torch.bfloat16,
        device="cuda",
        strided=False,
        with_bias=False,
        pad=False,
    )
    case.history.fill_(True)
    # The benchmark uses the model's compact 3-token state. Unit tests also
    # cover wider state storage with untouched spare columns.
    case.initial_state = case.initial_state[:, :, :3].contiguous()
    return case


def dump_binary(binary, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for kind in ("ptx", "cubin", "ttgir"):
        contents = binary.asm[kind]
        path = directory / f"{name}.{kind}"
        path.write_bytes(contents if isinstance(contents, bytes) else contents.encode())
    ptx = binary.asm["ptx"]
    return {
        "registers_per_thread": binary.n_regs,
        "spills": binary.n_spills,
        "shared_bytes_per_cta": binary.metadata.shared,
        "ptx_shared_load_lines": sum("ld.shared" in line for line in ptx.splitlines()),
        "ptx_shared_store_lines": sum("st.shared" in line for line in ptx.splitlines()),
        "note": "PTX line counts are static instructions, not dynamic memory traffic.",
        "cubin_sha256": hashlib.sha256(binary.asm["cubin"]).hexdigest(),
    }


class MeasuredGraph:
    def __init__(self, case, config, *, scope, pdl):
        self.case = case
        self.config = config
        self.scope = scope
        self.pdl = pdl
        self.state = case.initial_state.clone()
        self.out = torch.empty_like(case.x)
        self.metadata = case.metadata(config.block_m)
        self.source = case.x.clone() if scope == "chain" else None
        # Keep event record nodes in the graph so each replay provides fresh
        # timestamps that can be read outside capture.
        self.begin = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        self.graph = torch.cuda.CUDAGraph()
        self.binary = None
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(10):
                self.state.copy_(case.initial_state)
                self.forward()
        torch.cuda.current_stream().wait_stream(self.stream)
        with torch.cuda.graph(self.graph, stream=self.stream):
            # Reset is outside the timed range. Both graphs get the same state
            # and the same reset-induced state-cache warming on every sample.
            self.state.copy_(case.initial_state)
            self.begin.record()
            self.forward()
            self.end.record()
        for _ in range(20):
            self.graph.replay()
        torch.cuda.synchronize()

    def forward(self):
        if self.scope == "chain":
            _projection_copy[(triton.cdiv(self.case.x.numel(), 1024),)](
                self.source,
                self.case.x,
                self.case.x.numel(),
                ENABLE_PDL=self.pdl,
                BLOCK=1024,
                **({"launch_pdl": True} if self.pdl else {}),
            )
        metadata = (
            self.case.metadata(self.config.block_m)
            if self.scope == "metadata"
            else self.metadata
        )
        self.binary = self.case.launch(
            self.config, self.state, self.out, metadata, silu=True, enable_pdl=self.pdl
        )

    def samples(self):
        values = []
        for _ in range(100):
            self.graph.replay()
            self.end.synchronize()
            values.append(self.begin.elapsed_time(self.end) * 1000)
        return values

    def check_replay(self):
        for _ in range(3):
            if self.source is not None:
                self.source.normal_()
            self.graph.replay()
            torch.cuda.synchronize()
            old_state = self.case.initial_state.clone()
            old_out = torch.empty_like(self.out)
            self.case.launch(
                BASELINE,
                old_state,
                old_out,
                self.case.metadata(8),
                silu=True,
                enable_pdl=False,
            )
            assert_bitwise(self.out, old_out)
            assert_bitwise(self.state, old_state)


def summarize(samples):
    return {
        "median_us": statistics.median(samples),
        "p95_us": statistics.quantiles(samples, n=100, method="inclusive")[94],
        "samples_us": samples,
    }


def measure_pair(case, config, *, scope, pdl):
    baseline = MeasuredGraph(case, BASELINE, scope=scope, pdl=pdl)
    candidate = MeasuredGraph(case, config, scope=scope, pdl=pdl)
    baseline.check_replay()
    candidate.check_replay()
    # Restore identical producer data after the changed-input replay checks.
    if scope == "chain":
        baseline.source.copy_(case.x)
        candidate.source.copy_(case.x)
    values = [[], []]
    rounds = [[], []]
    graphs = [baseline, candidate]
    for iteration in range(5):
        for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
            measured = graphs[index].samples()
            values[index].extend(measured)
            rounds[index].append(statistics.median(measured))
    result = {
        "baseline": summarize(values[0]),
        "candidate": summarize(values[1]),
        "round_medians_us": rounds,
        "scope": scope,
        "pdl": pdl,
    }
    result["speedup"] = (
        result["baseline"]["median_us"] / result["candidate"]["median_us"]
    )
    if scope == "kernel":
        useful_bytes = 2 * case.x.numel() * case.x.element_size()
        result["useful_input_output_bytes"] = useful_bytes
        result["bandwidth_note"] = (
            "One input read plus one output write; excludes halo, weights, state, "
            "and cache traffic. This is not a measured DRAM utilization."
        )
        for key in ("baseline", "candidate"):
            bandwidth = useful_bytes / (result[key]["median_us"] * 1e6)
            result[key]["effective_payload_TB_s"] = bandwidth
            result[key]["effective_payload_SOL_at_8_TB_s_percent"] = bandwidth / 8 * 100
    return result


def clock_snapshot():
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False}
    result = subprocess.run(
        [
            executable,
            "--query-gpu=name,uuid,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used",
            "--format=csv",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def ncu_command(config, output, shape):
    return [
        "ncu",
        "--profile-from-start",
        "off",
        "--set",
        "full",
        "--cache-control",
        "none",
        "--page",
        "raw",
        "--csv",
        "--log-file",
        str(output.with_suffix(".csv")),
        "--export",
        str(output),
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "profile",
        "--implementation",
        config.implementation,
        "--block-m",
        str(config.block_m),
        "--block-n",
        str(config.block_n),
        "--warps",
        str(config.num_warps),
        "--shape",
        str(shape),
        "--scope",
        "kernel",
        "--pdl",
        "off",
        "--output",
        str(output.with_suffix(".json")),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sweep", "refine", "profile"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", type=Path)
    parser.add_argument(
        "--implementation",
        choices=(
            "serial",
            "window",
            "direct",
            "gluon_serial",
            "gluon_direct",
            "gluon_window",
            "gluon_window_r64",
            "gluon_window_r48",
        ),
    )
    parser.add_argument("--block-m", type=int)
    parser.add_argument("--block-n", type=int)
    parser.add_argument("--warps", type=int)
    parser.add_argument("--shape", type=int, choices=range(4))
    parser.add_argument("--scope", choices=("kernel", "metadata", "chain"))
    parser.add_argument("--pdl", choices=("on", "off"))
    args = parser.parse_args()
    if (args.mode == "refine") != (args.configs is not None):
        parser.error("refine requires --configs; other modes do not accept it")
    if args.mode == "profile" and any(
        x is None
        for x in (
            args.implementation,
            args.block_m,
            args.block_n,
            args.warps,
            args.shape,
            args.scope,
            args.pdl,
        )
    ):
        parser.error(
            "profile requires implementation, tile, warps, shape, scope, and pdl"
        )
    if args.mode in ("sweep", "refine") and any(
        x is not None
        for x in (
            args.implementation,
            args.block_m,
            args.block_n,
            args.warps,
            args.shape,
            args.scope,
            args.pdl,
        )
    ):
        parser.error(
            "sweep uses its fixed configuration set; profile-only arguments are not accepted"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "a B300 or GB300 GPU is required; no timing can be produced on this host"
        )
    if (
        torch.cuda.get_device_capability() != (10, 3)
        or "B300" not in torch.cuda.get_device_name()
    ):
        raise RuntimeError("this experiment requires B300 or GB300 (SM103)")
    baseline_hash = baseline_prefill_sha256()
    if baseline_hash != BASELINE_PREFILL_SHA256:
        raise RuntimeError("baseline source differs from the fixed experiment revision")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Each counter run has its own report stem. Keep its exact binary instead
    # of overwriting another request partition's compiled artifacts.
    artifact_dir = args.output.parent / "compiled" / args.output.stem
    report = {
        "status": "running",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "baseline_prefill_sha256": baseline_hash,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(
            Path(__file__)
            .resolve()
            .parents[3]
            .joinpath("python/tokenspeed_kernel/ops/conv/_triton/prefill_candidate.py")
            .read_bytes()
        ).hexdigest(),
        "gluon_candidate_sha256": hashlib.sha256(
            Path(__file__)
            .resolve()
            .parents[3]
            .joinpath("python/tokenspeed_kernel/ops/conv/_gluon/prefill_candidate.py")
            .read_bytes()
        ).hexdigest(),
        "initial_clocks": clock_snapshot(),
        "compiled_artifact_directory": str(artifact_dir),
        "records": [],
        "binaries": {},
        "timing": "CUDA events inside one-call CUDA graphs; 5x100 samples; state reset before start event; no explicit L2 flush; alternating baseline/candidate order",
        "metadata_scope": "GPU metadata build plus kernel; Python allocation/planning time is excluded; production builds metadata once per forward, not per layer",
        "workload": {
            "channels": 36864,
            "width": 4,
            "dtype": "BF16",
            "silu": True,
            "lengths": MAIN_LENGTHS,
        },
    }
    try:
        if args.mode == "profile":
            config = PrefillCandidateConfig(
                args.implementation, args.block_m, args.block_n, args.warps
            )
            case = case_for(MAIN_LENGTHS[args.shape], 36864, 713 + args.shape)
            check_case(case, config, silu=True, enable_pdl=args.pdl == "on")
            graph = MeasuredGraph(case, config, scope=args.scope, pdl=args.pdl == "on")
            graph.check_replay()
            graph.state.copy_(case.initial_state)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            graph.forward()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            report["binaries"][tag(config)] = dump_binary(
                graph.binary, artifact_dir, tag(config)
            )
            report["status"] = "profile_complete"
            return
        configs = [
            PrefillCandidateConfig("serial", m, 256, 4) for m in (16, 32, 64, 128)
        ]
        configs += [
            PrefillCandidateConfig(kind, 8, 256, 4) for kind in ("window", "direct")
        ]
        configs += [
            PrefillCandidateConfig(kind, m, n, w)
            for kind, m, n, w in itertools.product(
                ("window", "direct"), (16, 32, 64, 128), (64, 128, 256), (4, 8)
            )
        ]
        if args.mode == "refine":
            configs = [
                PrefillCandidateConfig(**settings)
                for settings in json.loads(args.configs.read_text())
            ]
            if not configs or all(
                config.implementation == "serial" for config in configs
            ):
                raise ValueError("refine requires at least one parallel candidate")
            report["requested_configs"] = [asdict(config) for config in configs]
        # Complete correctness at the actual model width before collecting timings.
        for shape, lengths in enumerate(MAIN_LENGTHS):
            case = case_for(lengths, 36864, 713 + shape)
            for config in [BASELINE, *configs]:
                binary = check_case(case, config, silu=True, enable_pdl=False)
                name = f"shape{shape}-{tag(config)}"
                report["binaries"][name] = dump_binary(binary, artifact_dir, name)
            del case
        report["correctness"] = (
            "all configurations passed exact baseline and FP32 reference checks"
        )
        for shape, lengths in enumerate(MAIN_LENGTHS):
            case = case_for(lengths, 36864, 713 + shape)
            for config in configs:
                for scope in ("kernel", "metadata"):
                    result = measure_pair(case, config, scope=scope, pdl=False)
                    report["records"].append(
                        {"shape": shape, "config": asdict(config), **result}
                    )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(f"shape={shape} {tag(config)} complete", flush=True)
            del case
        rankings = []
        for config in configs:
            ratios = [
                r["speedup"]
                for r in report["records"]
                if r["config"] == asdict(config) and r["scope"] == "kernel"
            ]
            geomean = math.prod(ratios) ** (1 / len(ratios))
            rankings.append(
                {
                    "config": asdict(config),
                    "geomean_speedup": geomean,
                    "min_speedup": min(ratios),
                    "accepted": geomean >= 1.5 and min(ratios) >= 1 / 1.05,
                    "target_reached": geomean >= 2 and min(ratios) >= 1 / 1.05,
                }
            )
        rankings.sort(key=lambda r: r["geomean_speedup"], reverse=True)
        report["rankings"] = rankings
        parallel = [r for r in rankings if r["config"]["implementation"] != "serial"]
        accepted = [r for r in parallel if r["accepted"]]
        winner = accepted[0] if accepted else parallel[0]
        best = PrefillCandidateConfig(**winner["config"])
        report["selected_for_followup"] = winner
        report["pdl_chain"] = []
        for shape, lengths in enumerate(MAIN_LENGTHS):
            for pdl in (False, True):
                case = case_for(lengths, 36864, 713 + shape)
                check_case(case, best, silu=True, enable_pdl=pdl)
                report["pdl_chain"].append(
                    {"shape": shape, **measure_pair(case, best, scope="chain", pdl=pdl)}
                )
        report["small_requests"] = []
        for length in (1, 2, 3, 8, 64, 1330):
            case = case_for([length], 36864, 713)
            check_case(case, best, silu=True, enable_pdl=False)
            report["small_requests"].append(
                {
                    "tokens": length,
                    **measure_pair(case, best, scope="kernel", pdl=False),
                }
            )
        report["counters"] = []
        for config in (BASELINE, best):
            for shape in range(4):
                output = args.output.parent / f"ncu-{tag(config)}-shape{shape}"
                command = ncu_command(config, output, shape)
                result = {
                    "command": command,
                    "available": shutil.which("ncu") is not None,
                }
                if result["available"]:
                    completed = subprocess.run(
                        command, text=True, capture_output=True, check=False
                    )
                    output.with_suffix(".log").write_text(
                        completed.stdout + completed.stderr
                    )
                    result["returncode"] = completed.returncode
                    result["available"] = completed.returncode == 0
                report["counters"].append(result)
                if result.get("returncode", 0) != 0:
                    # A denied counter request is evidence; never retry with privileges.
                    break
            if report["counters"][-1].get("returncode", 0) != 0:
                break
        report["status"] = "accepted" if accepted else "below_acceptance_threshold"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["final_clocks"] = clock_snapshot()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
