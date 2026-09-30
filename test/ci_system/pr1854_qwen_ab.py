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

"""Temporary PR #1854 diagnostic: same-runner main/merge/main/merge.

Not part of the feature PR. The ordinary CI installer supplies one environment.
Build each revision's kernel wheel once, then switch only that wheel and the
runtime source path between rounds. Both revisions retain the original perf
gate. A below-threshold round is recorded, not omitted from the comparison.
"""

import copy
import hashlib
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml
from huggingface_hub import HfApi

BASE = "0fbc6fa9e38e15129df009f09f46d7c34ae7808f"
CANDIDATE = "55a42ac03039db62bb61844f41bd7a33f27148e7"
CONFIG = "test/ci/perf/qwen3.5-397b-a17b-nvfp4-evalscope-agentic-b200-8gpu.yaml"
MODEL = "nvidia/Qwen3.5-397B-A17B-NVFP4"
GPU_FIELDS = (
    "index,name,pstate,clocks.sm,clocks.mem,power.draw,power.limit,"
    "temperature.gpu,utilization.gpu,memory.used"
)


def command(args, cwd, env, log, check):
    print(f"[ab] {shlex.join(map(str, args))}", flush=True)
    with log.open("w") as stream:
        result = subprocess.run(
            list(map(str, args)),
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if check and result.returncode:
        print(log.read_text()[-12000:], flush=True)
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
    return result.returncode


def gpu_identity():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    )
    if len(raw.strip().splitlines()) != 8:
        raise RuntimeError("This diagnostic requires eight GPUs in one CI job")
    return hashlib.sha256(raw.encode()).hexdigest()


def telemetry(stop, path):
    with path.open("w") as stream:
        while not stop.is_set():
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
            stream.write(
                json.dumps({"time": time.time(), "gpus": result.stdout}) + "\n"
            )
            stream.flush()
            stop.wait(2)


def package_versions():
    # Do not dump environment variables or credentials into public artifacts.
    from importlib.metadata import distributions

    return {
        package.metadata["Name"].lower().replace("_", "-"): package.version
        for package in distributions()
        if package.metadata["Name"].lower().replace("_", "-") != "tokenspeed-kernel"
    }


def main():
    root = Path.cwd().resolve()
    artifacts = root / ".ci-artifacts" / "published" / "ab"
    artifacts.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(
        TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE="1",
        TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE="1",
        TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE="1",
        TOKENSPEED_KERNEL_BACKEND="cuda",
        FLASHINFER_CUDA_ARCH_LIST="10.0a",
        MAX_JOBS="16",
        CPLUS_INCLUDE_PATH="/usr/local/cuda/include/cccl",
        C_INCLUDE_PATH="/usr/local/cuda/include/cccl",
    )
    source_root = root / ".ab-sources"
    source_root.mkdir(exist_ok=False)
    sources = {}
    wheels = {}
    recipes = {}
    for label, ref in (("main", BASE), ("pr", CANDIDATE)):
        command(
            ["git", "fetch", "origin", ref],
            root,
            env,
            artifacts / f"fetch-{label}.log",
            True,
        )
        source = source_root / label
        command(
            ["git", "worktree", "add", "--detach", source, ref],
            root,
            env,
            artifacts / f"checkout-{label}.log",
            True,
        )
        sources[label] = source
        recipes[label] = yaml.safe_load((source / CONFIG).read_text())
    if recipes["main"] != recipes["pr"]:
        raise RuntimeError("The pinned revisions do not have identical CI recipes")
    for path in (
        "tokenspeed-scheduler",
        "tokenspeed-mla",
        "python/pyproject.toml",
        "tokenspeed-kernel/python/requirements",
    ):
        subprocess.run(
            ["git", "diff", "--exit-code", BASE, CANDIDATE, "--", path],
            cwd=root,
            check=True,
        )

    model_revision = HfApi().model_info(MODEL).sha
    versions = package_versions()
    identity = gpu_identity()
    manifest = {
        "base": BASE,
        "candidate": CANDIDATE,
        "model": MODEL,
        "model_revision": model_revision,
        "seed": 0,
        "order": ["main", "pr", "main", "pr"],
        "gpu_identity_sha256": identity,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "packages": versions,
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2))
    command(["nvidia-smi", "topo", "-m"], root, env, artifacts / "topology.log", True)

    for label, source in sources.items():
        wheel_dir = root / ".ab-wheels" / label
        wheel_dir.mkdir(parents=True, exist_ok=False)
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                wheel_dir,
                source / "tokenspeed-kernel/python",
            ],
            source,
            env,
            artifacts / f"build-{label}.log",
            True,
        )
        matches = list(wheel_dir.glob("tokenspeed_kernel-*.whl"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one kernel wheel for {label}: {matches}")
        wheels[label] = matches[0]

    rounds = []
    for index, label in enumerate(manifest["order"], start=1):
        source = sources[label]
        run_dir = artifacts / f"{index}-{label}"
        run_dir.mkdir()
        if gpu_identity() != identity:
            raise RuntimeError("GPU identities changed during paired run")
        for _ in range(30):
            active = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
            ).strip()
            if not active:
                break
            time.sleep(1)
        if active:
            raise RuntimeError(
                "GPU compute processes remain before starting the next round"
            )
        command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                wheels[label],
            ],
            source,
            env,
            run_dir / "install.log",
            True,
        )
        if package_versions() != versions:
            raise RuntimeError("Non-kernel package versions changed during A/B")
        round_env = dict(env, PYTHONPATH=str(source / "python"))
        verify = (
            "import json, tokenspeed, tokenspeed_kernel; "
            "from pathlib import Path; "
            f"assert Path(tokenspeed.__file__).is_relative_to({str(source / 'python')!r}); "
            "print(json.dumps({'runtime': tokenspeed.__file__, "
            "'kernels': tokenspeed_kernel.__file__}))"
        )
        command(
            [sys.executable, "-c", verify],
            source,
            round_env,
            run_dir / "imports.log",
            True,
        )

        recipe = copy.deepcopy(recipes[label])
        recipe["server"]["command"] += f" --revision {model_revision} --seed 0"
        # Keep the CI workload unchanged; preserve raw outputs instead of the
        # recipe's temporary-directory cleanup, and install the client only once.
        perf_command = recipe["perf"]["command"]
        for fragment in (
            'trap \'rm -rf "$OUTPUTS_DIR" "$WARMUP_DIR"\' EXIT &&',
            'rm -rf "$OUTPUTS_DIR" "$WARMUP_DIR" &&',
        ):
            if fragment not in perf_command:
                raise RuntimeError(f"Unexpected CI recipe: missing {fragment}")
            perf_command = perf_command.replace(fragment, "")
        perf_command = perf_command.replace(
            "/tmp/tokenspeed-agentic-perf", str(run_dir / "requests")
        )
        perf_command = perf_command.replace(
            "/tmp/tokenspeed-agentic-warmup", str(run_dir / "warmup")
        )
        recipe["perf"]["command"] = perf_command
        config = source / ".ab-run.yaml"
        config.write_text(yaml.safe_dump(recipe, sort_keys=False))
        stop = threading.Event()
        sampler = threading.Thread(target=telemetry, args=(stop, run_dir / "gpu.jsonl"))
        sampler.start()
        try:
            status = command(
                [
                    sys.executable,
                    root / "test/ci_system/pipeline.py",
                    "execute",
                    "--config",
                    config.name,
                    "--runner",
                    "b200v2-8gpu",
                    "--work-dir",
                    source,
                    "--skip-stage",
                    "install",
                    "--skip-stage",
                    "perf.install",
                    "--keep-runner-state",
                    "--print-plan",
                    "--result-json",
                    run_dir / "result.json",
                ],
                source,
                round_env,
                run_dir / "pipeline.log",
                False,
            )
        finally:
            stop.set()
            sampler.join()
        server_log = source / ".ci-artifacts/server.log"
        if server_log.exists():
            shutil.copy2(server_log, run_dir / "server.log")
        result = json.loads((run_dir / "result.json").read_text())
        check = result.get("perf_reference_check")
        record = {
            "round": index,
            "label": label,
            "exit_code": status,
            "perf_reference_check": check,
        }
        rounds.append(record)
        (artifacts / "rounds.json").write_text(json.dumps(rounds, indent=2))
        print(f"[ab] completed {index}-{label}: {json.dumps(record)}", flush=True)
        if not check or not check.get("checks"):
            raise RuntimeError(
                f"Round {index}-{label} did not produce valid benchmark metrics"
            )
        # Do not stop at a valid below-floor measurement: the paired reference
        # must still run. Server/client failures above do stop the experiment.
    summary = {}
    for metric in ("Latency (tps/user)", "Throughput (tps/gpu)"):
        samples = {
            label: [
                item["perf_reference_check"]["checks"][0][metric]["actual"]
                for item in rounds
                if item["label"] == label
            ]
            for label in ("main", "pr")
        }
        summary[metric] = {
            "samples": samples,
            "main_median": statistics.median(samples["main"]),
            "pr_median": statistics.median(samples["pr"]),
            "pr_over_main": statistics.median(samples["pr"])
            / statistics.median(samples["main"]),
        }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ab] summary: {json.dumps(summary)}", flush=True)
    print(
        "[ab] All four rounds completed. Original gates and raw data retained.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
