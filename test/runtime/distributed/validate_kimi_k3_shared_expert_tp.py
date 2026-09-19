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

"""Distributed real-weight validation of the integrated Kimi shared-expert MLP."""

import argparse
import json
import os
from pathlib import Path
from test.runtime.distributed.shared_expert_helpers import dep_mapping

import torch
import torch.distributed as dist
from safetensors import safe_open

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.shared_expert_tp import (
    SharedExpertWorkspace,
    initialize_shared_expert_group,
    shared_expert_mapping,
)
from tokenspeed.runtime.models.kimi_k3 import KimiLinearMLP
from tokenspeed.runtime.utils.cuda_stream import StreamFork


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert 1 < args.tp_size < world and world % args.tp_size == 0
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
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = (
        f"language_model.model.layers.{args.layer}.block_sparse_moe.shared_experts."
    )
    weights = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        key = prefix + name + ".weight"
        with safe_open(args.model / index[key], framework="pt", device="cpu") as f:
            weights.append(f.get_tensor(key))
    assert all(w.dtype == torch.bfloat16 for w in weights)
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    mlps = []
    for size in ("1", str(args.tp_size)):
        parallel = shared_expert_mapping(mapping, size)
        with torch.device("cuda"):
            mlp = KimiLinearMLP(
                weights[0].shape[1],
                weights[0].shape[0],
                tp_rank=parallel.tp_rank if parallel is not None else 0,
                tp_size=parallel.tp_size if parallel is not None else 1,
                tp_group=parallel.tp_group if parallel is not None else None,
                shared_parallel=parallel,
                quant_config=None,
                prefix="shared_experts",
                reduce_results=False,
                is_shared_expert=True,
                activation_situ_beta=config["activation_situ_beta"],
                activation_situ_linear_beta=config["activation_situ_linear_beta"],
            )
        for i in (0, 1):
            mlp.gate_up_proj.weight.weight_loader(
                mlp.gate_up_proj.weight, weights[i], i
            )
        mlp.down_proj.weight.weight_loader(mlp.down_proj.weight, weights[2])
        mlps.append(mlp)
    baseline, candidate = mlps
    parallel = candidate.shared_parallel
    initialize_shared_expert_group(parallel)
    candidate.shared_workspace = SharedExpertWorkspace(
        parallel, 129, 7168, torch.device("cuda")
    )
    shard = weights[0].shape[0] // args.tp_size
    start = parallel.tp_rank * shard
    torch.testing.assert_close(
        candidate.gate_up_proj.weight,
        torch.cat([w[start : start + shard] for w in weights[:2]]).cuda(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        candidate.down_proj.weight,
        weights[2][:, start : start + shard].cuda(),
        rtol=0,
        atol=0,
    )
    stream = torch.cuda.Stream(priority=-1)
    fork = StreamFork(stream)
    worst = 0.0
    cases = []
    for rows in (0, 1, 32, 64, 128, 129):
        for pattern in ("balanced", "uneven", "empty_group"):
            counts = [
                (
                    rows
                    if pattern == "balanced"
                    else (rows * (r % args.tp_size) // (args.tp_size - 1))
                )
                for r in range(world)
            ]
            if pattern == "empty_group":
                counts = [0 if r < args.tp_size else rows for r in range(world)]
            x = (
                torch.randn(
                    counts[rank],
                    7168,
                    device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(800 + rank),
                )
                * 0.2
            )

            def run():
                with fork.scope(enable=True, overlap=True):
                    with fork.branch():
                        gathered = candidate.shared_workspace.gather_inputs(x, counts)
                        fork.record_checkpoint()
                        partial = candidate(gathered, down_out=None)
                    # Local routing overlaps AG; dispatch waits only for AG.
                    # Exercise all staged event generations during replay.
                    torch.cuda._sleep(20000)
                    fork.join_checkpoint()
                    torch.cuda._sleep(20000)
                    fork.join()
                    with fork.branch_after_main():
                        output = candidate.shared_workspace.reduce_outputs(
                            partial, counts[rank]
                        )
                    torch.cuda._sleep(20000)
                    fork.join()
                return output

            for _ in range(3):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run()
            original = x.clone()
            for factor in (1.0, 0.0, -1.0, 0.5):
                x.copy_(original * factor)
                expected = baseline(x, down_out=None)
                if rank % args.tp_size == 0:
                    torch.cuda._sleep(100000)
                graph.replay()
                err = (
                    actual.float() - expected.float()
                ).norm() / expected.float().norm().clamp_min(1e-8)
                dist.all_reduce(err, op=dist.ReduceOp.MAX)
                assert err < 0.01, (rows, pattern, err)
                worst = max(worst, err.item())
                retained = run()
                saved = retained.clone()
                x.copy_(original * (factor + 0.25))
                run()
                torch.testing.assert_close(retained, saved, rtol=0, atol=0)
            del graph, actual
            cases.append((rows, pattern))
    candidate.shared_workspace.close()
    dist.barrier()
    if rank == 0:
        print(
            json.dumps(
                dict(
                    status="SHARED_EXPERT_INTEGRATION_PASSED",
                    cases=cases,
                    max_relative_l2=worst,
                )
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
