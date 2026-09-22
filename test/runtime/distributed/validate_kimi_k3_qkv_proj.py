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

"""Real checkpoint validation of KDA and MLA QKV column TP.

Run with torchrun on four or sixteen GPUs. This tests projection arithmetic
and ownership, not the attention recurrence or full-model generation.
"""

import argparse
import json
import os
from pathlib import Path
from test.runtime.distributed.kimi_k3_o_proj_helpers import dep_mapping
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors import safe_open

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.attention.column_proj import (
    DistributedColumnProjection,
    column_projection_width,
)
from tokenspeed.runtime.layers.attention.o_proj import (
    initialize_projection_group,
    projection_mapping,
)
from tokenspeed.runtime.layers.linear import ColumnParallelLinear
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.kimi_k3 import (
    KimiKDAColumnProj,
    KimiLinearKDA,
    _assemble_fp8_fused_qkv_a,
)


def load_segments(root, layer, leaves):
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    result = []
    for leaf in leaves:
        name = f"language_model.model.layers.{layer}.self_attn.{leaf}"
        tensors = []
        for suffix in (".weight", ".weight_scale"):
            key = name + suffix
            with safe_open(root / index[key], framework="pt", device="cpu") as handle:
                tensors.append(handle.get_tensor(key))
        weight, scale = tensors
        result.append((weight, scale.reshape((weight.shape[0] + 127) // 128, -1)))
    return result


def make_column(weight, scale, parallel):
    with torch.device("cuda"):
        linear = ColumnParallelLinear(
            weight.shape[1],
            weight.shape[0],
            bias=False,
            gather_output=False,
            quant_config=Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                ignored_layers=[],
                weight_block_size=[128, 128],
                scale_fmt=None,
            ),
            prefix="self_attn.qkv_proj",
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
            tp_group=parallel.tp_group,
        )
    linear.weight.weight_loader(linear.weight, weight)
    linear.weight_scale_inv.weight_loader(linear.weight_scale_inv, scale)
    linear.quant_method.process_weights_after_loading(linear)
    return linear


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world in (4, 16) and args.rows > 0
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_default_dtype(torch.bfloat16)
    pg_manager.init_distributed(
        dep_mapping(rank, world),
        distributed_init_method="env://",
        backend="nccl",
        timeout=600,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    parallel = projection_mapping(rank, world, 4)
    initialize_projection_group(parallel)
    cases = (
        (
            "KDA_QKV_gates",
            0,
            ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"),
        ),
        ("MLA_QKV_A_gate", 3, ("g_proj", "q_a_proj", "kv_a_proj_with_mqa")),
        ("MLA_Q_B", 3, ("q_b_proj",)),
    )
    for label, layer, leaves in cases:
        segments = load_segments(args.model, layer, leaves)
        used = sum(w.shape[0] for w, _ in segments)
        total = column_projection_width(used, 4, 128)
        weight, scale = _assemble_fp8_fused_qkv_a(segments, total)
        baseline = make_column(weight, scale, projection_mapping(rank, world, 1))
        if label == "KDA_QKV_gates":
            with torch.device("cuda"):
                linear = KimiKDAColumnProj(
                    weight.shape[1],
                    segments[0][0].shape[0],
                    segments[-1][0].shape[0],
                    segments[-2][0].shape[0],
                    parallel,
                    "self_attn.qkvgb_proj",
                )
            for key, (w, s) in zip(("q", "k", "v", "g", "f_a", "b"), segments):
                linear.weight.weight_loader(linear.weight, w, key)
                linear.weight_scale_inv.weight_loader(linear.weight_scale_inv, s, key)
            linear.verify_fp8_load_complete()
            linear.quant_method.process_weights_after_loading(linear)
        else:
            linear = make_column(weight, scale, parallel)
        shard = total // 4
        start = parallel.tp_rank * shard
        torch.testing.assert_close(
            linear.weight.float().cpu(),
            weight[start : start + shard].float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            linear.weight_scale_inv.cpu(),
            scale[start // 128 : (start + shard) // 128],
            rtol=0,
            atol=0,
        )
        communication = DistributedColumnProjection(
            parallel,
            weight.shape[1],
            total,
            total,
            max(args.rows, 513),
            torch.bfloat16,
            torch.device("cuda"),
            "trtllm",
            "flashinfer",
        )
        reference_weight = weight.float().cuda() * scale.cuda().repeat_interleave(
            128, 0
        ).repeat_interleave(128, 1)
        patterns = (
            [args.rows] * world,
            [129] * world,
            [512] * world,
            [513 if r % 4 == 0 else 0 for r in range(world)],
            [r % 4 for r in range(world)],
            [3 if r == 0 else 0 for r in range(world)],
            [0] * world,
        )
        for counts in patterns:
            x = torch.randn(
                counts[rank],
                weight.shape[1],
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(1000 + rank),
            )
            expected = baseline(x)[0] if x.shape[0] else x.new_empty((0, total))
            actual = communication.forward(x, linear, counts)
            if label == "KDA_QKV_gates":
                # Exercise the model's real split/dispatch, including empty
                # owners. The backend must still see every local attention head.
                attention = SimpleNamespace(
                    local_num_heads=segments[-1][0].shape[0],
                    head_dim=segments[-2][0].shape[0],
                    qkvgb_proj=linear,
                    input_projection_parallel=parallel,
                    input_projection_exchange=communication,
                )
                ctx = SimpleNamespace(
                    collective_global_num_tokens=counts, global_num_tokens=None
                )
                parts = KimiLinearKDA._project_qkvfab(
                    attention, x, attnres_partial_args=None, ctx=ctx
                )
                torch.testing.assert_close(
                    torch.cat(parts, dim=-1), actual[:, :used], rtol=0, atol=0
                )
            metrics = torch.zeros(3, device="cuda", dtype=torch.float32)
            if x.shape[0]:
                ref = x.float() @ reference_weight.T
                norm = ref[:, :used].norm().clamp_min(1e-8)
                metrics[0] = (
                    actual[:, :used].float() - expected[:, :used].float()
                ).norm() / expected[:, :used].float().norm().clamp_min(1e-8)
                metrics[1] = (expected[:, :used].float() - ref[:, :used]).norm() / norm
                metrics[2] = (actual[:, :used].float() - ref[:, :used]).norm() / norm
                assert torch.count_nonzero(actual[:, used:]).item() == 0
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
            assert metrics[0] < 0.015 and metrics[2] < metrics[1] + 0.01, (
                label,
                counts,
                metrics,
            )
            saved = actual.clone()
            communication.forward(x * 0.5, linear, counts)
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
            for _ in range(3):
                communication.forward(x, linear, counts)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = communication.forward(x, linear, counts)
            original = x.clone()
            for factor in (0.5, -1.0, 0.0):
                x.copy_(original * factor)
                graph.replay()
                eager = communication.forward(x, linear, counts)
                torch.testing.assert_close(captured, eager, rtol=0, atol=0)
            torch.cuda.synchronize()
            del graph, captured
            if rank == 0:
                print(
                    json.dumps(
                        dict(
                            case=label,
                            counts=counts,
                            error=metrics.tolist(),
                            result="PASS",
                        )
                    ),
                    flush=True,
                )
        communication.close()
        del communication, linear, baseline, reference_weight
    dist.barrier()
    if rank == 0:
        print("QKV_PROJECTION_VALIDATION_PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
