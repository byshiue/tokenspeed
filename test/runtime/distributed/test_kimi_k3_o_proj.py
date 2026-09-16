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

"""Small projection validation, without loading the full model.

Unit checks: pytest test/runtime/distributed/test_kimi_k3_o_proj.py
GPU harness: python -m torch.distributed.run --nproc-per-node=4 <this-file> --benchmark
Add --model /path/to/checkpoint to load actual FP8 attention weights from the
NVFP4 mixed-precision model. Run with 16 ranks to exercise four TP4 subgroups.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.comm_ops import all_to_all_single, reduce_scatter
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.models.kimi_k3_o_proj import (
    ENV_NAME,
    ProjectionWorkspace,
    initialize_projection_parallelism,
    make_output_projection,
    projection_mapping,
)


def dep_mapping(rank: int, world: int) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=world,
        attn_tp_size=1,
        attn_cp_size=1,
        attn_dp_size=world,
        attn_dcp_size=1,
        dense_tp_size=1,
        dense_dp_size=world,
        moe_tp_size=1,
        moe_ep_size=world,
        moe_dp_size=1,
        vision_tp_size=1,
        vision_dp_size=1,
        linear_attn_tp_size=1,
        pp_size=1,
        pp_layer_partition=None,
        nprocs_per_node=None,
        nnodes=None,
        base_gpu_id=0,
        gpu_id_step=1,
    )


def test_projection_mapping_and_validation():
    for size in (1, 2, 4, 8, 16):
        for rank in range(16):
            mapping = dep_mapping(rank, 16)
            parallel = projection_mapping(mapping, str(size))
            assert parallel.tp_group == tuple(
                range(rank // size * size, (rank // size + 1) * size)
            )
            assert parallel.tp_rank == rank % size
            assert parallel.dp_size == 16 // size
            assert mapping.attn.tp_size == mapping.linear_attn.tp_size == 1
            assert mapping.moe.ep_size == 16
    for value in ("", "bad", "0", "-1", "3", "32"):
        with pytest.raises(ValueError):
            projection_mapping(dep_mapping(0, 16), value)
    with pytest.raises(ValueError, match="requires"):
        projection_mapping(Mapping(rank=0, world_size=4), "4")


def test_disabled_projection_and_shard_loader(monkeypatch):
    mapping = dep_mapping(2, 4)
    monkeypatch.setenv(ENV_NAME, "1")
    local, exchange = make_output_projection(
        mapping=mapping,
        input_size=32,
        output_size=16,
        quant_config=None,
        prefix="self_attn.o_proj",
        default_parallel=mapping.attn,
        reduce_results=False,
    )
    assert exchange is None
    assert local.weight.shape == (16, 32)
    monkeypatch.setenv(ENV_NAME, "4")
    sharded, exchange = make_output_projection(
        mapping=mapping,
        input_size=32,
        output_size=16,
        quant_config=None,
        prefix="self_attn.o_proj",
        default_parallel=mapping.attn,
        reduce_results=False,
    )
    weight = torch.arange(16 * 32, dtype=sharded.weight.dtype).view(16, 32)
    sharded.weight.weight_loader(sharded.weight, weight)
    torch.testing.assert_close(sharded.weight, weight[:, 16:24])
    assert not sharded.reduce_results
    assert exchange is not None


def test_attention_construction_and_empty_rank_participation(monkeypatch):
    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.models.kimi_k3 import KimiLinearKDA, KimiLinearMLAAttention

    monkeypatch.setenv(ENV_NAME, "4")
    mapping = dep_mapping(2, 4)
    config = KimiLinearConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=16,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "num_heads": 4,
            "head_dim": 16,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
    )
    layers = [
        KimiLinearKDA(
            config=config, mapping=mapping, layer_id=0, quant_config=None, prefix=""
        ),
        KimiLinearMLAAttention(
            config=config,
            mapping=mapping,
            hidden_size=64,
            num_heads=4,
            qk_nope_head_dim=16,
            qk_rope_head_dim=0,
            v_head_dim=16,
            q_lora_rank=16,
            kv_lora_rank=16,
            rope_theta=10000,
            rope_scaling=None,
            max_position_embeddings=128,
            quant_config=None,
            layer_id=0,
            prefix="",
            reduce_attn_results=False,
            alt_stream=None,
        ),
    ]
    for layer in layers:
        assert layer.o_proj.weight.shape == (64, 16)
        assert layer.mapping.attn.tp_size == layer.mapping.linear_attn.tp_size == 1
        calls = []

        def exchange(inputs, linear, counts):
            calls.append((inputs.shape, counts))
            return inputs.new_empty((0, linear.output_size))

        monkeypatch.setattr(layer.output_projection_exchange, "forward", exchange)
        counts = [7, 0, 0, 0]
        result = layer(
            positions=torch.empty(0, dtype=torch.int64),
            hidden_states=torch.empty(0, 64),
            ctx=SimpleNamespace(
                collective_global_num_tokens=counts, global_num_tokens=None
            ),
            comm_manager=None,
            block_scale=None,
            attnres_partial_args=None,
        )
        assert result.shape == (0, 64)
        assert calls == [(torch.Size([0, 64]), counts)]


def load_projection(model: str, layer: int):
    from safetensors import safe_open

    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config

    root = Path(model)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    name = f"language_model.model.layers.{layer}.self_attn.o_proj"
    with safe_open(root / index[name + ".weight"], framework="pt", device="cpu") as f:
        weight = f.get_tensor(name + ".weight")
    with safe_open(
        root / index[name + ".weight_scale"], framework="pt", device="cpu"
    ) as f:
        scale = f.get_tensor(name + ".weight_scale").squeeze()
    assert (
        weight.dtype == torch.float8_e4m3fn
    ), "Harness expects checkpoint FP8 attention weights"
    quant = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=[],
        weight_block_size=[128, 128],
        scale_fmt=None,
    )
    return weight, scale, quant


def make_linears(mapping: Mapping, weight, scale, quant):
    k, n = weight.shape[1], weight.shape[0]
    result = []
    for size in (1, 4):
        os.environ[ENV_NAME] = str(size)
        with torch.device("cuda"):
            linear, exchange = make_output_projection(
                mapping=mapping,
                input_size=k,
                output_size=n,
                quant_config=quant,
                prefix="self_attn.o_proj",
                default_parallel=mapping.attn,
                reduce_results=False,
            )
        linear.weight.weight_loader(linear.weight, weight)
        if scale is not None:
            linear.weight_scale_inv.weight_loader(linear.weight_scale_inv, scale)
        linear.quant_method.process_weights_after_loading(linear)
        result.append((linear, exchange))
    os.environ[ENV_NAME] = "4"
    return result


def measure(call, iterations: int, graph: bool) -> float:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            call()
        call = capture.replay
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    elapsed = torch.tensor(
        start.elapsed_time(end) * 1000 / iterations, device="cuda", dtype=torch.float32
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def measure_components(exchange, linear, inputs, counts, iterations: int):
    """Measure isolated stages separately; their sum need not equal pipeline latency."""
    size = exchange.parallel.tp_size
    rows = max(counts[r] for r in exchange.parallel.tp_group)
    shard = exchange.input_size // size
    send = exchange.workspace.send[: rows * exchange.input_size].view(size, rows, shard)
    recv = exchange.workspace.recv[: rows * exchange.input_size].view(
        size * rows, shard
    )
    if rows == 0:
        return {}

    def pack():
        send.zero_()
        send[:, : inputs.shape[0]].copy_(
            inputs.reshape(inputs.shape[0], size, shard).permute(1, 0, 2)
        )

    def exchange_inputs():
        all_to_all_single(
            recv,
            send.view(size * rows, shard),
            exchange.parallel.tp_group,
            backend=None,
        )

    pack()
    exchange_inputs()
    partial, _ = linear(recv)
    stages = {
        "pack": pack,
        "all_to_all": exchange_inputs,
        "gemm": lambda: linear(recv),
        "reduce_scatter": lambda: reduce_scatter(
            partial, exchange.parallel.tp_group, backend=None
        ),
    }
    return {name: measure(call, iterations, True) for name, call in stages.items()}


def profile_projection(exchange, linear, baseline, k: int, world: int):
    """Capture a short warmed baseline/TP4 graph comparison for NSYS."""
    inputs = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    counts = [8] * world
    for _ in range(5):
        baseline(inputs)
        exchange.forward(inputs, linear, counts)
    torch.cuda.synchronize()
    local_graph, tp_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(local_graph):
        baseline(inputs)
    with torch.cuda.graph(tp_graph):
        exchange.forward(inputs, linear, counts)
    dist.barrier()
    torch.cuda.cudart().cudaProfilerStart()
    for name, graph in (
        ("baseline_projection", local_graph),
        ("tp4_projection", tp_graph),
    ):
        with torch.cuda.nvtx.range(name):
            for _ in range(20):
                graph.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--layer", type=int, choices=(0, 3))
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
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
    os.environ[ENV_NAME] = "4"
    initialize_projection_parallelism(mapping)
    shapes = [(128, 256, None), (7168, 12288, None)]
    if args.model:
        shapes = [
            (0, 0, layer)
            for layer in ((args.layer,) if args.layer is not None else (0, 3))
        ]
    for n, k, layer in shapes:
        if layer is None:
            generator = torch.Generator().manual_seed(42)
            weight = (
                torch.randn(n, k, generator=generator, dtype=torch.float32).to(
                    torch.bfloat16
                )
                / k**0.5
            )
            scale, quant = None, None
        else:
            weight, scale, quant = load_projection(args.model, layer)
            n, k = weight.shape
        (baseline, _), (linear, exchange) = make_linears(mapping, weight, scale, quant)
        reference_weight = weight.float()
        if scale is not None:
            reference_weight *= scale.repeat_interleave(128, 0).repeat_interleave(
                128, 1
            )
        reference_weight = reference_weight.cuda()
        exchange.workspace = ProjectionWorkspace(
            512, k, torch.bfloat16, torch.device("cuda")
        )
        patterns = [
            [1] * world,
            [8] * world,
            [64] * world,
            [7 if r == 0 else 0 for r in range(world)],
            [r % 4 for r in range(world)],
            [0] * world,
        ]
        if args.benchmark:
            patterns += [
                [256] * world,
                [512 if r % 4 == 0 else 1 for r in range(world)],
            ]
        for counts in patterns:
            generator = torch.Generator(device="cuda").manual_seed(100 + rank)
            x = torch.randn(
                counts[rank],
                k,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            expected = baseline(x)[0] if counts[rank] else x.new_empty((0, n))
            actual = exchange.forward(x, linear, counts)
            if x.shape[0]:
                reference = x.float() @ reference_weight.T
                delta = actual.float() - expected.float()
                absolute_max = delta.abs().max()
                relative_l2 = delta.norm() / expected.float().norm().clamp_min(1e-8)
                max_scaled = delta.abs().max() / expected.float().abs().max().clamp_min(
                    1e-8
                )
                assert relative_l2 < 0.015, (rank, counts, relative_l2.item())
                assert max_scaled < 0.03, (rank, counts, max_scaled.item())
                norm = reference.norm().clamp_min(1e-8)
                baseline_error = (expected.float() - reference).norm() / norm
                feature_error = (actual.float() - reference).norm() / norm
                assert feature_error < baseline_error + 0.01
            else:
                relative_l2 = torch.tensor(0.0, device="cuda", dtype=torch.float32)
                absolute_max = torch.zeros_like(relative_l2)
                baseline_error = torch.zeros_like(relative_l2)
                feature_error = torch.zeros_like(relative_l2)
                assert actual.shape == (0, n)
            dist.all_reduce(relative_l2, op=dist.ReduceOp.MAX)
            dist.all_reduce(absolute_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(baseline_error, op=dist.ReduceOp.MAX)
            dist.all_reduce(feature_error, op=dist.ReduceOp.MAX)
            # Fixed collective shapes under graphs; change input contents on
            # replay to detect stale packing and graph-owned scratch aliases.
            if max(counts):
                for _ in range(3):
                    exchange.forward(x, linear, counts)
                torch.cuda.synchronize()
                capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(capture):
                    graphed = exchange.forward(x, linear, counts)
                for multiplier in (0.5, -1.0, 0.0):
                    x.mul_(multiplier)
                    capture.replay()
                    eager = exchange.forward(x, linear, counts)
                    torch.testing.assert_close(graphed, eager, rtol=0, atol=0)
                # Captured collective counts describe physical rows. Valid
                # token counts may change within that envelope between replays.
                for valid in (counts[rank] // 2, counts[rank], 0):
                    x.zero_()
                    x[:valid].normal_(generator=generator)
                    capture.replay()
                    eager = exchange.forward(x, linear, counts)
                    torch.testing.assert_close(graphed, eager, rtol=0, atol=0)
                torch.cuda.synchronize()
                # NCCL communicators cannot be destroyed while a live graph
                # still holds collective nodes referencing them.
                del capture, graphed
            record = {
                "shape": [n, k],
                "layer": layer,
                "counts": counts,
                "relative_l2": relative_l2.item(),
                "absolute_max": absolute_max.item(),
                "baseline_reference_l2": baseline_error.item(),
                "feature_reference_l2": feature_error.item(),
            }
            if args.benchmark and max(counts):
                # Restore nonzero inputs after the stale-input test.
                x.normal_(generator=generator)
                local = lambda: baseline(x) if counts[rank] else None
                feature = lambda: exchange.forward(x, linear, counts)
                for graph in (False, True):
                    record[f"baseline_us_graph_{graph}"] = measure(
                        local, args.iterations, graph
                    )
                    record[f"tp4_us_graph_{graph}"] = measure(
                        feature, args.iterations, graph
                    )
                # Component measurements use balanced rows, so every rank
                # participates in the same timing collectives.
                if len(set(counts)) == 1:
                    record["components_us"] = measure_components(
                        exchange, linear, x, counts, args.iterations
                    )
            if rank == 0:
                print(json.dumps(record), flush=True)
        if args.profile:
            profile_projection(exchange, linear, baseline, k, world)
        del baseline, linear, exchange, reference_weight
        torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        print("PROJECTION_VALIDATION_PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
