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
import builtins
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.flashinfer import (
    flashinfer_projection_a2a,
    flashinfer_projection_a2a_borrowed,
    flashinfer_projection_quantized_a2a,
)
from tokenspeed_kernel.ops.communication.triton import triton_pack_projection_input
from tokenspeed_kernel.ops.gemm import fp8_linear_accepts_prepacked_input

from tokenspeed.runtime.distributed.comm_ops import all_to_all_single, reduce_scatter
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.attention import o_proj as projection_ops
from tokenspeed.runtime.layers.attention.o_proj import (
    A2A_ENV_NAME,
    DEFAULT_A2A_BACKEND,
    DEFAULT_RS_BACKEND,
    RS_ENV_NAME,
    ProjectionWorkspace,
    initialize_projection_group,
    make_output_projection,
    projection_mapping,
    validate_projection_settings,
)

ENV_NAME = "TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE"


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


def test_projection_mapping_and_validation(monkeypatch):
    for size in (1, 2, 4, 8, 16):
        for rank in range(16):
            mapping = dep_mapping(rank, 16)
            parallel = projection_mapping(mapping.rank, mapping.world_size, size)
            assert parallel.tp_group == tuple(
                range(rank // size * size, (rank // size + 1) * size)
            )
            assert parallel.tp_rank == rank % size
            assert parallel.dp_size == 16 // size
            assert mapping.attn.tp_size == mapping.linear_attn.tp_size == 1
            assert mapping.moe.ep_size == 16
    for value in ("", "bad", "0", "-1", "3", "32"):
        with pytest.raises(ValueError):
            validate_projection_settings(dep_mapping(0, 16), value, "nccl", "nccl")
    from tokenspeed.runtime.models.kimi_k3 import _output_projection_mapping

    monkeypatch.setenv(ENV_NAME, "4")
    with pytest.raises(ValueError, match="requires"):
        _output_projection_mapping(Mapping(rank=0, world_size=4))
    monkeypatch.setenv(ENV_NAME, "1")
    monkeypatch.setenv(A2A_ENV_NAME, "nccl")
    monkeypatch.setenv(RS_ENV_NAME, "nccl")
    mapping = dep_mapping(0, 1)
    for name, valid, invalid in (
        (
            A2A_ENV_NAME,
            ("nccl", "auto", "flashinfer", "flashinfer_quantized"),
            ("", "invalid", "NVLINK"),
        ),
        (RS_ENV_NAME, ("nccl", "triton_rsag", "triton_peer"), ("", "auto", "invalid")),
    ):
        for value in valid:
            monkeypatch.setenv(name, value)
            validate_projection_settings(
                mapping,
                os.environ[ENV_NAME],
                os.environ.get(A2A_ENV_NAME, DEFAULT_A2A_BACKEND),
                os.environ.get(RS_ENV_NAME, DEFAULT_RS_BACKEND),
            )
        for value in invalid:
            monkeypatch.setenv(name, value)
            with pytest.raises(ValueError, match=name):
                validate_projection_settings(
                    mapping,
                    os.environ[ENV_NAME],
                    os.environ.get(A2A_ENV_NAME, DEFAULT_A2A_BACKEND),
                    os.environ.get(RS_ENV_NAME, DEFAULT_RS_BACKEND),
                )
        monkeypatch.setenv(name, "nccl")

    # Agreement precedes parsing, including disabled or malformed local settings.
    monkeypatch.setattr(projection_ops.dist, "is_initialized", lambda: True)
    groups = []
    monkeypatch.setattr(
        pg_manager,
        "init_process_group",
        lambda group, backend: groups.append((group, backend)),
    )
    monkeypatch.setattr(pg_manager, "get_process_group", lambda backend, group: group)

    def disagree(values, local, group):
        values[:] = [local] * len(values)
        values[-1] = ("4", "nccl", "nccl")

    monkeypatch.setattr(projection_ops.dist, "all_gather_object", disagree)
    for value in ("1", "bad"):
        with pytest.raises(ValueError, match="differ across ranks"):
            validate_projection_settings(dep_mapping(0, 4), value, "nccl", "nccl")
    assert groups == [(tuple(range(4)), "gloo")] * 2


def test_a2a_policy_and_lifecycle(monkeypatch):
    assert DEFAULT_A2A_BACKEND == "flashinfer"
    assert DEFAULT_RS_BACKEND == "triton_peer"
    monkeypatch.setenv(A2A_ENV_NAME, "nccl")
    workspace = ProjectionWorkspace(512, 256, torch.bfloat16, torch.device("cpu"))
    parallel = projection_mapping(0, 4, 4)
    workspace.initialize_a2a(parallel, 256, backend="nccl")
    assert workspace.a2a is None
    assert not workspace.use_flashinfer(parallel, [16] * 4, 256)
    closed = []
    workspace.a2a = SimpleNamespace(close=lambda: closed.append(True))
    for rank in range(4):
        parallel = projection_mapping(rank, 4, 4)
        for counts, channels, expected in (
            ([1] * 4, 256, True),
            ([16] * 4, 256, True),
            ([17] * 4, 256, True),
            ([64] * 4, 256, True),
            ([65] * 4, 256, True),
            ([512] * 4, 256, True),
            ([513] * 4, 256, False),
            ([16, 0, 16, 16], 256, False),
            ([0] * 4, 256, False),
            ([16] * 4, 100, False),
        ):
            assert workspace.use_flashinfer(parallel, counts, channels) == expected
    workspace.close()
    workspace.close()
    assert closed == [True]


def test_bf16_and_quantized_dispatch_across_peer_envelope(monkeypatch):
    # Shared execution must not read Kimi settings or require a MoE layout.
    for name in (ENV_NAME, A2A_ENV_NAME, RS_ENV_NAME):
        monkeypatch.setenv(name, "invalid-model-setting")
    parallel = projection_ops.projection_mapping(0, 4, 4)
    exchange = projection_ops.DistributedOutputProjection(parallel, 512)
    workspace = ProjectionWorkspace(64, 512, torch.bfloat16, torch.device("cpu"))
    exchange.workspace = workspace
    workspace.a2a = object()
    workspace.peer_states[128] = SimpleNamespace(
        input_buffer=lambda rows: torch.empty(rows * 4, 128)
    )
    calls = []

    def quantized_a2a(state, inputs):
        calls.append(("quantized", inputs.shape[0]))
        return inputs, None

    monkeypatch.setattr(
        projection_ops, "flashinfer_projection_quantized_a2a", quantized_a2a
    )
    monkeypatch.setattr(
        projection_ops, "fp8_linear_accepts_prepacked_input", lambda plan: True
    )
    monkeypatch.setattr(
        projection_ops,
        "triton_projection_reduce_scatter_after_a2a",
        lambda peer, partial, rows: partial[:rows],
    )
    monkeypatch.setattr(
        projection_ops,
        "flashinfer_projection_a2a_borrowed",
        lambda state, inputs: (calls.append(("bf16", inputs.shape[0])) or inputs),
    )
    linear = SimpleNamespace(
        output_size=128,
        forward_prepacked_into=lambda values, scales, output: (output, None),
        forward_into=lambda values, scales, output: (output, None),
    )
    for rows in range(1, 65):
        inputs = torch.empty(rows, 512, dtype=torch.bfloat16)
        assert exchange.forward(inputs, linear, [rows] * 4).shape == (rows, 128)
    assert calls == [("bf16", rows) for rows in range(1, 65)]
    workspace.quantized_a2a = True
    for rows in (1, 16, 64):
        inputs = torch.empty(rows, 512, dtype=torch.bfloat16)
        exchange.forward(inputs, linear, [rows] * 4)
    assert calls[-3:] == [("quantized", rows) for rows in (1, 16, 64)]

    ordinary_mapping = Mapping(rank=0, world_size=4)
    linear, wrapper = projection_ops.make_output_projection(
        parallel=parallel,
        input_size=512,
        output_size=128,
        quant_config=None,
        prefix="attention.o_proj",
        default_parallel=ordinary_mapping.attn,
        reduce_results=True,
    )
    assert isinstance(wrapper, projection_ops.DistributedOutputProjection)
    assert linear.weight.shape == (128, 128)
    assert not linear.reduce_results


def test_rsag_policy_and_disabled_initialization(monkeypatch):
    parallel = projection_mapping(0, 4, 4)
    workspace = ProjectionWorkspace(512, 256, torch.bfloat16, torch.device("cpu"))
    monkeypatch.setenv(RS_ENV_NAME, "nccl")
    workspace.initialize_reduce_scatter(parallel, [128], backend="nccl")
    assert workspace.rsag_states == {}
    # Policy depends on padded subgroup capacity, not local valid rows.
    workspace.rsag_states[128] = object()
    peer = object()
    workspace.peer_states[128] = peer
    large_workspace = ProjectionWorkspace(8193, 8, torch.bfloat16, torch.device("cpu"))
    large_workspace.peer_states[128] = peer
    for rows in (512, 513, 8192, 8193):
        assert large_workspace.peer_state(128, rows) is (peer if rows <= 8192 else None)
    for rows, width, expected in (
        (1, 128, True),
        (16, 128, True),
        (17, 128, True),
        (64, 128, True),
        (65, 128, False),
        (256, 128, False),
        (257, 128, False),
        (0, 128, False),
        (16, 256, False),
    ):
        assert workspace.use_rsag(width, rows, True) == expected
        assert not workspace.use_rsag(width, rows, False)
        assert workspace.peer_state(width, rows) is (
            peer if width == 128 and 0 < rows <= 512 else None
        )

    monkeypatch.setenv(RS_ENV_NAME, "triton_peer")
    no_peer = ProjectionWorkspace(16, 256, torch.bfloat16, torch.device("cpu"))
    no_peer.initialize_reduce_scatter(
        projection_ops.projection_mapping(0, 4, 2), [128], backend="triton_peer"
    )
    assert no_peer.peer_states == {}
    assert no_peer.borrowed_a2a is None
    monkeypatch.setenv(RS_ENV_NAME, "triton_rsag")
    no_a2a = ProjectionWorkspace(16, 256, torch.bfloat16, torch.device("cpu"))
    no_a2a.initialize_reduce_scatter(parallel, [128], backend="triton_rsag")
    assert no_a2a.rsag_states == {}
    bad_dtype = ProjectionWorkspace(16, 256, torch.float16, torch.device("cpu"))
    with pytest.raises(ValueError, match="BF16"):
        bad_dtype.initialize_reduce_scatter(parallel, [128], backend="triton_rsag")
    bad_width = ProjectionWorkspace(16, 256, torch.bfloat16, torch.device("cpu"))
    with pytest.raises(ValueError, match="aligned"):
        bad_width.initialize_reduce_scatter(parallel, [127], backend="triton_rsag")


def test_peer_reduction_reuse_fence_contract(monkeypatch):
    import tokenspeed_kernel.ops.communication.triton_projection as kernels

    state = kernels.ProjectionPeerState.__new__(kernels.ProjectionPeerState)
    state.max_rows, state.p, state.r, state.hidden = 16, 4, 0, 128
    state.buffer = torch.empty(64, 128, dtype=torch.bfloat16)
    state.ptrs = None
    barriers = []
    state.handle = SimpleNamespace(barrier=lambda channel: barriers.append(channel))

    class FakeReduction:
        def __getitem__(self, grid):
            return lambda *args: args[1].zero_()

    monkeypatch.setattr(kernels, "owner_reduce", FakeReduction())
    for synchronize_reuse, expected in ((True, [0, 1]), (False, [0])):
        barriers.clear()
        output = state._reduce(state.input_buffer(16), 16, synchronize_reuse)
        assert barriers == expected
        assert output.shape == (16, 128)
        assert output.data_ptr() != state.buffer.data_ptr()


def test_optional_a2a_import_and_topology_fallback(monkeypatch):
    from tokenspeed_kernel.thirdparty.flashinfer.projection_alltoall import (
        create_projection_a2a,
    )

    group = SimpleNamespace(size=lambda: 4)

    def gather(values, value, group):
        values[:] = [value] * group.size()

    monkeypatch.setattr(dist, "all_gather_object", gather)
    original_import = builtins.__import__
    fake = None

    def optional_import(name, *args, **kwargs):
        if name == "flashinfer.comm.ulysses":
            if fake is None:
                raise ImportError("optional API absent")
            return fake
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", optional_import)
    kwargs = dict(
        group=group, max_elems=4096, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    comm, reason = create_projection_a2a(**kwargs, backend="auto")
    assert comm is None and "optional API absent" in reason
    with pytest.raises(RuntimeError, match="unavailable"):
        create_projection_a2a(**kwargs, backend="flashinfer")
    closed = []
    fallback = SimpleNamespace(
        backend="nccl", fallback_reason="no NVLink", close=lambda: closed.append(True)
    )
    fake = SimpleNamespace(UlyssesCommunicator=lambda **kwargs: fallback)
    comm, reason = create_projection_a2a(**kwargs, backend="auto")
    assert comm is None and reason == "no NVLink" and closed == [True]


def test_disabled_projection_and_shard_loader(monkeypatch):
    mapping = dep_mapping(2, 4)
    monkeypatch.setenv(ENV_NAME, "1")
    local, exchange = make_output_projection(
        parallel=projection_mapping(
            mapping.rank, mapping.world_size, int(os.environ[ENV_NAME])
        ),
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
        parallel=projection_mapping(
            mapping.rank, mapping.world_size, int(os.environ[ENV_NAME])
        ),
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
                parallel=projection_mapping(mapping.rank, mapping.world_size, size),
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
    """Time complete operations, amortizing graph submission over a short chain."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    chain = min(20, iterations) if graph else 1
    replays = (iterations + chain - 1) // chain
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            for _ in range(chain):
                call()
        call = capture.replay
        for _ in range(5):
            call()
        torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(replays):
        call()
    end.record()
    end.synchronize()
    elapsed = torch.tensor(
        start.elapsed_time(end) * 1000 / (replays * chain),
        device="cuda",
        dtype=torch.float32,
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

    packed = send.view(size * rows, shard)
    fused = exchange.workspace.use_flashinfer(
        exchange.parallel, counts, exchange.input_size
    )
    peer = exchange.workspace.peer_state(linear.output_size, rows)
    quantized = (
        peer is not None
        and fused
        and exchange.workspace.quantized_a2a
        and exchange.input_size % 512 == 0
        and fp8_linear_accepts_prepacked_input(
            getattr(linear, "_prepared_fp8_linear", None)
        )
    )
    recv_scales = None

    def pack():
        nonlocal packed
        packed = triton_pack_projection_input(inputs, send)

    def exchange_inputs():
        nonlocal recv, recv_scales
        if quantized:
            recv, recv_scales = flashinfer_projection_quantized_a2a(
                exchange.workspace.borrowed_a2a, inputs
            )
        elif peer is not None and fused:
            recv = flashinfer_projection_a2a_borrowed(
                exchange.workspace.borrowed_a2a, inputs
            )
        elif fused:
            recv = flashinfer_projection_a2a(exchange.workspace.a2a, inputs)
        else:
            all_to_all_single(recv, packed, exchange.parallel.tp_group, backend=None)

    def gemm():
        if quantized:
            return linear.forward_prepacked_into(
                recv, recv_scales, peer.input_buffer(rows)
            )
        if peer is not None:
            return linear.forward_into(recv, None, peer.input_buffer(rows))
        return linear(recv)

    pack()
    exchange_inputs()
    partial, _ = gemm()
    stages = {
        "quantize_all_to_all" if quantized else "all_to_all": exchange_inputs,
        "gemm": gemm,
        # Isolated reduction needs its own trailing reuse fence; complete
        # projection timings instead reuse the next A2A's entry barrier.
        "reduce_scatter_standalone": lambda: exchange.workspace.reduce_scatter(
            partial,
            exchange.parallel,
            rows,
            exchange.workspace.use_flashinfer(
                exchange.parallel, counts, exchange.input_size
            ),
        ),
    }
    if not fused:
        stages["pack"] = pack
    return {name: measure(call, iterations, True) for name, call in stages.items()}


def profile_projection(
    exchange, linear, baseline, k: int, world: int, baseline_label: str
):
    """Capture a short warmed baseline/TP4 graph comparison for NSYS."""
    inputs = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    counts = [16] * world
    for _ in range(5):
        baseline(inputs)
        exchange.forward(inputs, linear, counts)
    torch.cuda.synchronize()
    local_graph, tp_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(local_graph):
        for _ in range(20):
            baseline(inputs)
    with torch.cuda.graph(tp_graph):
        for _ in range(20):
            exchange.forward(inputs, linear, counts)
    dist.barrier()
    torch.cuda.cudart().cudaProfilerStart()
    for name, graph in (
        (baseline_label, local_graph),
        ("tp4_projection", tp_graph),
    ):
        with torch.cuda.nvtx.range(name):
            for _ in range(5):
                graph.replay()
    torch.cuda.synchronize()
    control_group = pg_manager.get_process_group("gloo", tuple(range(world)))
    dist.barrier(group=control_group)
    torch.cuda.cudart().cudaProfilerStop()
    # Keep later GPU work out of the profiler-stop window on every rank.
    dist.barrier(group=control_group)


def validate_backend_transitions(exchange, linear, baseline, k, world):
    """Replay mixed backend boundaries with delayed peers and retained outputs."""
    rank = dist.get_rank()
    patterns = [
        [512] * world,
        [513] * world,
        [8192] * world,
        [8193] * world,
        [0] * world,
        [512] * world,
        [513 if r % 4 == 0 else 0 for r in range(world)],
        [512] * world,
    ]
    inputs = [
        torch.randn(counts[rank], k, device="cuda", dtype=torch.bfloat16)
        for counts in patterns
    ]
    # Different progress within each subgroup stresses cross-backend reuse.
    delay = torch.ones(262144, device="cuda")
    expected = [
        baseline(x)[0] if x.shape[0] else x.new_empty((0, linear.output_size))
        for x in inputs
    ]
    for x, counts in zip(inputs, patterns):
        exchange.forward(x, linear, counts)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = []
        for x, counts in zip(inputs, patterns):
            if rank % 4 == 0:
                for _ in range(4):
                    delay.mul_(1.0001)
            outputs.append(exchange.forward(x, linear, counts))
    for _ in range(5):
        graph.replay()
        for actual, reference in zip(outputs, expected):
            if actual.numel():
                relative = (
                    actual.float() - reference.float()
                ).norm() / reference.float().norm()
                assert relative < 0.015
    torch.cuda.synchronize()
    del outputs, graph


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--large-tokens", action="store_true")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--layer", type=int, choices=(0, 3))
    parser.add_argument("--reference-module", type=Path)
    parser.add_argument("--reference-shared-workspace", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
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
    parallel = validate_projection_settings(
        mapping,
        os.environ[ENV_NAME],
        os.environ.get(A2A_ENV_NAME, DEFAULT_A2A_BACKEND),
        os.environ.get(RS_ENV_NAME, DEFAULT_RS_BACKEND),
    )
    initialize_projection_group(parallel)
    reference_module = None
    if args.reference_module is not None:
        spec = importlib.util.spec_from_file_location(
            "projection_reference", args.reference_module
        )
        reference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_module)
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
            8193 if args.large_tokens else 512, k, torch.bfloat16, torch.device("cuda")
        )
        exchange.workspace.initialize_a2a(
            exchange.parallel,
            k,
            backend=os.environ.get(A2A_ENV_NAME, DEFAULT_A2A_BACKEND),
        )
        exchange.workspace.initialize_reduce_scatter(
            exchange.parallel,
            [n],
            backend=os.environ.get(RS_ENV_NAME, DEFAULT_RS_BACKEND),
        )
        reference_exchange = None
        if reference_module is not None:
            reference_exchange = reference_module.KimiOutputProjection(
                exchange.parallel, k
            )
            if args.reference_shared_workspace:
                # Only for references with the same workspace contract.
                # Serial calls then compare identical communication addresses.
                reference_exchange.workspace = exchange.workspace
            else:
                reference_exchange.workspace = reference_module.ProjectionWorkspace(
                    512, k, torch.bfloat16, torch.device("cuda")
                )
                if hasattr(reference_exchange.workspace, "initialize_a2a"):
                    reference_exchange.workspace.initialize_a2a(exchange.parallel, k)
                if hasattr(reference_exchange.workspace, "initialize_reduce_scatter"):
                    reference_exchange.workspace.initialize_reduce_scatter(
                        exchange.parallel, [n]
                    )
        patterns = [
            [1] * world,
            [8] * world,
            [2] * world,
            [4] * world,
            [16] * world,
            [16 if r < 4 else 0 for r in range(world)],
            [17] * world,
            [32] * world,
            [64] * world,
            [65] * world,
            [257] * world,
            [7 if r == 0 else 0 for r in range(world)],
            [r % 4 for r in range(world)],
            [0] * world,
        ]
        if args.large_tokens:
            patterns += [[rows] * world for rows in (512, 513, 8192, 8193, 512, 513)]
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
            fused_a2a = exchange.workspace.use_flashinfer(exchange.parallel, counts, k)
            using_rsag = exchange.workspace.use_rsag(
                n, max(counts[r] for r in exchange.parallel.tp_group), fused_a2a
            )
            using_peer = (
                exchange.workspace.peer_state(
                    n, max(counts[r] for r in exchange.parallel.tp_group)
                )
                is not None
            )
            custom_reduction = using_rsag or using_peer
            reduction_errors = torch.zeros(2, device="cuda", dtype=torch.float32)
            if custom_reduction and fused_a2a:
                # Same quantized GEMM partials: isolate reduction rounding from
                # weight/activation quantization and TP1 accumulation changes.
                partial, _ = linear(
                    flashinfer_projection_a2a(exchange.workspace.a2a, x.contiguous())
                )
                reduction_reference = partial.float()
                dist.all_reduce(
                    reduction_reference,
                    group=pg_manager.get_process_group(
                        "nccl", exchange.parallel.tp_group
                    ),
                )
                rows = counts[rank]
                offset = exchange.parallel.tp_rank * rows
                reduction_reference = reduction_reference[offset : offset + rows]
                nccl_output = reduce_scatter(
                    partial, exchange.parallel.tp_group, backend=None
                )
                norm = reduction_reference.norm().clamp_min(1e-8)
                reduction_errors[0] = (
                    actual.float() - reduction_reference
                ).norm() / norm
                reduction_errors[1] = (
                    nccl_output.float() - reduction_reference
                ).norm() / norm
                assert reduction_errors[0] <= reduction_errors[1] + 0.001
            dist.all_reduce(reduction_errors, op=dist.ReduceOp.MAX)
            reference_relative_l2 = torch.tensor(
                0.0, device="cuda", dtype=torch.float32
            )
            if reference_exchange is not None:
                prior = reference_exchange.forward(x, linear, counts)
                if custom_reduction:
                    reference_relative_l2 = (
                        actual.float() - prior.float()
                    ).norm() / prior.float().norm().clamp_min(1e-8)
                    assert reference_relative_l2 < 0.01
                else:
                    torch.testing.assert_close(actual, prior, rtol=0, atol=0)
            # Outputs must survive reuse of symmetric scratch by a later layer.
            preserved = actual.clone()
            exchange.forward(x * 0.5, linear, counts)
            torch.testing.assert_close(actual, preserved, rtol=0, atol=0)
            dist.all_reduce(reference_relative_l2, op=dist.ReduceOp.MAX)
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
                for multiplier in (0.5, -1.0, 1e-10, 0.0):
                    x.mul_(multiplier)
                    capture.replay()
                    eager = exchange.forward(x, linear, counts)
                    torch.testing.assert_close(graphed, eager, rtol=0, atol=0)
                    if reference_exchange is not None and using_peer:
                        prior = reference_exchange.forward(x, linear, counts)
                        torch.testing.assert_close(eager, prior, rtol=0, atol=0)
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
                "reference_shared_workspace": args.reference_shared_workspace,
                "layer": layer,
                "counts": counts,
                "reference_relative_l2": reference_relative_l2.item(),
                "reduction_l2_vs_fp32": (
                    reduction_errors.tolist()
                    if custom_reduction and fused_a2a
                    else None
                ),
                "rs_backend": (
                    "triton_peer"
                    if using_peer
                    else "triton_rsag" if using_rsag else "nccl"
                ),
                "a2a_backend": (
                    "flashinfer"
                    if exchange.workspace.use_flashinfer(exchange.parallel, counts, k)
                    else "nccl"
                ),
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
                if reference_exchange is not None:
                    variants = {
                        "tp1": local,
                        "reference": lambda: reference_exchange.forward(
                            x, linear, counts
                        ),
                        "optimized": feature,
                    }
                    samples = []
                    for repeat in range(args.repeats):
                        names = ("tp1", "reference", "optimized")
                        offset = repeat % len(names)
                        order = names[offset:] + names[:offset]
                        samples.append(
                            {
                                name: measure(variants[name], args.iterations, True)
                                for name in order
                            }
                        )
                    record["matched_graph_us"] = samples
                # Component measurements use balanced rows, so every rank
                # participates in the same timing collectives.
                if len(set(counts)) == 1:
                    record["components_us"] = measure_components(
                        exchange, linear, x, counts, args.iterations
                    )
            if rank == 0:
                print(json.dumps(record), flush=True)
        if args.large_tokens:
            validate_backend_transitions(exchange, linear, baseline, k, world)
        if args.profile:
            profile_baseline = (
                baseline
                if reference_exchange is None
                else lambda inputs: reference_exchange.forward(
                    inputs, linear, [16] * world
                )
            )
            label = (
                "tp1_projection"
                if reference_exchange is None
                else "reference_tp4_projection"
            )
            profile_projection(exchange, linear, profile_baseline, k, world, label)
        if reference_exchange is not None and hasattr(
            reference_exchange.workspace, "close"
        ):
            reference_exchange.workspace.close()
        del reference_exchange
        exchange.workspace.close()
        del baseline, linear, exchange, reference_weight
        torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        print("PROJECTION_VALIDATION_PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
