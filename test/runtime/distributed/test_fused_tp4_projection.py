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

"""Registered projection dispatch, shared weights/workspace and graph lifetime."""

import os
import sys
from datetime import timedelta
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-2gpu")


@torch.no_grad()
def _worker(rank, rendezvous, result_dir):
    from tokenspeed_kernel.registry import KernelRegistry

    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.linear import (
        DPColumnParallelLinear,
        prepare_dp_linear_communication,
        release_dp_linear_communication,
    )
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
    from tokenspeed.runtime.utils.env import envs

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=300),
        device_id=device,
    )
    parallel = DenseLayerMapping(rank=rank, world_size=4, tp_size=4, dp_size=1)
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=[],
        weight_block_size=[128, 128],
        scale_fmt=None,
    )
    models = []
    for enabled in (False, True):
        with torch.device(device):
            model = torch.nn.ModuleList(
                [
                    DPColumnParallelLinear(
                        7168,
                        49376,
                        padded_output_size=49664,
                        parallel=parallel,
                        params_dtype=torch.bfloat16,
                        quant_config=config,
                        prefix=f"projection_{index}",
                    )
                    for index in range(2)
                ]
            )
        for index, linear in enumerate(model):
            torch.manual_seed(917 + rank + 4 * index)
            linear.weight.copy_(
                torch.randn_like(linear.weight, dtype=torch.bfloat16).to(
                    torch.float8_e4m3fn
                )
            )
            linear.weight_scale_inv.copy_(
                torch.rand_like(linear.weight_scale_inv) * 0.005 + 0.005
            )
        with envs.TOKENSPEED_FUSED_TP4_PROJECTION.override(enabled):
            prepare_dp_linear_communication(model, 256, torch.bfloat16, AutoBackend())
        models.append(model)
    baseline, candidate = models
    workspace = candidate[0].projection_workspace
    assert workspace is candidate[1].projection_workspace
    state = workspace.fused_column
    assert state is not None and baseline[0].projection_workspace.fused_column is None
    assert KernelRegistry.get().get_impl("cute_dsl_fused_tp4_projection") is not None

    def context(counts, physical):
        return ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=physical[rank],
            num_extends=0,
            input_num_tokens=physical[rank],
            forward_mode=ForwardMode.DECODE,
            output_layout=ForwardOutputLayout(0, 0, physical[rank], 1),
            global_num_tokens=counts,
            collective_global_num_tokens=physical,
        )

    for counts in ([128] * 4, [127, 128, 1, 0], [129] * 4, [0] * 4):
        torch.manual_seed(712 + rank)
        x = torch.randn((counts[rank], 7168), device=device, dtype=torch.bfloat16)
        ctx = context(counts, counts)
        epoch = int(state.ready[0])
        for left, right in zip(baseline, candidate):
            expected = left(x, ctx=ctx)[0]
            actual = right(x, ctx=ctx)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert int(state.ready[0]) - epoch == (2 if counts == [128] * 4 else 0)

    # Physical graph counts take precedence over smaller logical counts.
    ctx = context([1] * 4, [128] * 4)
    x = torch.randn((128, 7168), device=device, dtype=torch.bfloat16)
    source = [torch.randn_like(x) for _ in range(3)]
    source[0].zero_()
    expected = [[layer(value, ctx=ctx)[0] for layer in baseline] for value in source]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for layer in candidate:
            layer(x, ctx=ctx)
    stream.synchronize()
    # Exercise device epoch width and successive layers without host joins.
    for flags in (state.ready, state.ack, state.input_flags):
        flags.fill_(2**32 - 4)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = []
        for index in range(12):
            x.copy_(source[index % 3])
            torch.cuda._sleep(25000 * ((rank + index) % 4))
            outputs.append(candidate[index % 2](x, ctx=ctx)[0])
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    for index, actual in enumerate(outputs):
        torch.testing.assert_close(
            actual, expected[index % 3][index % 2], rtol=0, atol=0
        )
    assert int(state.ready[0]) > 2**32
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        graph.replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(f"{result_dir}/fused-rank{rank}.json")
    retained = outputs[0].clone()
    del graph, outputs
    for model in models:
        release_dp_linear_communication(model)
    assert state.closed
    torch.testing.assert_close(retained, expected[0][0], rtol=0, atol=0)
    with envs.TOKENSPEED_FUSED_TP4_PROJECTION.override(True):
        prepare_dp_linear_communication(candidate, 256, torch.bfloat16, AutoBackend())
    assert candidate[0].projection_workspace.fused_column is not state
    x.copy_(source[1])
    torch.testing.assert_close(
        candidate[1](x, ctx=ctx)[0], expected[1][1], rtol=0, atol=0
    )
    release_dp_linear_communication(candidate)
    dist.destroy_process_group()


def test_registered_tp4_projection(tmp_path):
    if torch.cuda.device_count() < 4 or torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("requires four GB300 GPUs")
    mp.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=4,
        join=True,
    )


def test_fused_projection_shape_gate():
    from tokenspeed_kernel.ops.communication.cute_dsl import (
        fused_tp4_projection_supported,
    )

    assert not fused_tp4_projection_supported(
        4, 7168, 49664, 128, torch.bfloat16, torch.device("cpu")
    )
    # Unsupported layouts must not probe CUDA or import optional CuTe code.
    with mock.patch(
        "torch.cuda.get_device_capability", side_effect=AssertionError("CUDA probe")
    ):
        for size, k, n, rows, dtype in (
            (2, 7168, 49664, 128, torch.bfloat16),
            (4, 128, 49664, 128, torch.bfloat16),
            (4, 7168, 49664, 127, torch.bfloat16),
            (4, 7168, 49664, 128, torch.float16),
        ):
            assert not fused_tp4_projection_supported(
                size, k, n, rows, dtype, torch.device("cuda", 0)
            )
