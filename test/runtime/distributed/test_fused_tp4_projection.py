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
    from tokenspeed_kernel.ops.communication.cute_dsl import (
        cute_dsl_fused_tp4_projection,
    )
    from tokenspeed_kernel.ops.quantization import quantize_fp8
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
            prepare_dp_linear_communication(model, 1024, torch.bfloat16, AutoBackend())
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

    for counts, fused in (
        ([1] * 4, False),
        ([16] * 4, False),
        ([32] * 4, False),
        ([33] * 4, False),
        ([48] * 4, False),
        ([63] * 4, False),
        ([64] * 4, True),
        ([65] * 4, False),
        ([87] * 4, False),
        ([88] * 4, True),
        ([96] * 4, True),
        ([127] * 4, True),
        ([128] * 4, True),
        ([32, 64, 128, 0], True),
        ([127, 128, 1, 0], True),
        ([129] * 4, True),
        ([255] * 4, True),
        ([256] * 4, True),
        ([257] * 4, False),
        ([383] * 4, False),
        ([384] * 4, True),
        ([512] * 4, True),
        ([513] * 4, True),
        ([640] * 4, True),
        ([1024] * 4, True),
        ([0, 0, 0, 65], False),
        ([511, 256, 33, 0], True),
        ([1, 0, 0, 0], False),
        ([0] * 4, False),
    ):
        torch.manual_seed(712 + rank)
        x = torch.randn((counts[rank], 7168), device=device, dtype=torch.bfloat16)
        ctx = context(counts, counts)
        epoch = int(state.ready[0])
        for left, right in zip(baseline, candidate):
            expected = left(x, ctx=ctx)[0]
            actual = right(x, ctx=ctx)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert int(state.ready[0]) - epoch == (2 if fused else 0)
        if fused:
            rows = max(counts)
            bucket = max(32, 1 << (rows - 1).bit_length())
            gathered, _ = baseline[0].comm_backend.projection_all_gather(
                x, rows, False, baseline[0].projection_workspace
            )
            padded = gathered.new_zeros((4, bucket, 7168))
            padded[:, :rows].copy_(gathered.view(4, rows, 7168))
            values, scales = quantize_fp8(
                padded.view(4 * bucket, 7168),
                granularity="token_group",
                group_size=128,
                scale_encoding="float32",
            )
            torch.testing.assert_close(
                state.a[: bucket * 4].view(torch.uint8),
                values.view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(state.sa[: bucket * 4], scales, rtol=0, atol=0)

    # Physical graph counts take precedence over smaller logical counts.
    graph_cases = (
        ((32,) * 4, False, False),
        ((33,) * 4, True, True),
        ((48,) * 4, False, False),
        ((64, 17, 0, 63), False, True),
        ((88,) * 4, False, True),
        ((96,) * 4, False, True),
        ((127, 128, 1, 0), False, True),
        ((129, 255, 128, 0), False, True),
        ((257, 512, 33, 1), False, True),
        ((513, 640, 0, 129), False, True),
        ((32,) * 4, True, True),
    )
    contexts = {
        counts: context([int(n > 0) for n in counts], list(counts))
        for counts, _, _ in graph_cases
    }
    inputs = {
        counts: torch.randn((counts[rank], 7168), device=device, dtype=torch.bfloat16)
        for counts in contexts
    }
    source = {
        counts: [torch.randn_like(value) for _ in range(3)]
        for counts, value in inputs.items()
    }
    expected = {}
    for counts in contexts:
        source[counts][0].zero_()
        expected[counts] = [
            [layer(value, ctx=contexts[counts])[0] for layer in baseline]
            for value in source[counts]
        ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for counts in contexts:
            for layer in candidate:
                layer(inputs[counts], ctx=contexts[counts])
    stream.synchronize()
    # Exercise device epoch width and successive layers without host joins.
    for flags in (state.ready, state.ack, state.input_flags):
        flags.fill_(2**32 - 4)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    # Interleave automatic 32-row fallback with padded and unequal fused calls.
    # Explicit 32-row fusion remains available for kernel experiments.
    calls = graph_cases * 6
    with torch.cuda.graph(graph, stream=stream):
        outputs = []
        for index, (counts, native, _) in enumerate(calls):
            repeat = index // len(graph_cases)
            inputs[counts].copy_(source[counts][repeat % 3])
            torch.cuda._sleep(25000 * ((rank + index) % 4))
            layer = candidate[repeat % 2]
            if native:
                out = inputs[counts].new_empty((counts[rank], layer.output_size))
                cute_dsl_fused_tp4_projection(
                    state,
                    inputs[counts],
                    layer.weight,
                    layer.weight_scale_inv,
                    counts,
                    out,
                )
                outputs.append(out[:, : layer.logical_output_size])
            else:
                outputs.append(layer(inputs[counts], ctx=contexts[counts])[0])
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    for index, actual in enumerate(outputs):
        counts, _, _ = calls[index]
        repeat = index // len(graph_cases)
        torch.testing.assert_close(
            actual, expected[counts][repeat % 3][repeat % 2], rtol=0, atol=0
        )
    fused_calls = sum(fused for _, _, fused in calls)
    assert int(state.ready[0]) == 2**32 - 4 + 3 * fused_calls
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
    torch.testing.assert_close(retained, expected[(32,) * 4][0][0], rtol=0, atol=0)
    with envs.TOKENSPEED_FUSED_TP4_PROJECTION.override(True):
        prepare_dp_linear_communication(candidate, 32, torch.bfloat16, AutoBackend())
    assert candidate[0].projection_workspace.fused_column is None
    counts = (32,) * 4
    inputs[counts].copy_(source[counts][1])
    torch.testing.assert_close(
        candidate[1](inputs[counts], ctx=contexts[counts])[0],
        expected[counts][1][1],
        rtol=0,
        atol=0,
    )
    release_dp_linear_communication(candidate)
    with envs.TOKENSPEED_FUSED_TP4_PROJECTION.override(True):
        prepare_dp_linear_communication(candidate, 88, torch.bfloat16, AutoBackend())
    assert candidate[0].projection_workspace.fused_column is not None
    counts = (88,) * 4
    inputs[counts].copy_(source[counts][2])
    torch.testing.assert_close(
        candidate[0](inputs[counts], ctx=contexts[counts])[0],
        expected[counts][2][0],
        rtol=0,
        atol=0,
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
            (4, 7168, 49664, 0, torch.bfloat16),
            (4, 7168, 49664, 128, torch.float16),
        ):
            assert not fused_tp4_projection_supported(
                size, k, n, rows, dtype, torch.device("cuda", 0)
            )
