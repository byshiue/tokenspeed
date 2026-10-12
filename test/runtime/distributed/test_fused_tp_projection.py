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

"""Fused DP column-linear dispatch, shared workspace, and graph lifetime."""

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
def _worker(rank, world_size, rendezvous, result_dir):
    from tokenspeed_kernel.ops.communication.cute_dsl import (
        cute_dsl_fused_fp8_dp_column_linear,
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
        world_size=world_size,
        timeout=timedelta(seconds=300),
        device_id=device,
    )
    parallel = DenseLayerMapping(
        rank=rank, world_size=world_size, tp_size=world_size, dp_size=1
    )
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
                        padded_output_size=(
                            (49376 + world_size * 128 - 1) // (world_size * 128)
                        )
                        * world_size
                        * 128,
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
        with envs.TOKENSPEED_FUSED_TP_PROJECTION.override(enabled):
            prepare_dp_linear_communication(model, 1024, torch.bfloat16, AutoBackend())
        models.append(model)
    baseline, candidate = models
    workspace = candidate[0].projection_workspace
    assert workspace is candidate[1].projection_workspace
    state = workspace.fused_column
    assert state is not None and baseline[0].projection_workspace.fused_column is None
    assert (
        KernelRegistry.get().get_impl("cute_dsl_fused_fp8_dp_column_linear") is not None
    )

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

    def counts_for_group(pattern):
        return (tuple(pattern) * ((world_size + len(pattern) - 1) // len(pattern)))[
            :world_size
        ]

    def uses_fusion(rows):
        return (world_size == 2 and rows >= 64) or (world_size == 4 and rows >= 88)

    equal_counts = [
        1,
        8,
        9,
        16,
        17,
        32,
        33,
        48,
        63,
        64,
        65,
        87,
        88,
        96,
        127,
        128,
        129,
        255,
        256,
        257,
        320,
        383,
        384,
        512,
        513,
        640,
        1024,
    ]
    count_cases = [tuple([rows] * world_size) for rows in equal_counts]
    count_cases += [
        counts_for_group(pattern)
        for pattern in (
            (32, 64, 128, 0),
            (127, 128, 1, 0),
            (0, 65),
            (128, 0),
            (511, 256, 33, 0),
            (1, 0),
            (0, 0),
        )
    ]
    for counts in count_cases:
        counts = counts_for_group(counts)
        fused = uses_fusion(max(counts))
        torch.manual_seed(712 + rank)
        x = torch.randn((counts[rank], 7168), device=device, dtype=torch.bfloat16)
        ctx = context(counts, counts)
        epoch = int(state.ready[0])
        for left, right in zip(baseline, candidate):
            expected = left(x, ctx=ctx)[0]
            actual = right(x, ctx=ctx)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if max(counts) > 0:
                native_expected = expected
                if 17 <= max(counts) * world_size <= 32:
                    # The ordinary path excludes FlashInfer at these total M
                    # values. Compare the native kernel with the canonical
                    # padded GEMM arithmetic, without relaxing output tolerance.
                    padded_inputs = x.new_zeros((32, x.shape[1]))
                    padded_inputs[: x.shape[0]].copy_(x)
                    padded_counts = (32,) * world_size
                    native_expected = left(
                        padded_inputs, ctx=context(padded_counts, padded_counts)
                    )[0][: x.shape[0]]
                out = x.new_empty((counts[rank], right.output_size))
                cute_dsl_fused_fp8_dp_column_linear(
                    state, x, right.weight, right.weight_scale_inv, counts, out
                )
                torch.testing.assert_close(
                    out[:, : right.logical_output_size], native_expected, rtol=0, atol=0
                )
        assert int(state.ready[0]) - epoch == (2 if fused else 0) + (
            2 if max(counts) > 0 else 0
        )
        if max(counts) > 0:
            rows = max(counts)
            bucket = max(32, 1 << (rows - 1).bit_length())
            gathered, _ = baseline[0].comm_backend.projection_all_gather(
                x, rows, False, baseline[0].projection_workspace
            )
            padded = gathered.new_zeros((world_size, bucket, 7168))
            padded[:, :rows].copy_(gathered.view(world_size, rows, 7168))
            values, scales = quantize_fp8(
                padded.view(world_size * bucket, 7168),
                granularity="token_group",
                group_size=128,
                scale_encoding="float32",
            )
            torch.testing.assert_close(
                state.a[: bucket * world_size].view(torch.uint8),
                values.view(torch.uint8),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                state.sa[: bucket * world_size], scales, rtol=0, atol=0
            )

    # Physical graph counts take precedence over smaller logical counts.
    graph_cases = (
        ((32,) * 4, False, False),
        ((33,) * 4, True, True),
        ((48,) * 4, False, False),
        ((64, 17, 0, 63), False, True),
        ((87, 64, 0, 1), False, False),
        ((88,) * 4, False, True),
        ((96,) * 4, False, True),
        ((127, 128, 1, 0), False, True),
        ((129, 255, 128, 0), False, True),
        ((257, 256, 0, 1), False, True),
        ((383, 320, 0, 1), False, True),
        ((257, 512, 33, 1), False, True),
        ((513, 640, 0, 129), False, True),
        ((32,) * 4, True, True),
        ((0, 128), True, True),
        ((128, 0), True, True),
    )
    graph_cases = tuple(
        (
            counts_for_group(counts),
            native,
            native or uses_fusion(max(counts_for_group(counts))),
        )
        for counts, native, _ in graph_cases
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
            torch.cuda._sleep(25000 * ((rank + index) % world_size))
            layer = candidate[repeat % 2]
            if native:
                out = inputs[counts].new_empty((counts[rank], layer.output_size))
                cute_dsl_fused_fp8_dp_column_linear(
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
    torch.testing.assert_close(
        retained, expected[(32,) * world_size][0][0], rtol=0, atol=0
    )
    with envs.TOKENSPEED_FUSED_TP_PROJECTION.override(True):
        prepare_dp_linear_communication(candidate, 32, torch.bfloat16, AutoBackend())
    assert candidate[0].projection_workspace.fused_column is None
    counts = (32,) * world_size
    inputs[counts].copy_(source[counts][1])
    torch.testing.assert_close(
        candidate[1](inputs[counts], ctx=contexts[counts])[0],
        expected[counts][1][1],
        rtol=0,
        atol=0,
    )
    release_dp_linear_communication(candidate)
    with envs.TOKENSPEED_FUSED_TP_PROJECTION.override(True):
        prepare_dp_linear_communication(candidate, 88, torch.bfloat16, AutoBackend())
    assert candidate[0].projection_workspace.fused_column is not None
    counts = (88,) * world_size
    inputs[counts].copy_(source[counts][2])
    torch.testing.assert_close(
        candidate[0](inputs[counts], ctx=contexts[counts])[0],
        expected[counts][2][0],
        rtol=0,
        atol=0,
    )
    release_dp_linear_communication(candidate)
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4, 8))
def test_registered_tp_projection(tmp_path, world_size):
    if torch.cuda.device_count() < world_size or torch.cuda.get_device_capability() != (
        10,
        3,
    ):
        pytest.skip(f"requires {world_size} node-local GB300 GPUs")
    mp.spawn(
        _worker,
        args=(world_size, f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=world_size,
        join=True,
    )


def test_fused_projection_shape_gate():
    from tokenspeed_kernel.ops.communication.cute_dsl import (
        fused_tp_projection_supported,
    )

    assert not fused_tp_projection_supported(
        4, 7168, 49664, 128, torch.bfloat16, torch.device("cpu")
    )
    # Unsupported layouts must not probe CUDA or import optional CuTe code.
    with mock.patch(
        "torch.cuda.get_device_capability", side_effect=AssertionError("CUDA probe")
    ):
        for size, k, n, rows, dtype in (
            (1, 7168, 49408, 128, torch.bfloat16),
            (3, 7168, 49536, 128, torch.bfloat16),
            (2, 7168, 49664, 128, torch.bfloat16),
            (8, 7168, 49664, 128, torch.bfloat16),
            (4, 128, 49664, 128, torch.bfloat16),
            (4, 7168, 49664, 0, torch.bfloat16),
            (4, 7168, 49664, 128, torch.float16),
        ):
            assert not fused_tp_projection_supported(
                size, k, n, rows, dtype, torch.device("cuda", 0)
            )


@pytest.mark.parametrize(
    ("world_size", "output_size"), ((2, 49408), (4, 49664), (8, 50176))
)
def test_fused_projection_aligned_layout(world_size, output_size):
    from tokenspeed_kernel.ops.communication.cute_dsl import (
        fused_tp_projection_supported,
    )

    with (
        mock.patch(
            "tokenspeed_kernel.ops.communication.cute_dsl.current_platform"
        ) as platform,
        mock.patch("torch.cuda.get_device_capability", return_value=(10, 3)),
        mock.patch("torch.cuda.get_device_properties") as properties,
    ):
        platform.return_value.is_nvidia = True
        properties.return_value.multi_processor_count = 152
        assert fused_tp_projection_supported(
            world_size, 7168, output_size, 128, torch.bfloat16, torch.device("cuda", 0)
        )
        properties.return_value.multi_processor_count = 148
        assert not fused_tp_projection_supported(
            world_size, 7168, output_size, 128, torch.bfloat16, torch.device("cuda", 0)
        )
