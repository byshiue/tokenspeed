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

"""Local staged projection experiment; all measurements are fresh GPU runs."""

import importlib.util
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import reference


def initialize(model_path):
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda", local)
    mapping = reference.Mapping(
        rank=rank,
        world_size=world,
        attn_tp_size=1,
        attn_dp_size=world,
        linear_attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=world,
    )
    reference.process_group_manager.init_distributed(
        mapping,
        distributed_init_method="env://",
        backend="nccl",
        timeout=300,
        device_id=device,
    )
    parallel, _ = reference.validate_projection_settings(mapping, world, 1)
    raw = json.loads((model_path / "config.json").read_text())
    config = reference.KimiLinearConfig(**raw["text_config"])
    quant = reference.ModelOptMixedConfig.from_config(raw["quantization_config"])
    models = []
    for parallelism in (None, parallel):
        model = reference.ProjectionCheckpoint(
            config, mapping, quant, parallelism, device
        )
        model.load_weights(reference.checkpoint(model_path, device))
        for module in model.modules():
            method = getattr(module, "quant_method", None)
            if method is not None:
                method.process_weights_after_loading(module)
        reference.prepare_dp_linear_communication(
            model, 128, torch.bfloat16, reference.AutoBackend()
        )
        models.append(model)
    return rank, world, device, models


def measure(calls, repeats, graph_calls, replay_count):
    graphs = []
    for name, call in calls.items():
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_calls):
                call()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        graphs.append((name, graph))
    samples = {name: [] for name in calls}
    for repeat in range(repeats):
        ordered = graphs if repeat % 2 == 0 else list(reversed(graphs))
        for name, graph in ordered:
            dist.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(replay_count):
                graph.replay()
            end.record()
            end.synchronize()
            elapsed = torch.tensor(
                start.elapsed_time(end) * 1000 / (graph_calls * replay_count),
                device="cuda",
                dtype=torch.float32,
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            samples[name].append(elapsed.item())
    return {
        name: {"median_us": statistics.median(values), "samples_us": values}
        for name, values in samples.items()
    }
