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

"""Real Kimi-K3 attention forwards with replicated and weight-prefetched O projections.

Run with torchrun on 4 or 16 GPUs and --model pointing to the real checkpoint.
This is attention-module validation, not a reduced-depth model accuracy test.
"""

import argparse
import json
import os
from pathlib import Path
from test.runtime.conftest import block_tables_for
from test.runtime.distributed.kimi_k3_o_proj_helpers import dep_mapping
from test.runtime.test_kimi_k3_kda import _stub_contract, _StubContractPool
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors import safe_open

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    active_forward,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.dp_linear_communication import (
    initialize_projection_group,
    projection_mapping,
)
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig
from tokenspeed.runtime.models.kimi_k3 import KimiLinearForCausalLM, KimiLinearKDA


class AttentionLayer(torch.nn.Module):
    def __init__(self, config, mapping, quant, layer_id):
        super().__init__()
        self.self_attn = KimiLinearKDA(
            config,
            mapping,
            layer_id,
            quant_config=quant,
            prefix=f"model.layers.{layer_id}.self_attn",
        )


class AttentionTree(torch.nn.Module):
    def __init__(self, config, mapping, quant):
        super().__init__()
        self.pp_start_layer, self.pp_end_layer = 0, 2
        self.layers = torch.nn.ModuleList(
            [AttentionLayer(config, mapping, quant, layer) for layer in range(2)]
        )


class AttentionOnlyModel(KimiLinearForCausalLM):
    """Reuse production loading, startup and KDA forward without allocating MoE."""

    def __init__(self, root, mapping, weight_tp, compute_tp, max_rows):
        torch.nn.Module.__init__(self)
        self.config = KimiLinearConfig(
            **json.loads((root / "config.json").read_text())["text_config"]
        )
        self.mapping = mapping
        self.quant_config = ModelOptMixedConfig.from_config(
            json.loads((root / "hf_quant_config.json").read_text())
        )
        self.quant_config.apply_checkpoint_name_replacements((("language_model.", ""),))
        with patch.dict(
            os.environ,
            {
                "TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE": str(compute_tp),
                "TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE": "1",
                "TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE": "1",
                "TOKENSPEED_KIMI_K3_O_PROJ_WEIGHT_TP_SIZE": str(weight_tp),
            },
        ), torch.device("cuda"):
            self.model = AttentionTree(self.config, mapping, self.quant_config)
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

        def weights():
            for name, shard in index.items():
                if any(
                    name.startswith(f"language_model.model.layers.{layer}.self_attn.")
                    for layer in range(2)
                ):
                    with safe_open(
                        root / shard, framework="pt", device="cpu"
                    ) as handle:
                        yield name.removeprefix("language_model."), handle.get_tensor(
                            name
                        )

        self.load_weights(weights())
        for module in self.modules():
            method = getattr(module, "quant_method", None)
            if method is not None:
                method.process_weights_after_loading(module)
        self.prepare_communication_runtime(max_rows)


class KdaRuntimeFixture:
    """Real backend and cache metadata with distinct immutable-read/write pages."""

    def __init__(self, config, mapping, max_rows):
        self.mapping, self.max_rows = mapping, max_rows
        self.config = config
        la = config.linear_attn_config
        self.contract = _stub_contract(
            prefix_granularity=128, usable_pages=2 * max_rows + 1
        )
        self.pool = _StubContractPool(
            self.contract,
            "cuda",
            3 * la["num_heads"] * la["head_dim"],
            la["short_conv_kernel_size"],
            la["num_heads"],
            la["head_dim"],
        )
        spec = MHAConfig(
            num_attention_heads=la["num_heads"],
            num_kv_heads=la["num_heads"],
            head_dim=la["head_dim"],
            attn_tp_size=1,
        )
        attn_config = AttnConfig(
            device="cuda",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            kv_cache_quant_method="none",
            prefix_granularity=128,
            context_len=4096,
            max_bs=max_rows,
            is_draft=False,
            speculative_num_draft_tokens=1,
            components=(spec,),
        )
        self.backend = KdaAttnBackend(
            attn_config, spec, enable_prefill_graph=True, kda_backend="cutedsl_kda"
        )
        self.backend.set_kv_pool(self.pool)
        self.backend.init_cuda_graph_state(max_bs=max_rows)
        self.hybrid = HybridLinearAttnBackend(self.backend, self.backend, [])
        self.ctx = None
        self.inputs = None
        self.positions = None

    def set_rows(self, counts):
        rows = counts[self.mapping.rank]
        self.inputs = (
            torch.randn(
                rows, self.config.hidden_size, device="cuda", dtype=torch.bfloat16
            )
            * 0.1
        )
        self.positions = torch.full((rows,), 128, device="cuda", dtype=torch.int64)
        if rows:
            table = torch.stack(
                (
                    torch.arange(1, rows + 1, dtype=torch.int32),
                    torch.arange(
                        self.max_rows + 1, self.max_rows + rows + 1, dtype=torch.int32
                    ),
                ),
                dim=1,
            )
            tables = {spec.group_id: table for spec in self.contract.group_specs}
            self.backend.refresh_decode_metadata(
                rows,
                rows,
                torch.arange(rows, device="cuda", dtype=torch.int32),
                torch.full((rows,), 129, device="cuda", dtype=torch.int32),
                forward_mode=ForwardMode.DECODE,
                block_tables=block_tables_for(self.contract, tables, "cuda"),
            )
        self.ctx = SimpleNamespace(
            attn_backend=self.hybrid,
            token_to_kv_pool=self.pool,
            forward_mode=ForwardMode.DECODE,
            bs=rows,
            num_extends=0,
            global_num_tokens=counts,
            collective_global_num_tokens=counts,
        )

    def forward(self, model, layer):
        return model.model.layers[layer].self_attn(
            self.positions,
            self.inputs,
            self.ctx,
            None,
            block_scale=None,
            attnres_partial_args=None,
        )

    def pair(self, model):
        return self.forward(model, 0), self.forward(model, 1)

    def set_prefill(self, lengths_list):
        self.set_rows([2] * self.mapping.world_size)
        lengths = torch.tensor(lengths_list, dtype=torch.int32)
        rows = sum(lengths_list)
        prefixes = torch.full_like(lengths, 128)
        table = torch.tensor(
            [[1, self.max_rows + 1], [2, self.max_rows + 2]], dtype=torch.int32
        )
        self.backend.init_forward_metadata(
            bs=2,
            num_extends=2,
            req_pool_indices=torch.arange(2, dtype=torch.int32, device="cuda"),
            seq_lens=(prefixes + lengths).cuda(),
            forward_mode=ForwardMode.EXTEND,
            block_tables=block_tables_for(
                self.contract,
                {spec.group_id: table for spec in self.contract.group_specs},
                "cuda",
            ),
            extend_seq_lens=lengths.cuda(),
            extend_seq_lens_cpu=lengths,
            extend_prefix_lens=prefixes.cuda(),
            extend_prefix_lens_cpu=prefixes,
            extend_replay_lens_cpu=torch.zeros_like(prefixes),
            extend_prompt_lens_cpu=prefixes + lengths,
            extend_with_prefix=True,
        )
        self.inputs = (
            torch.randn(
                rows, self.config.hidden_size, device="cuda", dtype=torch.bfloat16
            )
            * 0.1
        )
        self.positions = torch.cat(
            [
                torch.arange(128, 128 + length, device="cuda", dtype=torch.int64)
                for length in lengths_list
            ]
        )
        self.ctx.forward_mode = ForwardMode.EXTEND
        self.ctx.num_extends = 2
        self.ctx.global_num_tokens = self.ctx.collective_global_num_tokens = [
            rows
        ] * self.mapping.world_size


def setup(root, tp_size, max_rows):
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(813 + rank)
    mapping = dep_mapping(rank, world)
    pg_manager.init_distributed(
        mapping,
        distributed_init_method="env://",
        backend="nccl",
        timeout=600,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )
    parallel = projection_mapping(rank, world, tp_size)
    initialize_projection_group(parallel)
    baseline = AttentionOnlyModel(root, mapping, 1, 1, max_rows)
    candidate = AttentionOnlyModel(root, mapping, tp_size, 1, max_rows)
    fixture = KdaRuntimeFixture(baseline.config, mapping, max_rows)
    return baseline, candidate, fixture


def assert_output(actual, expected, full_weight):
    if full_weight or actual.numel() == 0:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    else:
        # K-sharded GEMM changes accumulation/rounding, as ordinary compute TP
        # does. Keep the existing projection tolerance against replicated FP8.
        delta = (actual.float() - expected.float()).norm()
        assert delta / expected.float().norm().clamp_min(1e-12) < 0.015
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)


@torch.no_grad()
def validate(baseline, candidate, compute, fixture, counts):
    fixture.set_rows(counts)
    workspace = candidate.model.layers[0].self_attn.o_proj.workspace
    large = counts[fixture.mapping.rank] >= 128
    for _ in range(2):
        fixture.pair(candidate)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graphed = fixture.pair(candidate)
    for multiplier in (1.0, 0.5, -1.0):
        fixture.inputs.mul_(multiplier)
        expected = fixture.pair(baseline)
        for actual, reference in zip(fixture.pair(candidate), expected):
            assert_output(actual, reference, large)
        # Below threshold the hybrid must execute the unchanged compute-TP path.
        if max(counts) < 128:
            for actual, reference in zip(
                fixture.pair(candidate), fixture.pair(compute)
            ):
                torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        workspace.weight_storage.view(torch.uint8).fill_(255)
        workspace.scale_storage.fill_(float("nan"))
        for _, _, plan in workspace.views.values():
            plan.prepared_weight_scales.fill_(float("nan"))
        if fixture.mapping.rank % 4 == 0:
            torch.cuda._sleep(100000)
        graph.replay()
        for actual, reference in zip(graphed, expected):
            assert_output(actual, reference, large)
        if large:
            linear = candidate.model.layers[1].self_attn.o_proj
            weight, scales, _ = workspace.views[
                (linear.logical_output_size, linear.input_size)
            ]
            reference = baseline.model.layers[1].self_attn.o_proj
            torch.testing.assert_close(
                weight.view(torch.uint8),
                reference.weight.view(torch.uint8),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                scales, reference.weight_scale_inv, atol=0, rtol=0
            )
        else:
            # A small/idle owner must not refresh any full weight, even while
            # contributing its K shard to another owner's activation collective.
            assert torch.all(workspace.weight_storage.view(torch.uint8) == 255)
    graph.reset()
    # Exercise the same eager-attention boundary used by breakable prefill
    # graphs, including repeated scratch refresh after the captured segment.
    with active_forward(fixture.ctx):
        capture = BreakableCapture()
        with capture:
            output = fixture.forward(candidate, 0)
        for _ in range(2):
            fixture.inputs.mul_(0.75)
            reference = fixture.forward(baseline, 0)
            capture.replay()
            assert_output(output, reference, large)
    del capture, output
    torch.cuda.synchronize()
    dist.barrier()


@torch.no_grad()
def validate_prefill(baseline, candidate, compute, fixture, lengths):
    workspace = candidate.model.layers[0].self_attn.o_proj.workspace
    for inline_attention in (False, True):
        fixture.set_prefill(lengths)
        if inline_attention:
            assert fixture.backend.prepare_prefill_metadata(
                sum(lengths), 2, ForwardMode.EXTEND, capture=True
            )
        for _ in range(2):
            fixture.forward(candidate, 0)
        with active_forward(fixture.ctx):
            capture = BreakableCapture()
            with capture:
                output = fixture.forward(candidate, 0)
            for _ in range(2):
                fixture.inputs.mul_(0.75)
                reference = fixture.forward(baseline, 0)
                workspace.weight_storage.zero_()
                capture.replay()
                assert_output(output, reference, sum(lengths) >= 128)
                if sum(lengths) < 128:
                    torch.testing.assert_close(
                        output, fixture.forward(compute, 0), atol=0, rtol=0
                    )
                    assert (
                        torch.count_nonzero(workspace.weight_storage.view(torch.uint8))
                        == 0
                    )
        del capture, output
        torch.cuda.synchronize()
        dist.barrier()
        if fixture.mapping.rank == 0:
            print(
                json.dumps(
                    {
                        "prefill_inline_attention": inline_attention,
                        "rows": sum(lengths),
                        "passed": True,
                    }
                ),
                flush=True,
            )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    args = parser.parse_args()
    baseline, candidate, fixture = setup(args.model, args.tp_size, 257)
    compute = AttentionOnlyModel(args.model, fixture.mapping, 1, args.tp_size, 257)
    world = fixture.mapping.world_size
    for counts in (
        [1] * world,
        [3] * world,
        [127] * world,
        [128] * world,
        [129] * world,
        [257] * world,
        [0 if rank % args.tp_size == 0 else 7 for rank in range(world)],
        [(0, 127, 128, 129)[rank % 4] for rank in range(world)],
        [128 if rank % args.tp_size else 0 for rank in range(world)],
        [3 if rank % args.tp_size == 0 else 0 for rank in range(world)],
        [0] * world,
    ):
        validate(baseline, candidate, compute, fixture, counts)
        if fixture.mapping.rank == 0:
            print(json.dumps({"counts": counts, "passed": True}), flush=True)
    for lengths in ((7, 10), (63, 64), (64, 64), (64, 65)):
        validate_prefill(baseline, candidate, compute, fixture, lengths)
    for layer in candidate.model.layers:
        layer.self_attn.o_proj.source.close()
    candidate.model.layers[0].self_attn.o_proj.communication.close()
    compute.model.layers[0].self_attn.o_proj.communication.close()
    dist.destroy_process_group()
    if fixture.mapping.rank == 0:
        print("WEIGHT_PREFETCH_RUNTIME_VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
