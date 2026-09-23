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

"""Projection shard arithmetic and collective participation on empty owners.

Numerical multi-GPU and graph-lifetime checks run in the real-weight validators.
"""

import os
from test.runtime.distributed.kimi_k3_o_proj_helpers import dep_mapping
from types import SimpleNamespace

import torch

from tokenspeed.runtime.layers.dp_row_parallel_linear import (
    make_output_projection,
    projection_mapping,
)
from tokenspeed.runtime.utils.env import envs

ENV_NAME = envs.TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE.name


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
    # Methods without native output-buffer support retain ordinary Linear
    # semantics, then copy into the communication destination.
    sharded.quant_method = SimpleNamespace(
        apply=lambda layer, x, bias: torch.nn.functional.linear(x, layer.weight, bias)
    )
    inputs = torch.ones(2, 8, dtype=weight.dtype)
    expected, expected_bias = sharded(inputs)
    destination = torch.full_like(expected, float("nan"))
    actual, output_bias = sharded.forward_into(inputs, None, destination)
    assert actual is destination and output_bias is expected_bias
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_independent_projection_flags_and_empty_owner_order(monkeypatch):
    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
    from tokenspeed.runtime.models.kimi_k3 import KimiLinearKDA, KimiLinearMLAAttention

    class RoutedFp8Config(Fp8Config):
        def fp8_pb_wo_route(self, prefix):
            return "w8a8"

    quant = RoutedFp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=[],
        weight_block_size=[128, 128],
        scale_fmt=None,
    )
    config = KimiLinearConfig(
        hidden_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=128,
        kv_lora_rank=128,
        mla_use_output_gate=True,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [],
            "num_heads": 4,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
    )
    mapping = dep_mapping(2, 4)
    for qkv_tp, output_tp in ((1, 1), (1, 4), (4, 1), (4, 4)):
        monkeypatch.setenv(envs.TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE.name, str(qkv_tp))
        monkeypatch.setenv(ENV_NAME, str(output_tp))
        layers = [
            KimiLinearKDA(
                config=config,
                mapping=mapping,
                layer_id=0,
                quant_config=quant,
                prefix="",
            ),
            KimiLinearMLAAttention(
                config=config,
                mapping=mapping,
                hidden_size=512,
                num_heads=4,
                qk_nope_head_dim=128,
                qk_rope_head_dim=0,
                v_head_dim=128,
                q_lora_rank=128,
                kv_lora_rank=128,
                rope_theta=10000,
                rope_scaling=None,
                max_position_embeddings=128,
                quant_config=quant,
                layer_id=0,
                prefix="",
                reduce_attn_results=False,
                alt_stream=None,
            ),
        ]
        for layer in layers:
            calls = []
            counts = [7, 0, 0, 0]

            def exchange(name):
                def forward(inputs, linear, physical_counts):
                    assert physical_counts == counts
                    assert inputs.shape[0] == 0
                    calls.append(name)
                    return inputs.new_empty((0, linear.output_size))

                return SimpleNamespace(forward=forward)

            if qkv_tp > 1:
                layer.input_projection_exchange = exchange("qkv")
                if isinstance(layer, KimiLinearMLAAttention):
                    layer.query_projection_exchange = exchange("q_b")
            if output_tp > 1:
                layer.output_projection_exchange = exchange("o")
            result = layer(
                positions=torch.empty(0, dtype=torch.int64),
                hidden_states=torch.empty(0, 512, dtype=torch.bfloat16),
                ctx=SimpleNamespace(
                    collective_global_num_tokens=counts, global_num_tokens=None
                ),
                comm_manager=None,
                block_scale=None,
                attnres_partial_args=None,
            )
            expected = []
            if qkv_tp > 1:
                expected = (
                    ["qkv", "q_b"]
                    if isinstance(layer, KimiLinearMLAAttention)
                    else ["qkv"]
                )
            if output_tp > 1:
                expected.append("o")
            assert calls == expected
            assert result.shape == (0, 512)
