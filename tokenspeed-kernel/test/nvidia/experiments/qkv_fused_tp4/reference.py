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

"""Local real-checkpoint projection validation/benchmark, not full-model E2E."""

import json
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.linear import (
    prepare_dp_linear_communication,
    release_dp_linear_communication,
)
from tokenspeed.runtime.layers.quantization.modelopt_mixed import ModelOptMixedConfig
from tokenspeed.runtime.models.kimi_k3 import (
    KimiLinearForCausalLM,
    KimiLinearKDA,
    KimiLinearMLAAttention,
)
from tokenspeed.runtime.models.kimi_k3_projection import validate_projection_settings


class ProjectionCheckpoint(KimiLinearForCausalLM):
    """Only real attention layers 0 and 3; use the production checkpoint loader."""

    def __init__(self, config, mapping, quant_config, parallel, device):
        torch.nn.Module.__init__(self)
        self.config, self.mapping, self.quant_config = config, mapping, quant_config
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
        self.model.pp_start_layer, self.model.pp_end_layer = 0, 4
        with torch.device(device):
            self.model.layers[0].self_attn = KimiLinearKDA(
                config,
                mapping,
                0,
                quant_config=quant_config,
                prefix="model.layers.0.self_attn",
                qkv_parallel=parallel,
                output_parallel=None,
            )
            self.model.layers[3].self_attn = KimiLinearMLAAttention(
                config=config,
                mapping=mapping,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                quant_config=quant_config,
                layer_id=3,
                prefix="model.layers.3.self_attn",
                qkv_parallel=parallel,
                output_parallel=None,
            )


def checkpoint(root, device):
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = [
        name
        for name in index
        if name.startswith(
            (
                "language_model.model.layers.0.self_attn.",
                "language_model.model.layers.3.self_attn.",
            )
        )
    ]
    with ExitStack() as stack:
        handles = {
            file: stack.enter_context(
                safe_open(root / file, framework="pt", device=device.index)
            )
            for file in sorted({index[name] for name in names})
        }
        for name in names:
            yield name.removeprefix("language_model."), handles[index[name]].get_tensor(
                name
            )


def context(rank, counts):
    rows = counts[rank]
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=rows,
        num_extends=0,
        input_num_tokens=rows,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, rows, 1),
        global_num_tokens=counts,
    )
