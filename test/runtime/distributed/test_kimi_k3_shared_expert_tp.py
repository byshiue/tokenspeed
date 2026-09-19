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

"""CPU contract tests for shared-expert TP mapping, sharding and empty owners."""

from test.runtime.distributed.shared_expert_helpers import dep_mapping
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.layers.shared_expert_tp import (
    SharedExpertWorkspace,
    shared_expert_mapping,
    validate_shared_expert_settings,
)
from tokenspeed.runtime.models.kimi_k3 import KimiLinearMLP


def test_mapping_and_disabled_mlp(monkeypatch):
    mapping = dep_mapping(5, 16)
    assert shared_expert_mapping(mapping, "1") is None
    parallel = shared_expert_mapping(mapping, "4")
    assert parallel.tp_group == (4, 5, 6, 7)
    with mock.patch(
        "tokenspeed.runtime.layers.shared_expert_tp.dist.is_initialized",
        return_value=False,
    ):
        assert validate_shared_expert_settings(mapping, "1") is None
    module = "tokenspeed.runtime.layers.shared_expert_tp"

    def disagree(values, value, group):
        values[:] = ["1"] + ["4"] * 15

    with mock.patch(f"{module}.dist.is_initialized", return_value=True), mock.patch(
        f"{module}.pg_manager"
    ), mock.patch(f"{module}.dist.all_gather_object", side_effect=disagree):
        with pytest.raises(ValueError, match="differ across ranks"):
            validate_shared_expert_settings(mapping, "1")
    for value in ("0", "2", "bad"):
        with pytest.raises(ValueError):
            shared_expert_mapping(mapping, value)
    with pytest.raises(ValueError):
        shared_expert_mapping(dep_mapping(0, 2), "4")
    monkeypatch.setenv("TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE", "1")
    with torch.device("meta"):
        layer = KimiLinearMLP(
            7168,
            6144,
            tp_rank=0,
            tp_size=1,
            tp_group=None,
            shared_parallel=None,
            quant_config=None,
            prefix="shared_experts",
            reduce_results=False,
            is_shared_expert=True,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
        )
    assert layer.shared_parallel is None
    assert layer.gate_up_proj.weight.shape == (12288, 7168)


def test_mlp_shards_match_gate_up_and_down(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE", "4")
    parallel = shared_expert_mapping(dep_mapping(5, 16), "4")
    with torch.device("meta"):
        layer = KimiLinearMLP(
            7168,
            6144,
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
            tp_group=parallel.tp_group,
            shared_parallel=parallel,
            quant_config=None,
            prefix="shared_experts",
            reduce_results=False,
            is_shared_expert=True,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
        )
    assert layer.gate_up_proj.weight.shape == (3072, 7168)
    assert layer.down_proj.weight.shape == (7168, 1536)
    assert not layer.down_proj.reduce_results
    with pytest.raises(RuntimeError, match="prepared"):
        layer.forward_shared_tp(torch.empty(0, 7168), [0] * 16)


def test_empty_owner_participates_but_empty_group_skips():
    workspace = SharedExpertWorkspace.__new__(SharedExpertWorkspace)
    workspace.parallel = shared_expert_mapping(dep_mapping(0, 4), "4")
    workspace.capacity, workspace.hidden = 8, 2
    workspace.send = torch.empty(8, 2, dtype=torch.bfloat16)
    workspace.gather = object()
    workspace.reduction = object()
    compute = mock.Mock(side_effect=lambda x, down_out: x)
    with mock.patch(
        "tokenspeed.runtime.layers.shared_expert_tp.trtllm_shared_expert_allgather",
        return_value=torch.zeros(8, 2, dtype=torch.bfloat16),
    ) as gather, mock.patch(
        "tokenspeed.runtime.layers.shared_expert_tp.trtllm_shared_expert_reduce_scatter",
        return_value=torch.ones(2, 2, dtype=torch.bfloat16),
    ) as reduction:
        output = workspace.forward(
            torch.empty(0, 2, dtype=torch.bfloat16), [0, 2, 1, 0], compute
        )
        assert output.shape == (0, 2)
        gather.assert_called_once()
        reduction.assert_called_once()
        assert torch.count_nonzero(workspace.send[:2]) == 0
        workspace.forward(torch.empty(0, 2, dtype=torch.bfloat16), [0] * 4, compute)
        assert gather.call_count == 1
        assert reduction.call_count == 1
    with pytest.raises(ValueError):
        workspace.forward(torch.empty(1, 2, dtype=torch.bfloat16), [0] * 4, compute)
    with pytest.raises(ValueError, match="capacity"):
        workspace.forward(
            torch.empty(0, 2, dtype=torch.bfloat16), [0, 9, 0, 0], compute
        )


def test_collective_capacity_boundary_and_owned_rows():
    workspace = SharedExpertWorkspace.__new__(SharedExpertWorkspace)
    workspace.parallel = shared_expert_mapping(dep_mapping(0, 4), "4")
    workspace.capacity, workspace.hidden = 129, 2
    workspace.gather, workspace.reduction = object(), object()
    workspace.received = torch.empty(4 * 129, 2, dtype=torch.bfloat16)
    module = "tokenspeed.runtime.layers.shared_expert_tp"
    with mock.patch(
        f"{module}.trtllm_shared_expert_allgather",
        side_effect=lambda state, x: x.repeat(4, 1),
    ) as native_ag, mock.patch(f"{module}.all_gather_single") as nccl_ag, mock.patch(
        f"{module}.trtllm_shared_expert_reduce_scatter",
        side_effect=lambda state, x, rows: x[:rows].clone(),
    ) as native_rs, mock.patch(
        f"{module}.reduce_scatter",
        side_effect=lambda x, group, backend: x[: x.shape[0] // 4].clone(),
    ) as nccl_rs:
        for rows in (128, 129):
            inputs = torch.ones(rows, 2, dtype=torch.bfloat16)
            assert workspace.gather_inputs(inputs, [rows] * 4).shape == (4 * rows, 2)
            partial = inputs.repeat(4, 1)
            output = workspace.reduce_outputs(partial, rows - 1)
            partial.zero_()
            assert output.shape == (rows - 1, 2)
            assert torch.all(output == 1)
        native_ag.assert_called_once()
        nccl_ag.assert_called_once()
        native_rs.assert_called_once()
        nccl_rs.assert_called_once()
