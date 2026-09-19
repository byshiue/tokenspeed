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

"""Behavioral checks for shared-expert collectives and output ownership."""

from test.runtime.distributed.shared_expert_helpers import dep_mapping
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.layers.shared_expert_tp import (
    SharedExpertWorkspace,
    shared_expert_mapping,
    validate_shared_expert_settings,
)


def test_rank_disagreement_fails_before_subgroup_creation():
    mapping = dep_mapping(5, 16)
    module = "tokenspeed.runtime.layers.shared_expert_tp"

    def disagree(values, value, group):
        values[:] = ["1"] + ["4"] * 15

    with mock.patch(f"{module}.dist.is_initialized", return_value=True), mock.patch(
        f"{module}.pg_manager"
    ), mock.patch(f"{module}.dist.all_gather_object", side_effect=disagree):
        with pytest.raises(ValueError, match="differ across ranks"):
            validate_shared_expert_settings(mapping, "1")


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
