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


"""Deferred-state feedback: D2H ordering, no publication on failure, rank agreement."""

from concurrent.futures import Future
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.engine.state_commit import StateCommitValidator
from tokenspeed.runtime.execution.types import ModelExecutionResult, PendingExecution


def test_state_validity_contract_and_disabled_consumers():
    validator = StateCommitValidator(enabled=True, local_groups=2, groups=())
    validator.validate(None, bs=3, requires_commit=False)
    validator.validate(torch.ones(2, 3, dtype=torch.bool), bs=3, requires_commit=True)
    for flags in (
        None,
        torch.ones(2, 4, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.int32),
        torch.tensor([[True, False, True]] * 2),
    ):
        with pytest.raises(RuntimeError, match="must not be published"):
            validator.validate(flags, bs=3, requires_commit=True)
    # A failed check cannot contaminate the next result. Other models add no
    # collective or state check; an empty pipeline stage owes no local flags.
    validator.validate(torch.ones(2, 3, dtype=torch.bool), bs=3, requires_commit=True)
    StateCommitValidator(enabled=True, local_groups=0, groups=()).validate(
        None, bs=3, requires_commit=True
    )
    StateCommitValidator(enabled=False, local_groups=0, groups=(object(),)).validate(
        None, bs=3, requires_commit=True
    )


def test_invalid_commit_stops_feedback_after_copy_completion():
    from tokenspeed.runtime.engine.event_loop import EventLoop

    flags = torch.ones(2, 1, dtype=torch.bool)
    order = []

    def copied():
        order.append("copy completed")
        flags[1, 0] = False

    result = ModelExecutionResult(
        output_tokens=torch.tensor([7]),
        output_lengths=torch.tensor([1]),
        copy_event=SimpleNamespace(synchronize=copied),
        state_commit_validity=flags,
    )
    future = Future()
    future.set_result(result)
    pending = PendingExecution(future)
    loop = SimpleNamespace(
        _state_commit_validator=StateCommitValidator(
            enabled=True, local_groups=2, groups=()
        ),
        request_handler=SimpleNamespace(forward_ct=0),
        output_processor=Mock(),
        _pp_broadcast_output_tokens=Mock(),
    )
    op = SimpleNamespace(request_ids=["request"], num_extends=lambda: 0)
    with pytest.raises(RuntimeError, match="must not be published"):
        EventLoop._commit_forward_results(loop, op, pending)
    assert order == ["copy completed"] and loop.request_handler.forward_ct == 0
    loop.output_processor.post_process_forward_op.assert_not_called()
    loop._pp_broadcast_output_tokens.assert_not_called()
    assert pending.result() is result and order == ["copy completed"]


def _rank_agreement(rank, store):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        validator = StateCommitValidator(
            enabled=True, local_groups=1, groups=(dist.group.WORLD,)
        )
        for failure in ("none", "false", "missing", "shape", "none"):
            flags = torch.ones(1, 2, dtype=torch.bool)
            if rank == 1:
                if failure == "false":
                    flags[0, 1] = False
                elif failure == "missing":
                    flags = None
                elif failure == "shape":
                    flags = torch.ones(1, 3, dtype=torch.bool)
            if failure == "none":
                validator.validate(flags, bs=2, requires_commit=True)
            else:
                # Rank zero has valid local data but must still reject.
                with pytest.raises(RuntimeError, match="model rank"):
                    validator.validate(flags, bs=2, requires_commit=True)
        validator = StateCommitValidator(
            enabled=True,
            local_groups=rank,
            groups=(dist.group.WORLD,),
        )
        validator.validate(
            None if rank == 0 else torch.ones(1, 2, dtype=torch.bool),
            bs=2,
            requires_commit=True,
        )
    finally:
        dist.destroy_process_group()


def test_state_commit_agreement_across_real_cpu_ranks(tmp_path):
    mp.spawn(_rank_agreement, args=(str(tmp_path / "gloo-store"),), nprocs=2, join=True)
