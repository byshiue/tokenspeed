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

import pytest

pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.engine.scheduler_utils import make_config
from tokenspeed.runtime.utils.env import (
    StateCheckpointPrefillMode,
    envs,
    state_checkpoint_prefill_mode,
)


def test_state_checkpoint_prefill_mode_defaults_to_single_forward() -> None:
    with envs.TOKENSPEED_STATE_CHECKPOINT_PREFILL_MODE.override("single_forward"):
        envs.TOKENSPEED_STATE_CHECKPOINT_PREFILL_MODE.clear()
        assert (
            state_checkpoint_prefill_mode() is StateCheckpointPrefillMode.SINGLE_FORWARD
        )


@pytest.mark.parametrize("mode", list(StateCheckpointPrefillMode))
def test_state_checkpoint_prefill_mode_maps_to_scheduler_enum(
    mode: StateCheckpointPrefillMode,
) -> None:
    cfg = make_config(
        num_device_pages=32,
        max_scheduled_tokens=128,
        max_batch_size=8,
        prefix_granularity=128,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role="fused",
        state_checkpoint_prefill_mode=mode,
    )
    expected = {
        StateCheckpointPrefillMode.SINGLE_FORWARD: cfg.StateCheckpointPrefillMode.SingleForward,
        StateCheckpointPrefillMode.SPLIT_TAIL: cfg.StateCheckpointPrefillMode.SplitTail,
    }
    assert cfg.state_checkpoint_prefill_mode == expected[mode]


@pytest.mark.parametrize("value", ["", "unknown"])
def test_state_checkpoint_prefill_mode_rejects_invalid_value(value: str) -> None:
    with envs.TOKENSPEED_STATE_CHECKPOINT_PREFILL_MODE.override(value):
        with pytest.raises(ValueError, match="single_forward, split_tail"):
            state_checkpoint_prefill_mode()


def test_split_tail_is_ineffective_without_snapshot_state_prefix_cache() -> None:
    from tokenspeed_scheduler import SchedulerConfig

    cfg = SchedulerConfig()
    cfg.state_checkpoint_prefill_mode = cfg.StateCheckpointPrefillMode.SplitTail
    assert (
        cfg.effective_state_checkpoint_prefill_mode
        == cfg.StateCheckpointPrefillMode.SingleForward
    )


@pytest.mark.parametrize("role", ["Fused", "P", "D"])
@pytest.mark.parametrize("prefix_cache_enabled", [False, True])
def test_effective_mode_comes_from_scheduler_config(role, prefix_cache_enabled) -> None:
    from tokenspeed_scheduler import (
        CacheGroupConfig,
        CacheGroupFamily,
        CacheRetention,
        SchedulerConfig,
    )

    cfg = SchedulerConfig()
    cfg.role = getattr(cfg.Role, role)
    cfg.state_checkpoint_prefill_mode = cfg.StateCheckpointPrefillMode.SplitTail
    cfg.disable_prefix_cache = not prefix_cache_enabled
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="state",
            rows_per_page=4,
            entry_stride_tokens=1,
            total_pages=32,
            retention=CacheRetention.FullHistory,
            family=CacheGroupFamily.State,
        )
    ]
    expected = (
        cfg.StateCheckpointPrefillMode.SplitTail
        if prefix_cache_enabled and role != "D"
        else cfg.StateCheckpointPrefillMode.SingleForward
    )
    assert cfg.effective_state_checkpoint_prefill_mode == expected
