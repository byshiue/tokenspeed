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


"""Kimi replay geometry, zero-copy binding, and paged position lifecycle gates."""

from __future__ import annotations

from dataclasses import replace
from test.runtime.cache_pool_test_utils import make_pool
from test.runtime.conftest import TP8_PAGE_SET_BYTES, kimi_recipe
from unittest.mock import Mock

import pytest
import torch

from tokenspeed.runtime.layers.attention.kda_replay import KDAReplayLayout
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
    HybridKDATokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.kimi_k3 import KimiK3Recipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    compute_cache_group_page_counts,
)


def _recipe(capacity, **kwargs):
    base = kimi_recipe(**kwargs)
    return KimiK3Recipe(
        replay_buffer_capacity=capacity,
        **{
            name: getattr(base, name)
            for name in (
                "server_args",
                "model_config",
                "attn_config",
                "draft_model_config",
                "draft_attn_config",
                "cache_budget_bytes",
                "decode_input_tokens",
                "overlap_schedule_depth",
            )
        },
    )


def _layout(recipe):
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    recipe.check_layout(layout)
    return layout


def _pool(recipe, plan, device, first, last):
    return make_pool(
        HybridKDATokenToKVPool,
        plan,
        device=device,
        model_dtype=torch.bfloat16,
        dtype=torch.float8_e4m3fn,
        quant_method=None,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=last - first,
        rank=0,
        field_layer_offset=first,
        layer_types=recipe.layer_types[first:last],
        cache_group_specs=tuple(spec for spec, _ in recipe.groups()),
    )


@pytest.mark.parametrize(
    "tp,mla_packing", [(1, 96), (2, 48), (4, 24), (8, 12), (16, 6)]
)
def test_replay_geometry_and_capacity_budget(tp, mla_packing):
    for capacity, width in ((8, 1), (37, 4), (64, 4)):
        recipe = _recipe(capacity, tp_size=tp, decode_input_tokens=width, max_bs=4)
        layout = _layout(recipe)
        plan = layout.bind(2)
        groups = recipe.groups()
        assert len(groups) == 7
        assert len(plan.planes) == 24
        assert plan.lcm_block_bytes == 24 * mla_packing * 128 * 576
        replay_groups = [
            (spec, fields) for spec, fields in groups if spec.replay_checkpoint_group
        ]
        assert len(replay_groups) == 3
        specs = {spec.group_id: spec for spec, _ in groups}
        for spec, fields in replay_groups:
            assert spec.rows_per_page == 8 and spec.sliding_window_tokens == capacity
            assert (
                specs[spec.replay_checkpoint_group].max_state_lag_tokens
                == capacity - width
            )
            assert recipe.packing(groups)[spec.group_id] == 6
            assert spec.transfer_policy is None
            stamps = [
                field
                for field in fields
                if field.field_id.endswith(".replay_checkpoint")
            ]
            assert len(stamps) == 23
            assert {field.plane_id for field in stamps} == {"slot.23"}
            assert all(
                field.dtype == "int64" and field.shape == (8,) for field in stamps
            )
            floats = [field for field in fields if field not in stamps]
            assert len(floats) == 69
            assert all(
                field.dtype == "float32" and field.shape == (8, 96 // tp, 128)
                for field in floats
            )
        assert all(
            field.page_stride_bytes == 128 * 576
            for field in plan.fields
            if field.field_id.endswith(".latent_kv")
        )
        # Replay residency is bounded by live decode, not chunked-prefill size.
        parents = recipe.parents_needed(layout, 4096)
        recipe.server_args.chunked_prefill_size = 8192
        assert recipe.parents_needed(layout, 4096) == parents
        counts = compute_cache_group_page_counts(
            tuple(specs.values()),
            max_total_tokens=4096,
            **recipe.scheduler_limits,
        )
        assert all(counts[spec.group_id] > 1 for spec, _ in replay_groups)
        setup = recipe.setup()
        assert (
            recipe.parents_needed(layout, setup.spec.token_capacity)
            <= setup.spec.memory_plan.num_lcm_blocks
        )
        assert recipe.workspace_bytes() == 0
        if tp == 8:
            assert plan.lcm_block_bytes == TP8_PAGE_SET_BYTES


def test_planning_input_and_serving_factory_are_explicit():
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import _RECIPES

    assert _RECIPES["kimi_k3"].keywords == {"replay_buffer_capacity": None}
    for capacity, width, block in (
        (True, 1, 16),
        (0, 1, 16),
        (7, 4, 16),
        (8, 0, 16),
        (8, 1, 0),
        (2**31, 1, 16),
    ):
        with pytest.raises(ValueError):
            KDAReplayLayout(capacity=capacity, max_window=width, block_tokens=block)
    with pytest.raises(ValueError, match="handoff"):
        _recipe(64, pd_enabled=True)


def test_replay_views_share_arena_respect_layer_windows_and_fences():
    recipe = _recipe(37, decode_input_tokens=4, draft_layers=5)
    plan = _layout(recipe).bind(1)
    arena, pool = _pool(recipe, plan, "cpu", 0, 93)
    assert pool.paged_group_ids == ("full_attention",)
    assert len(pool.history_group_by_layer()) == 24
    assert len(pool._replay_buffers_by_layer) == 69
    layer = min(pool.state_group_by_layer)
    replay = pool.get_replay_buffers(layer)
    assert replay.layout == KDAReplayLayout(capacity=37, max_window=4, block_tokens=8)
    for suffix, tensor in (
        ("key", replay.key),
        ("correction", replay.correction),
        ("decay", replay.decay),
        ("checkpoint", replay.checkpoint),
    ):
        assert (
            tensor.data_ptr()
            == arena.field(f"layer.{layer}.replay_{suffix}").data_ptr()
        )
        assert tensor.untyped_storage().data_ptr() == arena.buffer.data_ptr()
    replay.checkpoint[1, 3] = 42
    assert arena.field(f"layer.{layer}.replay_checkpoint")[1, 3] == 42
    pool.layerwise_load_tracker = Mock()
    assert pool.get_replay_buffers(layer) is replay
    pool.layerwise_load_tracker.wait_for_layer.assert_called_once_with(layer)
    with pytest.raises(ValueError, match="no buffered"):
        pool.get_replay_buffers(next(iter(pool.history_group_by_layer())))

    # A PP view keeps global field ownership but exposes view-local layer ids.
    narrowed = plan.narrow_to_layers(4, 12)
    _, pp = _pool(recipe, narrowed, "cpu", 4, 12)
    assert len(pp._replay_buffers_by_layer) == sum(
        field.field_id.endswith(".replay_key") for field in narrowed.fields
    )
    local = min(pp.state_group_by_layer)
    assert (
        pp.get_replay_buffers(local).key.data_ptr()
        == pp.arena.field(f"layer.{local+4}.replay_key").data_ptr()
    )
    _, draft = _pool(recipe, plan, "cpu", 93, 98)
    assert not draft._replay_buffers_by_layer and not draft.state_group_by_layer
    _, replacement = _pool(recipe, plan, "cpu", 0, 93)
    assert replacement.get_replay_buffers(layer).key.data_ptr() != replay.key.data_ptr()


def test_incomplete_replay_layout_and_serving_dispatch_are_rejected():
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend

    recipe = _recipe(64)
    plan = _layout(recipe).bind(1)
    _, pool = _pool(recipe, plan, "cpu", 0, 93)
    backend = object.__new__(KdaAttnBackend)
    backend.cache_pool = None
    backend._state_group_ids = ()
    backend._checkpoint_granularity = None
    backend._state_layer_geometry = ()
    with pytest.raises(RuntimeError, match="planning-only"):
        backend.validate_cache_pool(pool)
    layer = min(pool.state_group_by_layer)
    for suffix in ("key", "checkpoint"):
        incomplete = replace(
            plan,
            fields=tuple(
                field
                for field in plan.fields
                if field.field_id != f"layer.{layer}.replay_{suffix}"
            ),
        )
        with pytest.raises(ValueError, match="all four"):
            _pool(recipe, incomplete, "cpu", 0, 93)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_arena_zeroing_seeds_only_the_reused_history_page():
    from tokenspeed_kernel.ops.attention.kda._triton.buffered_metadata import (
        commit_positions,
        prepare_positions,
    )

    recipe = _recipe(64)
    arena, pool = _pool(recipe, _layout(recipe).bind(2), "cuda", 0, 93)
    layer = min(pool.state_group_by_layer)
    replay = pool.get_replay_buffers(layer)
    replay.key.fill_(3)
    replay.checkpoint.fill_(99)
    pool.zero_new_blocks({replay.group_id: [1]})
    torch.cuda.synchronize()
    assert torch.count_nonzero(replay.key[1]) == 0
    assert torch.count_nonzero(replay.checkpoint[1]) == 0
    assert torch.all(replay.key[2] == 3)
    assert torch.all(replay.checkpoint[2] == 99)
    assert (
        replay.checkpoint.data_ptr()
        == arena.field(f"layer.{layer}.replay_checkpoint").data_ptr()
    )
    table = torch.tensor([[1]], dtype=torch.int32, device="cuda")
    end = torch.tensor([3], dtype=torch.int32, device="cuda")
    width = torch.ones_like(end)
    checkpoint = torch.empty(1, dtype=torch.int64, device="cuda")
    length = torch.empty_like(end)
    flush = torch.empty(1, dtype=torch.bool, device="cuda")
    ok = torch.empty_like(flush)
    prepare_positions(
        replay.checkpoint,
        table,
        end,
        width,
        checkpoint,
        length,
        flush,
        ok,
        capacity=replay.layout.capacity,
        max_window=replay.layout.max_window,
    )
    assert checkpoint.item() == 3 and length.item() == 0 and ok.item()
    commit_positions(
        replay.checkpoint, table, end, width, width, checkpoint, length, flush, ok
    )
    assert replay.checkpoint[1, 3] == 4 and ok.item()
