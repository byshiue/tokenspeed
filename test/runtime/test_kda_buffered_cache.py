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
from unittest.mock import Mock, patch

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
def test_gpu_arena_zeroing_and_recurrence_use_distinct_parents():
    from tokenspeed_kernel.ops.attention.kda._triton.buffered import (
        buffered_recurrent,
        validate_recurrent_blocks,
    )
    from tokenspeed_kernel.ops.attention.kda._triton.buffered_metadata import (
        commit_positions,
        prepare_positions,
    )

    recipe = _recipe(8)
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
    state = pool.get_component(layer, "recurrent_state")
    # Different cache groups cannot occupy the same physical LCM parent.
    # History blocks 1..6 use parent 1; state block 2 uses parent 2.
    state_table = torch.tensor([[2]], dtype=torch.int32, device="cuda")
    state[2].fill_(0.25)
    heads, value_dim, key_dim = state.shape[1:]
    q = torch.full((1, 1, heads, key_dim), key_dim**-0.5, device="cuda")
    v = torch.ones((1, 1, heads, value_dim), device="cuda")
    decay = torch.full_like(q, 0.9)
    beta = torch.full((1, 1, heads), 0.5, device="cuda")
    out = torch.empty_like(v)
    validate_recurrent_blocks(
        table,
        state_table,
        end,
        checkpoint,
        length,
        width,
        flush,
        ok,
        history_blocks=replay.key.shape[0],
        state_blocks=state.shape[0],
        history_block_tokens=replay.layout.block_tokens,
        state_block_tokens=128,
        capacity=replay.layout.capacity,
        max_window=replay.layout.max_window,
    )
    buffered_recurrent(
        q,
        q,
        v,
        decay,
        beta,
        state,
        replay.key,
        replay.correction,
        replay.decay,
        table,
        state_table,
        end,
        checkpoint,
        length,
        width,
        flush,
        ok,
        out,
        capacity=replay.layout.capacity,
        state_block_tokens=128,
        transform_inputs=False,
        A_log=None,
        dt_bias=None,
        lower_bound=None,
    )
    expected_state = state[2].clone() * 0.9
    correction = 0.5 * (v[0, 0] - torch.einsum("hvk,hk->hv", expected_state, q[0, 0]))
    expected_state += correction[:, :, None] * q[0, 0, :, None, :]
    torch.testing.assert_close(
        out[0, 0],
        torch.einsum("hvk,hk->hv", expected_state, q[0, 0]) * key_dim**-0.5,
        atol=2e-5,
        rtol=2e-4,
    )
    assert torch.all(state[2] == 0.25)  # No full-state store without a flush.
    torch.testing.assert_close(
        replay.correction[1, 3], correction, atol=2e-5, rtol=2e-4
    )
    commit_positions(
        (replay.checkpoint,), table, end, width, width, checkpoint, length, flush, ok
    )
    assert replay.checkpoint[1, 3] == 4 and ok.item()
    # Continue through a history-block boundary and a capacity flush using
    # the actual packed field strides, not separate dense allocations.
    pool.zero_new_blocks({replay.group_id: [2]})
    table = torch.tensor([[1, 2]], dtype=torch.int32, device="cuda")
    end.add_(1)
    saw_flush = False
    for _ in range(9):
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
        validate_recurrent_blocks(
            table,
            state_table,
            end,
            checkpoint,
            length,
            width,
            flush,
            ok,
            history_blocks=replay.key.shape[0],
            state_blocks=state.shape[0],
            history_block_tokens=replay.layout.block_tokens,
            state_block_tokens=128,
            capacity=replay.layout.capacity,
            max_window=replay.layout.max_window,
        )
        buffered_recurrent(
            q,
            q,
            v,
            decay,
            beta,
            state,
            replay.key,
            replay.correction,
            replay.decay,
            table,
            state_table,
            end,
            checkpoint,
            length,
            width,
            flush,
            ok,
            out,
            capacity=replay.layout.capacity,
            state_block_tokens=128,
            transform_inputs=False,
            A_log=None,
            dt_bias=None,
            lower_bound=None,
        )
        assert ok.item()
        if flush.item():
            saw_flush = True
            torch.testing.assert_close(state[2], expected_state, atol=2e-5, rtol=2e-4)
        expected_state *= 0.9
        correction = 0.5 * (
            v[0, 0] - torch.einsum("hvk,hk->hv", expected_state, q[0, 0])
        )
        expected_state += correction[:, :, None] * q[0, 0, :, None, :]
        torch.testing.assert_close(
            out[0, 0],
            torch.einsum("hvk,hk->hv", expected_state, q[0, 0]) * key_dim**-0.5,
            atol=2e-5,
            rtol=2e-4,
        )
        commit_positions(
            (replay.checkpoint,),
            table,
            end,
            width,
            width,
            checkpoint,
            length,
            flush,
            ok,
        )
        end.add_(1)
    assert saw_flush


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("captured", [False, True])
def test_shared_replay_metadata_refresh_commit_and_rebind(width, captured):
    """Exercise the real group fields; this test does not execute model/conv work."""
    from tokenspeed.runtime.layers.attention.backends.state import kda_buffered

    recipe = _recipe(2 * width, decode_input_tokens=width, max_bs=5)
    plan = _layout(recipe).bind(12)
    arena, pool = _pool(recipe, plan, "cuda", 0, 93)
    metadata = kda_buffered.KDAReplayMetadata(pool, max_bs=5, max_context_len=32)
    layers = tuple(pool.state_group_by_layer)
    groups = metadata._groups
    assert len(groups) == 3 and all(len(group) == 23 for group in groups.values())
    assert metadata.end.shape == (5,)  # Capture only B=2, not the runtime limit.
    tables = {}
    # A history parent gives six child pages (two per request). The following
    # three parents hold distinct request states; no groups alias LCM parents.
    for index, (gid, group) in enumerate(groups.items()):
        first = group[0]
        parent = index * 4 + 1
        history_start = (parent - 1) * 6 + 1
        tables[gid] = torch.tensor(
            [
                [history_start + 2 * req, history_start + 2 * req + 1]
                for req in range(3)
            ],
            dtype=torch.int32,
            device="cuda",
        )
        tables[first.checkpoint_group_id] = torch.tensor(
            [[parent + req + 1] for req in range(3)], dtype=torch.int32, device="cuda"
        )
        pool.zero_new_blocks({gid: list(range(history_start, history_start + 6))})
    seq_lens = torch.zeros(5, dtype=torch.int32, device="cuda")
    accepted = torch.zeros_like(seq_lens)
    metadata.refresh(2, 0, seq_lens, tables)

    def run(bs, live):
        metadata.prepare(bs)
        # There are no numerical payloads in this metadata-only fixture.
        # Serving must put all layer stores between these two operations.
        metadata.commit(bs, accepted[:live])

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(2, 2)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    if captured:
        with torch.cuda.graph(graph):
            run(2, 2)
    first_view = metadata.layer(layers[0], 2)
    pointers = [t.data_ptr() for t in vars(first_view).values()]
    for actual_bs in (2, 1, 0, 3, 2):
        bs = 5 if actual_bs == 3 else 2
        order = [2, 0, 1][:actual_bs]
        current = {gid: table[order] for gid, table in tables.items()}
        # Each iteration is a fresh exact-state/history fixture, independent of
        # the previous one. Committed history lengths differ across groups.
        ends = [7, 8, 9]
        expected = {}
        # Group views overlay the same arena planes. Zero only each group's
        # owned child pages, as the allocator does, never its whole field view.
        for index, (gid, group) in enumerate(groups.items()):
            parent = index * 4 + 1
            history_start = (parent - 1) * 6 + 1
            pool.zero_new_blocks({gid: list(range(history_start, history_start + 6))})
            checkpoints = [
                e - (width if (req + index) % 2 else 0) for req, e in enumerate(ends)
            ]
            expected[gid] = [checkpoints[req] for req in order]
            for req, e in enumerate(ends):
                page = tables[gid][req, (e - 1) // 8].item()
                for stamp in metadata._stamps[gid]:
                    stamp[page, (e - 1) % 8] = checkpoints[req] + 1
        seq_lens.fill_(2**31 - 1)  # Poison padding; it must not be read.
        if actual_bs:
            seq_lens[:actual_bs].copy_(torch.tensor([ends[r] + width for r in order]))
        accepted.zero_()
        accepted[:actual_bs].fill_(1)
        metadata.refresh(bs, actual_bs, seq_lens, current)
        if captured and bs == 2:
            graph.replay()
        else:
            with patch.object(
                kda_buffered, "prepare_positions", wraps=kda_buffered.prepare_positions
            ) as prepare:
                run(bs, actual_bs)
                assert prepare.call_count == 3
        for gid, group in groups.items():
            layer_ids = [
                layer for layer in layers if metadata._group_by_layer[layer] == gid
            ]
            view = metadata.layer(layer_ids[0], bs)
            assert all(metadata.layer(layer, bs) is view for layer in layer_ids)
            assert view.end.tolist() == [ends[r] for r in order] + [0] * (
                bs - actual_bs
            )
            assert view.width.tolist() == [width] * actual_bs + [0] * (bs - actual_bs)
            assert view.checkpoint.tolist() == expected[gid] + [0] * (bs - actual_bs)
            assert view.ok.all()
            assert not view.history_table[actual_bs:].count_nonzero()
            assert not view.history_table[:, 2:].count_nonzero()
            for row, req in enumerate(order):
                e, c = ends[req], expected[gid][row]
                page = current[gid][row, e // 8].item()
                stamp_value = e + 1 if e - c == width else c + 1
                for stamp in metadata._stamps[gid]:
                    assert stamp[page, e % 8] == stamp_value
        assert metadata.layer(layers[0], 2) is first_view
        assert [t.data_ptr() for t in vars(first_view).values()] == pointers
    # Live acceptance may be shorter than padded B; no copy/padding allocation
    # is needed by the group commit. Invalid live counts suppress their stamps.
    accepted[:2].fill_(width + 1)
    before = {gid: stamps[0].clone() for gid, stamps in metadata._stamps.items()}
    metadata.commit(2, accepted[:2])
    for gid, stamps in metadata._stamps.items():
        assert not metadata._batch(2)[gid].ok.any()
        for stamp in stamps:
            torch.testing.assert_close(stamp, before[gid], atol=0, rtol=0)
    with pytest.raises(ValueError, match="missing"):
        metadata.refresh(2, 2, seq_lens, {})
    gid = next(iter(groups))
    malformed = dict(tables)
    malformed[gid] = tables[gid][:, ::2]
    with pytest.raises(ValueError, match="invalid"):
        metadata.refresh(2, 2, seq_lens, malformed)
    with pytest.raises(ValueError, match="capacity"):
        metadata.layer(layers[0], 6)
    # Rebinding builds a new owner; old graph addresses and old cache fields
    # cannot leak into it. The ordinary pool is still rejected, never a fallback.
    _, replacement = _pool(recipe, plan, "cuda", 0, 93)
    rebound = kda_buffered.KDAReplayMetadata(replacement, max_bs=5, max_context_len=32)
    assert (
        rebound.layer(layers[0], 2).checkpoint.data_ptr()
        != first_view.checkpoint.data_ptr()
    )
    assert (
        rebound._stamps[next(iter(groups))][0].data_ptr()
        != metadata._stamps[next(iter(groups))][0].data_ptr()
    )
    assert rebound.tables.tables.data_ptr() != metadata.tables.tables.data_ptr()
    narrowed = plan.narrow_to_layers(4, 12)
    _, pp = _pool(recipe, narrowed, "cuda", 4, 12)
    pp_metadata = kda_buffered.KDAReplayMetadata(pp, max_bs=5, max_context_len=32)
    assert sum(len(stamps) for stamps in pp_metadata._stamps.values()) == len(
        pp.state_group_by_layer
    )
    for layer in pp.state_group_by_layer:
        replay = pp.get_replay_buffers(layer)
        assert pp_metadata._group_by_layer[layer] == replay.group_id
        assert any(
            stamp is replay.checkpoint for stamp in pp_metadata._stamps[replay.group_id]
        )
