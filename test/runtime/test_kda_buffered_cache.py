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

from tokenspeed.runtime.layers.attention.kda_replay import (
    KDAReplayLayout,
    kda_buffered_workspace_bytes,
)
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
        assert recipe.workspace_bytes() == kda_buffered_workspace_bytes(
            layers=69,
            max_bs=4,
            max_context_len=recipe.attn_config.context_len,
            max_window=width,
            heads=96 // tp,
            key_dim=128,
            value_dim=128,
            groups=3,
            state_grain=128,
            history_block_tokens=8,
        )
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


def test_incomplete_replay_layout_and_unsupported_dispatch_are_rejected():
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend

    recipe = _recipe(64)
    plan = _layout(recipe).bind(1)
    _, pool = _pool(recipe, plan, "cpu", 0, 93)
    backend = object.__new__(KdaAttnBackend)
    backend.cache_pool = None
    backend._state_group_ids = ()
    backend._checkpoint_granularity = None
    backend._state_layer_geometry = ()
    backend._buffered_replay = None
    backend.dtype = torch.bfloat16
    with pytest.raises(RuntimeError, match="BF16 CUDA"):
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
    materialized = torch.zeros_like(flush)
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
        for_handoff=False,
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
        for_handoff=False,
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
        (replay.checkpoint,),
        table,
        end,
        width,
        width,
        checkpoint,
        length,
        flush,
        ok,
        materialized,
        for_handoff=False,
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
            for_handoff=False,
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
            for_handoff=False,
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
            materialized,
            for_handoff=False,
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
    materialized = torch.zeros_like(metadata.ok)
    metadata.refresh(2, 0, seq_lens, tables, for_handoff=False)

    def run(bs, live):
        metadata.prepare(bs, for_handoff=False)
        # There are no numerical payloads in this metadata-only fixture.
        # Serving must put all layer stores between these two operations.
        metadata.commit(bs, accepted[:live], materialized, for_handoff=False)

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
        metadata.refresh(bs, actual_bs, seq_lens, current, for_handoff=False)
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
    metadata.commit(2, accepted[:2], materialized, for_handoff=False)
    for gid, stamps in metadata._stamps.items():
        assert not metadata._batch(2)[gid].ok.any()
        for stamp in stamps:
            torch.testing.assert_close(stamp, before[gid], atol=0, rtol=0)
    with pytest.raises(ValueError, match="missing"):
        metadata.refresh(2, 2, seq_lens, {}, for_handoff=False)
    gid = next(iter(groups))
    malformed = dict(tables)
    malformed[gid] = tables[gid][:, ::2]
    with pytest.raises(ValueError, match="invalid"):
        metadata.refresh(2, 2, seq_lens, malformed, for_handoff=False)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width,capacity", [(1, 8), (4, 8), (4, 37)])
@pytest.mark.parametrize("captured", [False, True])
def test_quiescent_endpoint_without_candidate_backing(width, capacity, captured):
    """Fresh handoff metadata, not a replay of the previous forward's commit.

    One row would capacity-flush on its next forward, another would not, and
    two rows are exact seeds (zero and nonzero). Candidate pages are missing.
    Reordering, duplicate handoff and failed backing cannot consume raw payload
    or overwrite any conv window or previously materialized source block.
    """
    from tokenspeed.runtime.layers.attention.backends.state.kda_buffered import (
        KDAReplayWorkspace,
    )

    torch.manual_seed(719)
    batch, context = 4, 256
    recipe = _recipe(
        capacity, decode_input_tokens=width, max_bs=batch, context_len=context
    )
    plan = _layout(recipe).bind(24).narrow_to_layers(28, 36)
    _, pool = _pool(recipe, plan, "cuda", 28, 36)
    workspace = KDAReplayWorkspace(pool, max_bs=batch, max_context_len=context)
    meta, layers = workspace.metadata, workspace.layer_ids
    ends = [152 if capacity == 37 else 136, 144, 0, 160]
    checkpoints = [
        ends[0] - (capacity - width),
        ends[1] - min(3, capacity - width),
        0,
        160,
    ]
    tables = {}
    for index, (gid, group) in enumerate(meta._groups.items()):
        first = index * 72 + 1
        ht = torch.zeros((batch, 20), dtype=torch.int32, device="cuda")
        ht[:2, 8:20].copy_(torch.arange(first, first + 24).view(2, 12))
        for row, end in enumerate(ends):
            ht[row, end // 8 :] = 0  # No space for even one future input token.
        st = torch.zeros((batch, 2), dtype=torch.int32, device="cuda")
        st[:2].copy_(torch.arange(index * 12 + 5, index * 12 + 9).view(2, 2))
        st[3].copy_(torch.arange(index * 12 + 9, index * 12 + 11))
        tables[gid], tables[group[0].checkpoint_group_id] = ht, st
        pool.zero_new_blocks({gid: list(range(first, first + 24))})
    tables_cpu = {gid: table.cpu() for gid, table in tables.items()}
    expected = {}
    for layer in layers:
        history = pool.get_replay_buffers(layer)
        conv, state = pool.get_state_buffers(layer)
        ht, st = tables_cpu[history.group_id], tables_cpu[history.checkpoint_group_id]
        for row in (0, 1, 3):
            c, e = checkpoints[row], ends[row]
            src, dst = int(st[row, (c - 1) // 128]), int(st[row, (e - 1) // 128])
            conv[st[row].long()] = torch.randn_like(conv[st[row].long()])
            state[dst].fill_(float("nan"))
            value = torch.randn(state[src].shape) * 0.02
            state[src].copy_(value)
            for position in range(c, e):
                page, slot = int(ht[row, position // 8]), position % 8
                k = torch.randn((workspace.heads, workspace.key_dim)) * 0.1
                u = torch.randn((workspace.heads, workspace.value_dim)) * 0.02
                d = torch.full_like(k, 0.97)
                for field, data in zip(
                    (history.key, history.correction, history.decay),
                    (k, u, d),
                    strict=True,
                ):
                    field[page, slot].copy_(data)
                value = value * d[:, None, :] + u[:, :, None] * k[:, None, :]
            if c != e:
                history.checkpoint[int(ht[row, (e - 1) // 8]), (e - 1) % 8] = c + 1
            expected[layer, row] = value
    saved = {
        layer: tuple(t.clone() for t in pool.get_state_buffers(layer))
        for layer in layers
    }
    workspace.payload.fill_(float("nan"))
    meta.conv_read.fill_(-1)
    meta.conv_writes.fill_(-1)
    # Decoy refresh represents an unrelated/idle prior batch. Handoff must
    # replace these endpoints, including the pending (not completed) flush.
    meta.end.fill_(2**31 - 1)
    meta.checkpoint.fill_(-1)
    meta.flushed.fill_(True)
    order = [1, 0, 3, 2]
    delivery = {gid: table[order] for gid, table in tables.items()}
    endpoints = torch.tensor(
        [ends[row] for row in order], dtype=torch.int32, device="cuda"
    )
    pool.layerwise_load_tracker = Mock()

    def run():
        return workspace.materialize_current(endpoints, delivery)

    if capacity == 37:
        # Lose a middle history block without losing the final stamp. The
        # validator must reject before any of this request's layers write.
        original = pool.arena.buffer.clone()
        for gid in meta._groups:
            delivery[gid][1, 15] = 0
        assert not run()[:, 1].any()
        for layer in layers:
            st = tables_cpu[pool.get_replay_buffers(layer).checkpoint_group_id]
            _, state = pool.get_state_buffers(layer)
            torch.testing.assert_close(
                state[st[0].long()],
                saved[layer][1][st[0].long()],
                atol=0,
                rtol=0,
                equal_nan=True,
            )
        pool.arena.buffer.copy_(original)
        for gid in delivery:
            delivery[gid].copy_(tables[gid][order])

    if captured:
        # Warm up kernels on a side stream, then restore the exact initial
        # representation before capture and again before the actual replay.
        original = pool.arena.buffer.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        pool.arena.buffer.copy_(original)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        pool.arena.buffer.copy_(original)
        graph.replay()
        validity = meta.ok[:, :batch]
    else:
        validity = run()
    assert validity.all() and not meta.flushed.any() and not meta.width.any()
    assert {
        call.args[0]
        for call in pool.layerwise_load_tracker.wait_for_layer.call_args_list
    } == set(layers)
    for layer in layers:
        history = pool.get_replay_buffers(layer)
        conv, state = pool.get_state_buffers(layer)
        ht, st = tables_cpu[history.group_id], tables_cpu[history.checkpoint_group_id]
        torch.testing.assert_close(
            conv, saved[layer][0], atol=0, rtol=0, equal_nan=True
        )
        for row in (0, 1, 3):
            c, e = checkpoints[row], ends[row]
            src, dst = int(st[row, (c - 1) // 128]), int(st[row, (e - 1) // 128])
            torch.testing.assert_close(
                state[dst].cpu(), expected[layer, row], atol=2e-5, rtol=2e-4
            )
            if src != dst:
                torch.testing.assert_close(
                    state[src], saved[layer][1][src], atol=0, rtol=0
                )
            if row != 3:
                assert (
                    int(history.checkpoint[int(ht[row, (e - 1) // 8]), (e - 1) % 8])
                    == e + 1
                )
    exact = pool.arena.buffer.clone()
    # The same endpoint now has zero history and must not be reconstructed a
    # second time. Reordering still uses current tables, not prior row slots.
    endpoints.copy_(torch.tensor(ends, dtype=torch.int32))
    for gid in delivery:
        delivery[gid].copy_(tables[gid])
    if captured:
        graph.replay()
    else:
        run()
    assert meta.ok.all() and not workspace.materialized.any()
    torch.testing.assert_close(pool.arena.buffer, exact, atol=0, rtol=0)

    # A missing checkpoint source/destination is an invariant failure, even
    # for an already-exact endpoint. Failed rows perform no pool stores.
    for group in meta._groups.values():
        delivery[group[0].checkpoint_group_id][0] = 0
    if captured:
        graph.replay()
    else:
        run()
    assert not meta.ok[:, 0].any() and meta.ok[:, 1:].all()
    torch.testing.assert_close(pool.arena.buffer, exact, atol=0, rtol=0)
    for live in (1, 0):
        validity = workspace.materialize_current(
            endpoints[:live], {gid: table[:live] for gid, table in tables.items()}
        )
        assert validity.shape == (len(meta._groups), live) and validity.all()
        torch.testing.assert_close(pool.arena.buffer, exact, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width,capacity", [(1, 8), (4, 8), (4, 37)])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("backend_dispatch", [False, True])
def test_buffered_workspace_forward_commit_and_budget(
    width, capacity, captured, backend_dispatch
):
    """Packed fields through the workspace and actual backend decode dispatch.

    Existing GPU producers fix BF16 conv/gate rounding. Independent CPU
    recurrence keeps M8's FP32 tolerance and BF16 output half-ULP allowance.
    Poisoning rejected raw candidates checks that commit only consumes acceptance.
    """
    from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
        fused_kda_verify_conv_update,
    )

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.graph_ptr_guard import (
        snapshot_graph_metadata,
        verify_graph_metadata,
    )
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
    from tokenspeed.runtime.layers.attention.backends.state.kda_buffered import (
        KDAReplayWorkspace,
    )
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

    torch.manual_seed(711)
    batch, requests, heads, dim, rank, context = 3, 2, 12, 128, 128, 256
    channels = 3 * heads * dim
    recipe = _recipe(
        capacity, decode_input_tokens=width, max_bs=batch, context_len=context
    )
    plan = _layout(recipe).bind(24)
    arena, full = _pool(recipe, plan, "cuda", 0, 93)
    full_workspace = KDAReplayWorkspace(full, max_bs=batch, max_context_len=context)
    assert full_workspace.nbytes == recipe.workspace_bytes()
    del full_workspace, full, arena
    # A nonzero PP window crossing a state-group boundary; pool IDs are local.
    arena, pool = _pool(recipe, plan.narrow_to_layers(28, 36), "cuda", 28, 36)
    backend = None
    if backend_dispatch:
        config = replace(
            recipe.attn_config, device="cuda", speculative_num_draft_tokens=width
        )
        backend = KdaAttnBackend(config, config.component(MLAConfig))
        backend.set_cache_pool(pool)
        backend.init_cuda_graph_state(batch)
        root_backend = HybridLinearAttnBackend(
            AttentionBackend(config, config.component(MLAConfig)), backend, []
        )
        workspace = backend._buffered_replay
        assert backend.preallocate_verify_workspace(batch, width) == workspace.nbytes
    else:
        workspace = KDAReplayWorkspace(pool, max_bs=batch, max_context_len=context)
    meta, layers = workspace.metadata, workspace.layer_ids
    assert len(layers) == 6 and len(meta._groups) == 2
    assert workspace.nbytes == kda_buffered_workspace_bytes(
        layers=len(layers),
        max_bs=batch,
        max_context_len=context,
        max_window=width,
        heads=heads,
        key_dim=dim,
        value_dim=dim,
        groups=2,
        state_grain=128,
        history_block_tokens=8,
    )
    tables = {}
    for index, (gid, group) in enumerate(meta._groups.items()):
        # Four history parents and four state parents per group. Field views
        # overlay physical planes: zero only the group's owned child pages.
        first = index * 8 * 6 + 1
        tables[gid] = torch.zeros((batch, 26), dtype=torch.int32, device="cuda")
        tables[gid][:requests, 14:26].copy_(
            torch.arange(first, first + 24).view(requests, 12)
        )
        tables[group[0].checkpoint_group_id] = torch.zeros(
            (batch, 2), dtype=torch.int32, device="cuda"
        )
        tables[group[0].checkpoint_group_id][:requests].copy_(
            torch.arange(index * 8 + 5, index * 8 + 9).view(requests, 2)
        )
        pool.zero_new_blocks({gid: list(range(first, first + 24))})
    tables_cpu = {gid: table.cpu() for gid, table in tables.items()}
    ends, checkpoints = [120, 124], [120, 124]
    states = torch.randn((len(layers), requests, heads, dim, dim)) * 0.02
    windows = (
        torch.randn((len(layers), requests, channels, 3), dtype=torch.bfloat16) * 0.1
    )
    for index, layer in enumerate(layers):
        history = pool.get_replay_buffers(layer)
        conv, state = pool.get_state_buffers(layer)
        for req in range(requests):
            page = int(tables_cpu[history.checkpoint_group_id][req, 0])
            state[page].copy_(states[index, req])
            conv[page].copy_(windows[index, req])
    raw = torch.empty(
        (len(layers), batch, width, channels), dtype=torch.bfloat16, device="cuda"
    )
    f_a = torch.empty(
        (len(layers), batch * width, rank), dtype=torch.bfloat16, device="cuda"
    )
    f_b = (
        torch.randn(
            (len(layers), heads * dim, rank), dtype=torch.bfloat16, device="cuda"
        )
        * 0.05
    )
    weights = (
        torch.randn((len(layers), channels, 4), dtype=torch.bfloat16, device="cuda")
        * 0.2
    )
    beta = torch.empty(
        (len(layers), batch, width, heads), dtype=torch.bfloat16, device="cuda"
    )
    a_log = torch.full((heads,), -1.0, device="cuda")
    bias = torch.randn((len(layers), heads * dim), device="cuda") * 0.2
    bias_cpu = bias.cpu().view(len(layers), 1, 1, heads, dim)
    outputs = torch.empty(
        (len(layers), batch, width, heads, dim), dtype=torch.bfloat16, device="cuda"
    )
    seq_lens = torch.zeros(batch, dtype=torch.int32, device="cuda")
    accepted = torch.zeros_like(seq_lens)
    force = torch.zeros_like(seq_lens, dtype=torch.bool)
    rejected = torch.zeros((1, batch, width, 1), dtype=torch.bool, device="cuda")

    def refresh(live, delivered):
        if backend is None:
            meta.refresh(batch, live, seq_lens, delivered, for_handoff=False)
        else:
            backend.refresh_decode_metadata(
                batch,
                live,
                seq_lens,
                seq_lens,
                forward_mode=ForwardMode.DECODE,
                block_tables=delivered,
                num_extends=0,
                for_graph_replay=captured,
            )

    if captured and backend is not None:
        backend.init_forward_metadata_capture_cuda_graph(
            batch, seq_lens, seq_lens, ForwardMode.DECODE, block_tables=tables
        )
    else:
        refresh(0, tables)

    def run():
        if backend is None:
            meta.prepare(batch, for_handoff=False)
        for index, layer in enumerate(layers):
            if backend is None:
                output = workspace.forward(
                    layer,
                    batch,
                    raw[index],
                    weights[index],
                    f_a[index],
                    f_b[index],
                    beta[index],
                    a_log,
                    bias[index],
                    -0.3,
                )
            else:
                output = root_backend.forward(
                    None,
                    None,
                    None,
                    None,
                    pool,
                    ForwardMode.DECODE,
                    batch,
                    save_kv_cache=True,
                    record_kv_cache=None,
                    layer_id=layer,
                    mixed_qkv=raw[index].view(batch * width, channels),
                    conv_weights=weights[index],
                    bias=None,
                    activation="silu",
                    f_a_out=f_a[index],
                    f_b_weight=f_b[index],
                    beta_raw=beta[index].view(batch * width, heads),
                    A_log=a_log,
                    dt_bias=bias[index],
                    lower_bound=-0.3,
                ).view(batch, width, heads, dim)
            outputs[index].copy_(output)  # Consume shared scratch before next layer.
        workspace.payload.masked_fill_(rejected, float("nan"))
        if backend is None:
            workspace.commit(
                batch,
                accepted if captured else accepted[:requests],
                force if captured else force[:requests],
                for_handoff=False,
            )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    if captured:
        with torch.cuda.graph(graph):
            run()
    snapshot = snapshot_graph_metadata(backend) if backend is not None else None
    buffers = (
        workspace.payload,
        workspace.conv_output,
        workspace.gate_output,
        workspace.output,
        meta.tables.tables,
    )
    pointers = [t.data_ptr() for t in buffers]
    saw_flush = saw_no_flush = saw_zero_accept_flush = saw_split_state_slots = False
    saw_endpoint = saw_zero_accept_endpoint = saw_aligned_endpoint = False
    published = {}
    for step in range(32):
        order = [step % 2, (step + 1) % 2]
        counts = [(step + row) % (width + 1) for row in range(requests)]
        flags = [ends[req] - checkpoints[req] + 2 * width > capacity for req in order]
        if not saw_zero_accept_flush:
            for row, flag in enumerate(flags):
                if flag:
                    counts[row] = 0
        forced = [step > 15 and step % 11 == 9, False]
        if step > 15 and not saw_zero_accept_endpoint:
            for row, req in enumerate(order):
                if ends[req] > checkpoints[req] and not flags[row]:
                    counts[row], forced[row] = 0, True
                    break
        if backend is not None:
            forced = [False] * requests  # Serving commit has no external handoff.
        endpoint_flags = [
            (forced[row] or (ends[req] + counts[row]) % 128 == 0)
            and ends[req] + counts[row]
            > (ends[req] if flags[row] else checkpoints[req])
            for row, req in enumerate(order)
        ]
        force.copy_(torch.tensor(forced + [True]))  # Padding cannot request a write.
        seq_lens.copy_(torch.tensor([ends[req] + width for req in order] + [2**31 - 1]))
        accepted.copy_(torch.tensor(counts + [0]))
        rejected.copy_(
            torch.arange(width)[None, None, :, None]
            >= torch.tensor(counts + [0])[None, :, None, None]
        )
        current = {gid: table[order + [requests]] for gid, table in tables.items()}
        refresh(requests, current)
        if backend is not None:
            verify_graph_metadata(backend, snapshot, context="buffered decode refresh")
        raw_cpu = torch.randn(raw.shape, dtype=torch.bfloat16) * 0.3
        raw.copy_(raw_cpu)
        f_a.normal_(std=0.1)
        beta.normal_()
        # Run the rounding oracle before commit changes the conv input windows.
        meta.prepare(batch, for_handoff=False)
        conv_cpu, gates_cpu = [], []
        for index, layer in enumerate(layers):
            view = meta.layer(layer, batch)
            conv_cpu.append(
                fused_kda_verify_conv_update(
                    raw[index].view(batch * width, channels),
                    weights[index],
                    pool.get_component(layer, "conv_state"),
                    view.conv_read,
                    num_heads=heads,
                    head_dim=dim,
                    draft_token_num=width,
                    out=None,
                    block_c=256,
                    num_warps=4,
                )
                .view(batch, width, 3, heads, dim)[:requests]
                .cpu()
                .float()
            )
            gates_cpu.append(
                torch.mm(f_a[index], f_b[index].t())
                .view(batch, width, heads, dim)[:requests]
                .cpu()
                .float()
            )
        query, key, value = torch.stack(conv_cpu).unbind(3)
        query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        decay = (
            -0.3
            * torch.sigmoid(
                a_log.cpu().exp()[:, None] * (torch.stack(gates_cpu) + bias_cpu)
            )
        ).exp()
        beta_cpu = beta[:, :requests].cpu().float().sigmoid()
        candidate, previous = states[:, order].clone(), states.clone()
        expected_outputs = []
        for token in range(width):
            candidate *= decay[:, :, token, :, None, :]
            correction = beta_cpu[:, :, token, :, None] * (
                value[:, :, token]
                - torch.einsum("lbhvk,lbhk->lbhv", candidate, key[:, :, token])
            )
            candidate += correction[..., None] * key[:, :, token, :, None, :]
            expected_outputs.append(
                torch.einsum("lbhvk,lbhk->lbhv", candidate, query[:, :, token])
                / dim**0.5
            )
            for row, req in enumerate(order):
                if counts[row] == token + 1:
                    states[:, req].copy_(candidate[:, row])
        if captured:
            graph.replay()
        else:
            run()
        if backend is not None:
            # Like ForwardStepRunner, acceptance commit follows the graph and
            # receives only live rows, not the capture ladder's padded output.
            assert root_backend.state_commit_validity(requests, num_extends=0) is None
            root_backend.commit_state_after_verify(accepted[:requests], num_extends=0)
            validity = root_backend.state_commit_validity(requests, num_extends=0)
            assert validity.shape == (2, requests) and validity.all()
        torch.testing.assert_close(
            outputs[:, :requests].cpu().float(),
            torch.stack(expected_outputs, dim=2),
            atol=2e-5,
            rtol=2e-4 + torch.finfo(torch.bfloat16).eps / 2,
        )
        assert meta.ok.all() and meta.flushed.tolist() == [flags + [False]] * 2
        assert workspace.materialized[:, :requests].tolist() == [endpoint_flags] * 2
        for (layer, block), (saved_conv, saved_state) in published.items():
            conv, state = pool.get_state_buffers(layer)
            torch.testing.assert_close(conv[block], saved_conv, atol=0, rtol=0)
            torch.testing.assert_close(state[block], saved_state, atol=0, rtol=0)
        for index, layer in enumerate(layers):
            history = pool.get_replay_buffers(layer)
            conv, state = pool.get_state_buffers(layer)
            for row, req in enumerate(order):
                e, c, count, flushed = (
                    ends[req],
                    checkpoints[req],
                    counts[row],
                    flags[row],
                )
                state_table = tables_cpu[history.checkpoint_group_id][req]
                saw_split_state_slots |= (c - 1) // 128 != (e - 1) // 128
                if flushed and (
                    not endpoint_flags[row] or (e - 1) // 128 != (e + count - 1) // 128
                ):
                    torch.testing.assert_close(
                        state[state_table[(e - 1) // 128]].cpu(),
                        previous[index, req],
                        atol=2e-5,
                        rtol=2e-4,
                    )
                if count:
                    windows[index, req] = torch.cat(
                        (windows[index, req], raw_cpu[index, row, :count].T), dim=-1
                    )[:, -3:]
                new_end, new_c = e + count, e if flushed else c
                if endpoint_flags[row]:
                    new_c = new_end
                    saw_endpoint = True
                    saw_zero_accept_endpoint |= count == 0
                    saw_aligned_endpoint |= new_end % 128 == 0
                torch.testing.assert_close(
                    conv[state_table[(new_end - 1) // 128]].cpu(),
                    windows[index, req],
                    atol=0,
                    rtol=0,
                )
                stamp_page = int(tables_cpu[history.group_id][req, (new_end - 1) // 8])
                stamp = int(history.checkpoint[stamp_page, (new_end - 1) % 8])
                assert (new_end if stamp == 0 else stamp - 1) == new_c
                reconstructed = state[state_table[(new_c - 1) // 128]].cpu().clone()
                positions = torch.arange(new_c, new_end, device="cuda")
                pages = tables[history.group_id][req, positions // 8]
                cached = [
                    field[pages, positions % 8].cpu()
                    for field in (history.key, history.correction, history.decay)
                ]
                for k, u, d in zip(*cached, strict=True):
                    reconstructed = (
                        reconstructed * d[:, None, :] + u[:, :, None] * k[:, None, :]
                    )
                torch.testing.assert_close(
                    reconstructed, states[index, req], atol=2e-5, rtol=2e-4
                )
                if new_end % 128 == 0:
                    block = int(state_table[(new_end - 1) // 128])
                    published[layer, block] = (
                        conv[block].clone(),
                        state[block].clone(),
                    )
        for row, req in enumerate(order):
            if flags[row]:
                checkpoints[req] = ends[req]
                saw_flush = True
                saw_zero_accept_flush |= counts[row] == 0
            else:
                saw_no_flush = True
            ends[req] += counts[row]
            if endpoint_flags[row]:
                checkpoints[req] = ends[req]
        assert [
            t.data_ptr()
            for t in (
                workspace.payload,
                workspace.conv_output,
                workspace.gate_output,
                workspace.output,
                meta.tables.tables,
            )
        ] == pointers
    assert saw_flush and saw_no_flush and saw_zero_accept_flush
    assert saw_endpoint and saw_aligned_endpoint
    if capacity > 2 * width and backend is None:
        assert saw_zero_accept_endpoint
    if capacity == 37:
        assert saw_split_state_slots
    # Acceptance is validated before conv, endpoint or stamp commits. Rejecting
    # the count must not partially materialize any layer, even with handoff set.
    before = [
        tuple(
            t.clone()
            for t in (
                *pool.get_state_buffers(layer),
                pool.get_replay_buffers(layer).checkpoint,
            )
        )
        for layer in layers
    ]
    accepted.fill_(width + 1)
    force.fill_(True)
    if backend is None:
        workspace.commit(batch, accepted, force, for_handoff=False)
        assert not meta.ok.any()
    else:
        # Re-enter through refresh: duplicate commit must never reconstruct an
        # endpoint on top of the state the previous commit just materialized.
        with pytest.raises(RuntimeError, match="already committed"):
            root_backend.commit_state_after_verify(accepted[:requests], num_extends=0)
        seq_lens.copy_(torch.tensor([ends[req] + width for req in order] + [2**31 - 1]))
        refresh(requests, current)
        assert meta.ok[:, :requests].all()
        root_backend.commit_state_after_verify(accepted[:requests], num_extends=0)
        assert not root_backend.state_commit_validity(requests, num_extends=0).any()
    for layer, saved in zip(layers, before, strict=True):
        for current, old in zip(
            (*pool.get_state_buffers(layer), pool.get_replay_buffers(layer).checkpoint),
            saved,
            strict=True,
        ):
            torch.testing.assert_close(current, old, atol=0, rtol=0, equal_nan=True)
    if backend is not None:
        _, replacement = _pool(recipe, plan.narrow_to_layers(28, 36), "cuda", 28, 36)
        backend.set_cache_pool(replacement)
        rebound = backend._buffered_replay
        assert rebound is not workspace and backend.forward_metadata is None
        assert backend._buffered_actual_bs == 0
        assert rebound.payload.data_ptr() != workspace.payload.data_ptr()
        assert backend._verify_scratch is None and backend._replay_payloads is None
        backend.init_cuda_graph_state(batch)
        assert backend.preallocate_verify_workspace(batch, width) == rebound.nbytes
        changed = _recipe(
            capacity + 8, decode_input_tokens=width, max_bs=batch, context_len=context
        )
        changed_plan = _layout(changed).bind(24).narrow_to_layers(28, 36)
        _, wrong = _pool(changed, changed_plan, "cuda", 28, 36)
        with pytest.raises(RuntimeError, match="different buffered replay layout"):
            backend.set_cache_pool(wrong)
        assert backend.cache_pool is replacement and backend._buffered_replay is rebound
