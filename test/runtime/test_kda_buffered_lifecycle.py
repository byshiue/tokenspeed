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


"""Scheduler-owned prefix resume and slot recycling with real KDA GPU dispatch."""

from dataclasses import replace
from test.runtime.test_kda_buffered_cache import _layout, _pool, _recipe

import pytest
import tokenspeed_scheduler as ts
import torch

from tokenspeed.runtime.engine.scheduler_utils import pool_to_cache_groups
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
    CacheBatchMetadata,
)
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig


@pytest.mark.parametrize(
    "width,capacity,prompt,acceptance,rounds,abort,expected_hit",
    [
        (1, 8, 124, 1, 4, False, 128),
        (4, 8, 124, 4, 1, False, 128),
        (4, 37, 124, 3, 2, False, 0),
        (4, 8, 140, 4, 6, False, 128),
        (4, 37, 140, 4, 6, True, 128),
    ],
)
def test_scheduler_prefix_resume_uses_exact_state_and_fresh_history(
    width, capacity, prompt, acceptance, rounds, abort, expected_hit
):
    """Exercise real allocation/zero/forward/commit/feedback, not hand-made IDs.

    The next agentic turn is a new request. Only published exact checkpoints
    may seed it; lagging live state and history are not exported. This covers
    finish/cancel, aligned/skipped endpoints, capacity flush and slot reuse.
    Numerical recurrence and graph equivalence are covered by the kernel and
    workspace suites; this test checks their actual cache-owner handoff.
    """
    torch.manual_seed(751)
    recipe = _recipe(capacity, decode_input_tokens=width, max_bs=1, context_len=512)
    memory_plan = _layout(recipe).bind(32).narrow_to_layers(28, 36)
    _, pool = _pool(recipe, memory_plan, "cuda", 28, 36)
    config = replace(
        recipe.attn_config, device="cuda", speculative_num_draft_tokens=width
    )
    backend = KdaAttnBackend(
        config, config.component(MLAConfig), kda_backend="cutedsl_kda"
    )
    backend.set_cache_pool(pool)
    backend.init_cuda_graph_state(1)
    root = HybridLinearAttnBackend(
        AttentionBackend(config, config.component(MLAConfig)), backend, []
    )
    workspace = backend._buffered_replay
    layers = workspace.layer_ids
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 128
    cfg.num_device_pages = memory_plan.num_lcm_blocks + 1
    cfg.num_host_pages = 0
    cfg.max_scheduled_tokens = 256
    cfg.max_batch_size = 1
    cfg.decode_input_tokens = width
    cfg.overlap_schedule_depth = 0
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = False
    cfg.cache_groups = pool_to_cache_groups(pool)
    scheduler = ts.Scheduler(cfg)

    heads, dim, channels, rank = 12, 128, 4608, 128
    raw = (
        torch.randn(len(layers), 512, channels, dtype=torch.bfloat16, device="cuda")
        * 0.1
    )
    f_a = torch.randn(len(layers), 512, rank, dtype=torch.bfloat16, device="cuda") * 0.1
    f_b = (
        torch.randn(len(layers), heads * dim, rank, dtype=torch.bfloat16, device="cuda")
        * 0.05
    )
    beta = (
        torch.randn(len(layers), 512, heads, dtype=torch.bfloat16, device="cuda") * 0.1
    )
    weights = (
        torch.randn(len(layers), channels, 4, dtype=torch.bfloat16, device="cuda") * 0.2
    )
    a_log = torch.full((heads,), -1.0, device="cuda")
    bias = torch.randn(len(layers), heads * dim, device="cuda") * 0.2
    snapshots = {}
    history = list(range(prompt))
    first_slot = None
    saw_history = False
    saw_flush = False

    def feedback(request_id, tokens, decode):
        result = ts.ForwardEvent.ExtendResult()
        result.request_id, result.tokens = request_id, tokens
        event = ts.ExecutionEvent()
        event.add_event(result)
        if decode:
            reserve = ts.ForwardEvent.UpdateReserveNumTokens()
            reserve.request_id = request_id
            reserve.reserve_num_tokens_in_next_schedule_event = len(tokens)
            event.add_event(reserve)
        scheduler.advance(event)

    def execute(request_id, accepted, expected_prefix):
        nonlocal first_slot, saw_history, saw_flush
        plan = scheduler.next_execution_plan()
        assert len(plan.forward) == 1
        op = plan.forward[0]
        assert list(op.request_ids) == [request_id]
        pool.zero_new_blocks(dict(plan.pages_to_zero))
        bridge = CacheBatchMetadata.from_forward_op(
            op, device="cuda", contract=pool.arena.runtime_contract, num_requests=1
        )
        tables = dict(bridge.tables(active_forward_op=op))
        slots = torch.tensor(
            list(op.request_pool_indices), dtype=torch.int32, device="cuda"
        )
        if first_slot is None:
            first_slot = int(slots[0])
        else:
            assert int(slots[0]) == first_slot
        extends = op.num_extends()
        count = int(op.input_lengths[0])
        start = int(op.extend_prefix_lens[0]) if extends else len(history) - 1
        end = start + count
        incomplete = bool(extends and end < len(history))
        if incomplete:
            accepted = 0  # A promotion-boundary chunk produces no sampled token.
        lens = torch.tensor([end], dtype=torch.int32, device="cuda")
        if extends:
            assert start == expected_prefix
            lengths_cpu = torch.tensor([count], dtype=torch.int32)
            prefixes_cpu = torch.tensor([start], dtype=torch.int32)
            backend.init_forward_metadata(
                1,
                1,
                slots,
                lens,
                ForwardMode.EXTEND,
                block_tables=tables,
                extend_seq_lens=lengths_cpu.cuda(),
                extend_seq_lens_cpu=lengths_cpu,
                extend_prefix_lens=prefixes_cpu.cuda(),
                extend_prefix_lens_cpu=prefixes_cpu,
                extend_with_prefix=start > 0,
            )
            # Fresh candidate pages carry zero stamps even if their physical
            # parents belonged to the finished/cancelled request.
            for layer in layers:
                replay = pool.get_replay_buffers(layer)
                for page in dict(plan.pages_to_zero).get(replay.group_id, []):
                    assert torch.count_nonzero(replay.checkpoint[page]) == 0
                if start:
                    page = int(
                        backend.forward_metadata.state_in_blocks_by_group[
                            replay.checkpoint_group_id
                        ][0]
                    )
                    old_page, old_conv, old_state = snapshots[layer]
                    assert page == old_page
                    conv, state = pool.get_state_buffers(layer)
                    torch.testing.assert_close(conv[page], old_conv, rtol=0, atol=0)
                    torch.testing.assert_close(state[page], old_state, rtol=0, atol=0)
        else:
            backend.refresh_decode_metadata(
                1,
                1,
                slots,
                lens,
                forward_mode=ForwardMode.DECODE,
                block_tables=tables,
                num_extends=0,
                for_graph_replay=False,
            )
            saw_history |= bool((workspace.metadata.length > 0).any())
            saw_flush |= bool(workspace.metadata.flushed.any())
        mode = ForwardMode.EXTEND if extends else ForwardMode.DECODE
        for index, layer in enumerate(layers):
            output = root.forward(
                None,
                None,
                None,
                None,
                pool,
                mode,
                1,
                save_kv_cache=True,
                record_kv_cache=None,
                mixed_qkv=raw[index, start:end].clone(),
                conv_weights=weights[index],
                bias=None,
                activation="silu",
                key_dim=heads * dim * 8,
                value_dim=heads * dim * 8,
                attention_tp_size=8,
                head_k_dim=dim,
                head_v_dim=dim,
                f_a_out=f_a[index, start:end],
                f_b_weight=f_b[index],
                beta_raw=beta[index, start:end],
                A_log=a_log,
                dt_bias=bias[index],
                lower_bound=-5.0,
                output_gate=None,
                norm_weight=None,
                norm_eps=None,
                layer_id=layer,
                seq_len=count,
            )
            assert torch.isfinite(output).all()
        counts = torch.tensor([accepted], dtype=torch.int32, device="cuda")
        root.commit_state_after_verify(counts, num_extends=extends)
        if not extends:
            assert root.state_commit_validity(1, num_extends=0).all()
        # Keep the exact boundary produced by prefill or accepted decode.
        endpoint = end if extends else start + accepted
        if (request_id == "parent" and prompt > 128 and extends) or endpoint == 128:
            for layer in layers:
                replay = pool.get_replay_buffers(layer)
                page = int(tables[replay.checkpoint_group_id][0, 0])
                conv, state = pool.get_state_buffers(layer)
                snapshots[layer] = (page, conv[page].clone(), state[page].clone())
        tokens = list(range(1000 + len(history), 1000 + len(history) + accepted))
        feedback(request_id, tokens, not extends)
        history.extend(tokens)
        return end, incomplete

    request = ts.RequestSpec()
    request.request_id, request.tokens, request.max_new_tokens = "parent", history, 128
    scheduler.submit_requests([request])
    execute("parent", 1, 0)
    for _ in range(rounds):
        execute("parent", acceptance, None)
    if rounds > 1:
        assert saw_history
    if width == 4 and capacity == 8 and rounds > 2:
        assert saw_flush
    finish = ts.ForwardEvent.Abort() if abort else ts.ForwardEvent.Finish()
    finish.request_id = "parent"
    done = ts.ExecutionEvent()
    done.add_event(finish)
    scheduler.advance(done)
    assert all(not op.request_ids for op in scheduler.next_execution_plan().forward)

    history += [42] * 11
    request = ts.RequestSpec()
    request.request_id, request.tokens, request.max_new_tokens = (
        "continuation",
        history,
        32,
    )
    scheduler.submit_requests([request])
    end, incomplete = execute("continuation", 1, expected_hit)
    if incomplete:
        _, incomplete = execute("continuation", 1, end)
    assert not incomplete
    # Every new decode starts from the prefill endpoint, not a stale stamp.
    saw_history = False
    execute("continuation", 1, None)
    assert not saw_history
    for layer, (page, old_conv, old_state) in snapshots.items():
        if expected_hit:
            conv, state = pool.get_state_buffers(layer)
            torch.testing.assert_close(conv[page], old_conv, rtol=0, atol=0)
            torch.testing.assert_close(state[page], old_state, rtol=0, atol=0)
    finish = ts.ForwardEvent.Finish()
    finish.request_id = "continuation"
    done = ts.ExecutionEvent()
    done.add_event(finish)
    scheduler.advance(done)
    assert all(not op.request_ids for op in scheduler.next_execution_plan().forward)
    assert scheduler.available_lcm_blocks() == memory_plan.num_lcm_blocks
