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

"""Regression coverage for the capacity-based prefill graph owner."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.state.mamba import (
    MambaForwardMetadata,
)
from tokenspeed.runtime.layers.attention.backends.state.prefill_graph import (
    KdaOuterGraphBinding,
    KdaPrefillGraphCache,
    _clone_metadata,
)


def metadata(device, page):
    return MambaForwardMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        scan_query_start_loc=torch.tensor([0, 1], device=device),
        query_start_loc_int64=torch.tensor([0, 1], device=device),
        extend_seq_lens_cpu=torch.tensor([1]),
        cu_extend_seq_lens_cpu=torch.tensor([0, 1]),
        state_in_blocks_by_group={"state": torch.tensor([page], device=device)},
        state_out_blocks_by_group={"state": torch.tensor([page], device=device)},
    )


def test_metadata_snapshot_does_not_alias():
    source = metadata("cpu", 1)
    cloned = _clone_metadata(source)
    source.state_in_blocks_by_group["state"].zero_()
    assert cloned.state_in_blocks_by_group["state"].item() == 1
    assert (
        cloned.extend_seq_lens_cpu.data_ptr() != source.extend_seq_lens_cpu.data_ptr()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_replay_refreshes_pages_and_writes_exactly_once():
    cache = KdaPrefillGraphCache()
    backend = SimpleNamespace(forward_metadata=metadata("cuda", 1))
    value = torch.ones(1, device="cuda")
    state = torch.zeros(4, device="cuda")

    def forward():
        pages = backend.forward_metadata.state_out_blocks_by_group["state"]
        state[pages] += value
        return state[pages].clone()

    for page in (1, 2, 3, 1, 2):
        live = metadata("cuda", page)
        backend.forward_metadata = live
        before = state.clone()
        result = cache.run(backend, 0, 1, {"value": value}, forward)
        torch.testing.assert_close(result, before[page : page + 1] + 1)
        before[page] += 1
        torch.testing.assert_close(state, before)
        assert backend.forward_metadata is live
    assert cache.captures == 1
    assert cache.replays == 4

    # Another input allocation must not replay the captured old address.
    value = torch.full_like(value, 3)
    before = state.clone()
    cache.run(backend, 0, 1, {"value": value}, forward)
    before[2] += 3
    torch.testing.assert_close(state, before)
    assert cache.replays == 4

    # Every outer token bucket remains eligible, including the ninth and later.
    for bucket in range(2, 11):
        for _ in range(3):
            backend.forward_metadata = metadata("cuda", 2)
            before = state.clone()
            result = cache.run(backend, 0, bucket, {"value": value}, forward)
            before[2] += 3
            torch.testing.assert_close(state, before)
            torch.testing.assert_close(result, before[2:3])
    assert len(cache.schedules) == 10
    assert cache.captures == 10
    assert cache.replays == 22


def test_hybrid_reinitialization_clears_kda_graphs():
    from unittest.mock import Mock

    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend

    child = object.__new__(KdaAttnBackend)
    child._prefill_graph_cache = object()
    backend = object.__new__(HybridLinearAttnBackend)
    backend.full_attn_backend = Mock()
    backend.linear_attn_backend = child
    backend.init_prefill_graph_state(1024, 4)
    assert child._prefill_graph_cache is None
    backend.full_attn_backend.init_prefill_graph_state.assert_called_once_with(1024, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_failure_restores_live_metadata():
    cache = KdaPrefillGraphCache()
    live = metadata("cuda", 1)
    backend = SimpleNamespace(forward_metadata=live)

    def forward():
        raise ValueError("test failure")

    with pytest.raises(ValueError, match="test failure"):
        cache.run(backend, 0, 1, {}, forward)
    assert backend.forward_metadata is live


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pool_sharing_is_scoped_to_replay_stream():
    cache = KdaPrefillGraphCache()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    cases = []

    def bind_forward(value):
        def forward():
            return (value.square() + 3).clone()

        return forward

    for stream in streams:
        with torch.cuda.stream(stream):
            for layer in range(2):
                backend = SimpleNamespace(forward_metadata=metadata("cuda", 1))
                value = torch.full((257,), layer + 1.0, device="cuda")

                forward = bind_forward(value)
                for _ in range(2):
                    cache.run(backend, layer, 257, {"value": value}, forward)
                cases.append((stream, backend, layer, value, forward))
    assert len(cache._capture_resources) == 2
    pools = [resources[0] for resources in cache._capture_resources.values()]
    assert pools[0] != pools[1]
    outputs = []
    # Queue both streams without a host synchronization between replays.
    for stream, backend, layer, value, forward in reversed(cases):
        with torch.cuda.stream(stream):
            value.add_(2)
            result = cache.run(backend, layer, 257, {"value": value}, forward)
            outputs.append((result.clone(), value.square() + 3))
    torch.cuda.synchronize()
    for actual, expected in outputs:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cache.captures == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_outer_graph_inlines_state_layers_and_retains_full_attention_break():
    from tokenspeed.runtime.execution.breakable_cuda_graph import BreakableCapture
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
        HybridLinearAttnBackend,
    )

    state = torch.zeros(4, device="cuda")
    value = torch.ones(8, 4, device="cuda")

    class Leaf(AttentionBackend):
        def forward_extend(self, q, k, v, layer, pool, bs, **kwargs):
            if layer.layer_id == 1:
                return q + 2
            pages = self.forward_metadata.state_out_blocks_by_group["state"]
            state[pages] += 1
            valid = self.forward_metadata.query_start_loc[-1]
            return (q + state[pages]).masked_fill(
                (torch.arange(q.shape[0], device=q.device) >= valid)[:, None], 0
            )

    leaf = object.__new__(Leaf)
    leaf.cache_pool = object()
    leaf.forward_metadata = metadata("cuda", 1)
    full = object.__new__(Leaf)
    full.device = torch.device("cuda")
    hybrid = HybridLinearAttnBackend(full, leaf, [1])
    binding = KdaOuterGraphBinding(leaf, 8)

    def forward():
        out = value * 2
        for layer_id in (0, 1, 2):
            out = hybrid.forward(
                out,
                None,
                None,
                SimpleNamespace(layer_id=layer_id),
                None,
                ForwardMode.EXTEND,
                1,
                True,
                None,
            )
        return out * 3

    forward()
    torch.cuda.synchronize()
    ordinary = BreakableCapture()
    with ordinary:
        forward()
    assert ordinary.num_segments == 7  # Three eager breaks + four graphs.
    with binding.bind(refresh=False):
        forward()
        torch.cuda.synchronize()
        merged = BreakableCapture()
        with merged:
            output = forward()
    assert merged.num_segments == 3  # Only full attention remains an eager break.

    ctx = SimpleNamespace(forward_mode=ForwardMode.EXTEND, bs=1, num_extends=1)
    for length, page in ((1, 2), (7, 3), (8, 1), (3, 2)):
        live = metadata("cuda", page)
        live.query_start_loc[-1] = length
        live.query_start_loc_int64[-1] = length
        live.scan_query_start_loc[-1] = length
        live.extend_seq_lens_cpu[0] = length
        live.cu_extend_seq_lens_cpu[-1] = length
        leaf.forward_metadata = live
        assert binding.compatible(ctx)
        before = state.clone()
        with binding.bind(refresh=True):
            merged.replay(valid_rows=length)
        expected = torch.zeros_like(output)
        expected[:length] = (value[:length] * 2 + 2 * before[page] + 5) * 3
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        before[page] += 2
        torch.testing.assert_close(state, before, rtol=0, atol=0)
        assert leaf.forward_metadata is live
        assert not leaf.prefill_graph_inline

    ctx.bs = ctx.num_extends = 2
    assert not binding.compatible(ctx)
    ctx.bs = ctx.num_extends = 1
    ctx.forward_mode = ForwardMode.MIXED
    assert not binding.compatible(ctx)
    ctx.forward_mode = ForwardMode.EXTEND
    leaf.forward_metadata.prefill_checkpoint_batch = object()
    assert not binding.compatible(ctx)
    leaf.forward_metadata.prefill_checkpoint_batch = None
    leaf.cache_pool = object()
    assert not binding.compatible(ctx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_outer_binding_restores_metadata_after_failure():
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend

    backend = object.__new__(AttentionBackend)
    backend.cache_pool = object()
    original = metadata("cuda", 1)
    backend.forward_metadata = original
    binding = KdaOuterGraphBinding(backend, 8)
    with pytest.raises(ValueError, match="test failure"):
        with binding.bind(refresh=True):
            assert backend.prefill_graph_inline
            raise ValueError("test failure")
    assert backend.forward_metadata is original
    assert not backend.prefill_graph_inline


def test_internal_checkpoint_batch_bypasses_capacity_graph():
    live = metadata("cpu", 1)
    live.prefill_checkpoint_batch = object()
    backend = SimpleNamespace(forward_metadata=live)
    cache = KdaPrefillGraphCache()
    calls = []

    def forward():
        calls.append(backend.forward_metadata)
        return "eager checkpoint result"

    assert cache.run(backend, 0, 8, {}, forward) == "eager checkpoint result"
    assert calls == [live]
    assert cache.schedules == {}
    with pytest.raises(ValueError, match="internal checkpoint"):
        KdaOuterGraphBinding(SimpleNamespace(cache_pool=None, forward_metadata=live), 8)


@pytest.mark.parametrize(
    "compatible,transfer,expected",
    [(True, False, 2), (False, False, 1), (True, True, 1)],
)
def test_outer_owner_selects_matching_graph_and_refreshes_before_replay(
    compatible, transfer, expected
):
    from contextlib import contextmanager, nullcontext
    from unittest.mock import patch

    from tokenspeed.runtime.execution.prefill_graph import CapturedForward, PrefillGraph

    events = []

    class Binding:
        def compatible(self, ctx):
            return compatible

        @contextmanager
        def bind(self, refresh):
            assert refresh
            events.append("refresh")
            try:
                yield
            finally:
                events.append("restore")

    def capture(label):
        return SimpleNamespace(replay=lambda **kwargs: events.append(label))

    owner = object.__new__(PrefillGraph)
    owner._captures = {8: capture("ordinary")}
    owner._outputs = {8: CapturedForward(torch.ones(8, 4), None)}
    owner._inline_captures = {
        8: (
            capture("inline"),
            CapturedForward(torch.full((8, 4), 2.0), None),
            [Binding()],
        )
    }
    owner.attn_backend = SimpleNamespace(step_counter=object() if transfer else None)
    owner._replay_bucket = lambda ctx: 8
    owner._log_engaged_once = lambda *args: None
    owner._embed_tokens = lambda ids: torch.zeros(8, 4)
    owner._land_input_embeds = lambda *args: None
    owner._padded_to = lambda *args: nullcontext()
    owner.text_model = SimpleNamespace(
        lm_head=None, logits_processor=lambda ids, hidden, *args: hidden
    )
    ctx = SimpleNamespace(input_num_tokens=8)
    with patch(
        "tokenspeed.runtime.execution.prefill_graph.LogitsMetadata.from_forward_context",
        return_value=None,
    ):
        result = owner.replay(ctx, torch.zeros(8, dtype=torch.int64), None)
    torch.testing.assert_close(result, torch.full((8, 4), float(expected)))
    assert events == (
        ["refresh", "inline", "restore"] if expected == 2 else ["ordinary"]
    )
