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
    KdaPrefillGraphCache,
    _clone_metadata,
)


def metadata(device, page):
    return MambaForwardMetadata(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
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
    cache = KdaPrefillGraphCache(max_shapes=1)
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

    # Capacity overflow runs the original callable without growing the cache.
    cache.run(backend, 0, 2, {"value": value}, forward)
    assert len(cache.schedules) == 1
    assert cache.replays == 4


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
    cache = KdaPrefillGraphCache(max_shapes=1)
    live = metadata("cuda", 1)
    backend = SimpleNamespace(forward_metadata=live)

    def forward():
        raise ValueError("test failure")

    with pytest.raises(ValueError, match="test failure"):
        cache.run(backend, 0, 1, {}, forward)
    assert backend.forward_metadata is live


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pool_sharing_is_scoped_to_replay_stream():
    cache = KdaPrefillGraphCache(max_shapes=8)
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
