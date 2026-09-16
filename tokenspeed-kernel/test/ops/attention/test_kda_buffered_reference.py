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

"""CPU semantic gates; no CUDA runtime or serving backend is needed."""

from __future__ import annotations

import copy

import pytest
import torch
from kda_buffered_reference import (
    BufferedKdaReference,
    kda_log_decay,
    sequential_kda,
)

# Separate FP32 reduction order, not permission for low-precision history drift.
FP32_ATOL = 2e-5
FP32_RTOL = 2e-4


def inputs(seed, width, heads, key_dim, value_dim, conv_width):
    rng = torch.Generator().manual_seed(seed)

    def normal(shape):
        return torch.randn(shape, generator=rng) * 0.2

    return (
        normal((heads, value_dim, key_dim)),
        normal((heads, 2 * key_dim + value_dim, conv_width - 1)),
        normal((width, heads, 2 * key_dim + value_dim)),
        normal((heads, 2 * key_dim + value_dim, conv_width)),
        -torch.rand((width, heads, key_dim), generator=rng) * 0.3,
        torch.rand((width, heads), generator=rng),
    )


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=FP32_ATOL, rtol=FP32_RTOL)


@pytest.mark.parametrize("max_window", [1, 4])
@pytest.mark.parametrize("extra_capacity", [0, 9])
@pytest.mark.parametrize("conv_width", [1, 4])
def test_rounds_rejection_flush_wrap_and_materialization(
    max_window, extra_capacity, conv_width
):
    capacity = 2 * max_window + extra_capacity
    state, window, _, weight, _, _ = inputs(1, max_window, 2, 7, 5, conv_width)
    reference = BufferedKdaReference(
        state, window, 127, capacity, max_window, torch.float32
    )
    endpoint = 127
    seen_starts = set()
    for step in range(90):
        _, _, raw, _, log_decay, beta = inputs(
            100 + step, max_window, 2, 7, 5, conv_width
        )
        width = 0 if step % 13 == 0 else 1 + step % max_window
        accepted = step % (width + 1)
        expected, _, _ = sequential_kda(
            state, window, raw[:width], weight, log_decay[:width], beta[:width]
        )
        checkpoint_before = reference.checkpoint.clone()
        stores_before = reference.state_stores
        output = reference.forward(raw, weight, log_decay, beta, width)
        close(output, expected)
        # Forward may flush only old accepted history, never candidate state.
        close(reference.reconstruct(), state)
        if reference.state_stores == stores_before:
            assert torch.equal(reference.checkpoint, checkpoint_before)
        else:
            close(reference.checkpoint, state)
        reference.commit(accepted)
        _, state, window = sequential_kda(
            state, window, raw[:accepted], weight, log_decay[:accepted], beta[:accepted]
        )
        endpoint += accepted
        close(reference.reconstruct(), state)
        assert torch.equal(reference.conv_window, window)
        assert reference.endpoint == endpoint
        assert reference.checkpoint_position + reference.length == endpoint
        assert reference.length + max_window <= capacity
        seen_starts.add(reference.start)
        if step == 37:
            close(reference.materialize(), state)
            assert reference.length == 0
            # Exact endpoint handoff seeds the next prefill/decode lifetime.
            reference = BufferedKdaReference(
                reference.checkpoint,
                window,
                endpoint,
                capacity,
                max_window,
                torch.float32,
            )
    assert reference.flushes > 1
    assert len(seen_starts) > 1
    close(reference.materialize(), state)
    assert reference.checkpoint_position == endpoint


@pytest.mark.parametrize("accepted", [0, 1, 3, 4])
def test_candidate_suffix_independence_and_protocol_guards(accepted):
    state, window, raw, weight, gate, beta = inputs(9, 4, 2, 7, 5, 4)
    original = BufferedKdaReference(state, window, 51_200, 9, 4, torch.float32)
    modified = copy.deepcopy(original)
    changed = raw.clone()
    changed[accepted:] = 1000
    out = original.forward(raw, weight, gate, beta, 4)
    altered = modified.forward(changed, weight, gate, beta, 4)
    with pytest.raises(RuntimeError, match="pending round"):
        original.materialize()
    with pytest.raises(RuntimeError, match="previous round"):
        original.forward(raw, weight, gate, beta, 4)
    with pytest.raises(ValueError, match="accepted_inputs"):
        original.commit(5)
    assert torch.equal(out[:accepted], altered[:accepted])
    original.commit(accepted)
    modified.commit(accepted)
    assert torch.equal(original.reconstruct(), modified.reconstruct())
    assert torch.equal(original.conv_window, modified.conv_window)
    # Poison every uncommitted slot. Accepted reconstruction must ignore them.
    for offset in range(original.length, original.capacity):
        slot = (original.start + offset) % original.capacity
        original.keys[slot].fill_(float("nan"))
        original.corrections[slot].fill_(float("nan"))
        original.decays[slot].fill_(float("nan"))
    assert torch.equal(original.reconstruct(), modified.reconstruct())
    with pytest.raises(RuntimeError, match="no pending"):
        original.commit(0)


def test_idle_does_not_flush_and_capacity_is_not_a_prefix_granularity():
    state, window, raw, weight, gate, beta = inputs(7, 4, 2, 7, 5, 4)
    for capacity in (8, 9, 13, 64):
        ref = BufferedKdaReference(state, window, 131, capacity, 4, torch.float32)
        ref.forward(raw, weight, gate, beta, 4)
        ref.commit(4)
        before = (ref.start, ref.length, ref.endpoint, ref.state_stores)
        ref.forward(raw * float("nan"), weight, gate, beta, 0)
        ref.commit(0)
        assert (ref.start, ref.length, ref.endpoint, ref.state_stores) == before
    for capacity, width in ((0, 1), (7, 4), (8, 0)):
        with pytest.raises(ValueError, match="capacity"):
            BufferedKdaReference(state, window, 0, capacity, width, torch.float32)


@pytest.mark.parametrize("lower_bound", [None, -5.0])
def test_channel_decay_is_not_a_scalar_gate(lower_bound):
    raw = torch.tensor([[[-0.5, 0.1, 0.4], [0.2, -0.7, 0.3]]])
    a_log = torch.tensor([0.1, 0.3])
    bias = torch.tensor([[0.0, 0.2, 0.4], [0.2, 0.1, 0.0]])
    log_decay = kda_log_decay(raw, a_log, bias, lower_bound)
    assert torch.all(log_decay < 0)
    assert torch.all(log_decay[..., 0] != log_decay[..., 1])
    if lower_bound is not None:
        assert torch.all(log_decay >= lower_bound)
    for head in range(2):
        for channel in range(3):
            x = raw[0, head, channel].double() + bias[head, channel].double()
            rate = a_log[head].double().exp()
            expected = (
                -rate * torch.log1p(x.exp())
                if lower_bound is None
                else lower_bound / (1 + (-rate * x).exp())
            )
            close(log_decay[0, head, channel], expected.float())


def test_tp8_geometry_reordered_requests_and_different_flush_decisions():
    # Full per-layer K3 TP8 geometry. Each request owns its state; batch order
    # never identifies a ring. CPU reference only, not a graph/lifecycle test.
    requests = []
    for request in range(3):
        state, window, _, weight, _, _ = inputs(700 + request, 4, 12, 128, 128, 4)
        requests.append(
            (
                BufferedKdaReference(
                    state, window, 51_327 + request, 16, 4, torch.float32
                ),
                state,
                window,
                weight,
            )
        )
    for step in range(16):
        for request in ((2, 0, 1) if step % 2 else (0, 1, 2)):
            buffered, state, window, weight = requests[request]
            _, _, raw, _, gate, beta = inputs(
                900 + step * 3 + request, 4, 12, 128, 128, 4
            )
            accepted = request + 1
            expected, _, _ = sequential_kda(state, window, raw, weight, gate, beta)
            close(buffered.forward(raw, weight, gate, beta, 4), expected)
            buffered.commit(accepted)
            _, state, window = sequential_kda(
                state, window, raw[:accepted], weight, gate[:accepted], beta[:accepted]
            )
            close(buffered.reconstruct(), state)
            assert torch.equal(buffered.conv_window, window)
            requests[request] = buffered, state, window, weight
    assert len({request[0].flushes for request in requests}) > 1


@pytest.mark.parametrize("capacity", [8, 32, 64])
def test_long_weak_decay_requires_fp32_history(capacity):
    # Decays near one deliberately expose BF16 history drift. This is an
    # adversarial numerical gate, not a claim about real-model distributions.
    state, window, _, weight, _, _ = inputs(17, 4, 2, 16, 12, 4)
    precise = BufferedKdaReference(state, window, 0, capacity, 4, torch.float32)
    rounded = BufferedKdaReference(state, window, 0, capacity, 4, torch.bfloat16)
    max_precise = 0.0
    max_rounded = 0.0
    for step in range(256):
        _, _, raw, _, gate, beta = inputs(500 + step, 4, 2, 16, 12, 4)
        gate *= 0.003
        accepted = 1 + step % 4
        expected, _, _ = sequential_kda(state, window, raw, weight, gate, beta)
        close(precise.forward(raw, weight, gate, beta, 4), expected)
        rounded.forward(raw, weight, gate, beta, 4)
        precise.commit(accepted)
        rounded.commit(accepted)
        _, state, window = sequential_kda(
            state, window, raw[:accepted], weight, gate[:accepted], beta[:accepted]
        )
        close(precise.reconstruct(), state)
        max_precise = max(
            max_precise, (precise.reconstruct() - state).abs().max().item()
        )
        max_rounded = max(
            max_rounded, (rounded.reconstruct() - state).abs().max().item()
        )
    print(
        f"capacity={capacity} fp32_max_state_abs={max_precise:.9g} bf16_max_state_abs={max_rounded:.9g}"
    )
    assert max_precise < FP32_ATOL
    assert max_rounded > 10 * FP32_ATOL
