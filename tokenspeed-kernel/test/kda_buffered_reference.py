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

"""Executable KDA buffered-state specification, not a serving implementation.

State is V-major: [heads, value channels, key channels]. For normalized k,
per-key-channel decay d, and scalar beta per head:

    decayed = state * d
    u = beta * (v - decayed @ k)
    state = decayed + u outer k

Saving (k, u, d) makes reconstruction independent of rejected future inputs.
All recurrence arithmetic is FP32. History storage dtype is explicit so tests
can measure rounding at the proposed cache boundary before adopting a layout.
Python loops and allocations below are intentional reference operations; this
class must never own runtime request state or be imported by serving code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kda_log_decay(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> torch.Tensor:
    """Convert [T,H,K] raw gates to FP32 log decay, including K3 safe gating."""
    gate = raw_gate.float() + dt_bias.float()
    rate = a_log.float().exp()[None, :, None]
    if lower_bound is None:
        return -rate * F.softplus(gate)
    return lower_bound * torch.sigmoid(rate * gate)


def sequential_kda(
    state: torch.Tensor,
    conv_window: torch.Tensor,
    raw_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct token recurrence oracle; returns outputs, final state and window.

    raw_qkv is [T,H,2*K+V]; conv_window is [H,2*K+V,W-1] containing
    accepted pre-convolution inputs oldest first. beta is already sigmoid-
    transformed [T,H]. No history compression or speculative bookkeeping is
    used in this independent spelling of the recurrence.
    """
    state = state.float().clone()
    window = conv_window.clone()
    heads, value_dim, key_dim = state.shape
    outputs = []
    for token in range(raw_qkv.shape[0]):
        full_window = torch.cat((window, raw_qkv[token, ..., None]), dim=-1)
        projected = F.silu((full_window.float() * conv_weight.float()).sum(-1))
        window = full_window[..., 1:]
        query, key, value = projected.split((key_dim, key_dim, value_dim), dim=-1)
        query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        decay = log_decay[token].float().exp()
        # Matrix form intentionally differs from the reconstruction below.
        decayed = state @ torch.diag_embed(decay)
        prediction = torch.einsum("hvk,hk->hv", decayed, key)
        correction = beta[token].float()[:, None] * (value - prediction)
        state = decayed + torch.einsum("hv,hk->hvk", correction, key)
        outputs.append(torch.einsum("hvk,hk->hv", state, query) / key_dim**0.5)
    output = torch.stack(outputs) if outputs else state.new_empty((0, heads, value_dim))
    return output, state, window


class BufferedKdaReference:
    """One request/layer's checkpoint, committed ring and candidate window.

    Constructor inputs model a materialized prefill endpoint. capacity counts
    committed AND candidate entries; max_window is the service's fixed maximum
    execution width, not the current number of valid tokens. No padding row
    may call forward with a positive valid width.
    """

    def __init__(
        self,
        checkpoint: torch.Tensor,
        conv_window: torch.Tensor,
        checkpoint_position: int,
        capacity: int,
        max_window: int,
        history_dtype: torch.dtype,
    ) -> None:
        if max_window < 1 or capacity < 2 * max_window:
            raise ValueError("capacity must be at least twice the positive max_window")
        if checkpoint_position < 0:
            raise ValueError("checkpoint_position must be nonnegative")
        if history_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("reference history must be FP32 or BF16")
        heads, value_dim, key_dim = checkpoint.shape
        if (
            conv_window.shape[:2] != (heads, 2 * key_dim + value_dim)
            or conv_window.ndim != 3
        ):
            raise ValueError("conv_window must contain the packed QKV channel window")
        self.checkpoint = checkpoint.float().clone()
        self.conv_window = conv_window.clone()
        self.checkpoint_position = checkpoint_position
        self.endpoint = checkpoint_position
        self.capacity = capacity
        self.max_window = max_window
        self.start = 0
        self.length = 0
        self.pending = 0
        self.round_open = False
        self.flushes = 0
        self.state_stores = 0
        options = {"device": checkpoint.device, "dtype": history_dtype}
        self.keys = torch.full((capacity, heads, key_dim), float("nan"), **options)
        self.corrections = torch.full(
            (capacity, heads, value_dim), float("nan"), **options
        )
        self.decays = torch.full((capacity, heads, key_dim), float("nan"), **options)
        self.raw_qkv = torch.full(
            (max_window, heads, 2 * key_dim + value_dim),
            float("nan"),
            dtype=conv_window.dtype,
            device=checkpoint.device,
        )

    def reconstruct(self) -> torch.Tensor:
        """Return the accepted endpoint without modifying or reading candidates."""
        state = self.checkpoint.clone()
        for offset in range(self.length):
            slot = (self.start + offset) % self.capacity
            state = (
                state * self.decays[slot].float()[:, None, :]
                + self.corrections[slot].float()[:, :, None]
                * self.keys[slot].float()[:, None, :]
            )
        return state

    def _flush(self, state: torch.Tensor) -> None:
        if self.length:
            self.checkpoint.copy_(state)
            self.checkpoint_position = self.endpoint
            self.start = (self.start + self.length) % self.capacity
            self.length = 0
            self.flushes += 1
            self.state_stores += 1

    def materialize(self) -> torch.Tensor:
        """Flush accepted history for an exact-state handoff; no candidate state."""
        if self.round_open:
            raise RuntimeError("resolve the pending round before handing off state")
        state = self.reconstruct()
        self._flush(state)
        return state

    def forward(
        self,
        raw_qkv: torch.Tensor,
        conv_weight: torch.Tensor,
        log_decay: torch.Tensor,
        beta: torch.Tensor,
        valid_tokens: int,
    ) -> torch.Tensor:
        """Compute candidates without advancing the accepted endpoint.

        Inputs use sequential_kda's layouts. Only the first valid_tokens rows
        are read; zero is the padding/idle case. Returns [valid_tokens,H,V].
        A subsequent commit is required even for zero width or zero acceptance.
        """
        if self.round_open:
            raise RuntimeError("commit the previous round before starting another")
        if not 0 <= valid_tokens <= min(self.max_window, raw_qkv.shape[0]):
            raise ValueError("valid_tokens exceeds the input or maximum window")
        heads, value_dim, key_dim = self.checkpoint.shape
        if raw_qkv.shape[1:] != (heads, 2 * key_dim + value_dim):
            raise ValueError("raw_qkv has incompatible channel geometry")
        if conv_weight.shape != (
            *self.conv_window.shape[:2],
            self.conv_window.shape[-1] + 1,
        ):
            raise ValueError("conv_weight has incompatible window geometry")
        if (
            log_decay.shape != (*raw_qkv.shape[:2], key_dim)
            or beta.shape != raw_qkv.shape[:2]
        ):
            raise ValueError("gate or beta has incompatible token geometry")
        self.round_open = True
        self.pending = valid_tokens
        if valid_tokens == 0:
            return self.checkpoint.new_empty((0, heads, value_dim))
        state = self.reconstruct()
        if self.length + 2 * self.max_window > self.capacity:
            self._flush(state)
        window = self.conv_window.clone()
        outputs = []
        for token in range(valid_tokens):
            full_window = torch.cat((window, raw_qkv[token, ..., None]), dim=-1)
            projected = F.silu((full_window.float() * conv_weight.float()).sum(-1))
            window = full_window[..., 1:]
            query, key, value = projected.split((key_dim, key_dim, value_dim), dim=-1)
            query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            key = key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            decay = log_decay[token].float().exp()
            state = state * decay[:, None, :]
            correction = beta[token].float()[:, None] * (
                value - (state * key[:, None, :]).sum(-1)
            )
            slot = (self.start + self.length + token) % self.capacity
            self.keys[slot].copy_(key)
            self.corrections[slot].copy_(correction)
            self.decays[slot].copy_(decay)
            self.raw_qkv[token].copy_(raw_qkv[token])
            state = state + correction[:, :, None] * key[:, None, :]
            outputs.append((state * query[:, None, :]).sum(-1) / key_dim**0.5)
        return torch.stack(outputs)

    def commit(self, accepted_inputs: int) -> None:
        """Accept a candidate prefix; zero discards all candidates, with no +1.

        The serving adapter must mask inactive rows and pass actual state-input
        counts. For ordinary live decode this is one. This primitive also
        supports zero for cancellation and low-level rejection tests.
        """
        if not self.round_open:
            raise RuntimeError("no pending round to commit")
        if not 0 <= accepted_inputs <= self.pending:
            raise ValueError(
                "accepted_inputs must be within the valid candidate window"
            )
        for offset in range(accepted_inputs):
            full_window = torch.cat(
                (self.conv_window, self.raw_qkv[offset, ..., None]), dim=-1
            )
            self.conv_window = full_window[..., 1:]
        self.length += accepted_inputs
        self.endpoint += accepted_inputs
        self.pending = 0
        self.round_open = False
        assert self.endpoint == self.checkpoint_position + self.length
