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

"""Inputs and an independent FP32 reference for private prefill experiments."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
    build_causal_conv1d_prefill_metadata,
)
from tokenspeed_kernel.ops.conv._triton.prefill_candidate import (
    PrefillCandidateConfig,
    launch_prefill_candidate,
)

MAIN_LENGTHS = [[8192], [4096, 4096], [1330] * 6 + [212], [1330] * 6 + [106, 106]]
BASELINE = PrefillCandidateConfig("serial", 8, 256, 4)


def assert_bitwise(actual, expected):
    bits = torch.int16 if actual.element_size() == 2 else torch.int32
    torch.testing.assert_close(actual.view(bits), expected.view(bits), atol=0, rtol=0)


@dataclass
class PrefillCase:
    lengths: list[int]
    x: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor | None
    initial_state: torch.Tensor
    slots: torch.Tensor
    history: torch.Tensor
    boundaries: torch.Tensor

    def metadata(self, block_m):
        return build_causal_conv1d_prefill_metadata(
            self.boundaries, torch.tensor(self.lengths, dtype=torch.int32), block_m
        )

    def launch(self, config, state, out, metadata, *, silu, enable_pdl):
        return launch_prefill_candidate(
            self.x,
            self.weight,
            self.bias,
            state,
            self.slots,
            self.history,
            self.boundaries,
            out,
            metadata=metadata,
            config=config,
            silu=silu,
            enable_pdl=enable_pdl,
        )


def make_case(lengths, channels, width, dtype, *, device, strided, with_bias, pad):
    """Use reversed state slots, mixed histories, and spare state columns."""
    total = sum(lengths)
    x = torch.randn(
        total, channels + (13 if strided else 0), device=device, dtype=dtype
    )
    x = x[:, :channels].T
    weight = torch.randn(channels, width, device=device, dtype=dtype) * 0.1
    state = torch.randn(
        len(lengths) + 2, channels, width + 1, device=device, dtype=dtype
    )
    slots = torch.arange(len(lengths), 0, -1, device=device, dtype=torch.int32)
    if pad and lengths:
        slots[len(lengths) // 2] = -1
    history = torch.arange(len(lengths), device=device) % 2 == 1
    bounds = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32
    )
    bias = (
        torch.randn(channels, device=device, dtype=dtype) * 0.1 if with_bias else None
    )
    return PrefillCase(list(lengths), x, weight, bias, state, slots, history, bounds)


def reference(case, *, silu):
    """FP32 convolution math; state stores raw input in its original dtype."""
    out = torch.full_like(case.x, float("nan"))
    state = case.initial_state.clone()
    width = case.weight.shape[1]
    begin = 0
    for length, slot, history in zip(
        case.lengths, case.slots.tolist(), case.history.tolist(), strict=True
    ):
        if length and slot != -1:
            prefix = state[slot, :, : width - 1].clone()
            if not history:
                prefix.zero_()
            values = torch.cat((prefix, case.x[:, begin : begin + length]), dim=1)
            acc = torch.zeros(
                case.x.shape[0], length, device=case.x.device, dtype=torch.float32
            )
            if case.bias is not None:
                acc = acc + case.bias.float()[:, None]
            for tap in range(width):
                acc = (
                    acc
                    + values[:, tap : tap + length].float()
                    * case.weight[:, tap, None].float()
                )
            out[:, begin : begin + length] = (F.silu(acc) if silu else acc).to(
                case.x.dtype
            )
            state[slot, :, : width - 1] = values[:, -(width - 1) :]
        begin += length
    return out, state


def check_case(case, config, *, silu, enable_pdl):
    """Require exact old-kernel output/state and compare independent FP32 math."""
    outputs = []
    states = []
    binary = None
    for candidate in (BASELINE, config):
        state = case.initial_state.clone()
        out = torch.full_like(case.x, float("nan"))
        binary = case.launch(
            candidate,
            state,
            out,
            case.metadata(candidate.block_m),
            silu=silu,
            enable_pdl=enable_pdl,
        )
        outputs.append(out)
        states.append(state)
    assert_bitwise(outputs[1], outputs[0])
    assert_bitwise(states[1], states[0])
    expected, expected_state = reference(case, silu=silu)
    tolerance = (
        2e-2
        if case.x.dtype == torch.bfloat16
        else (2e-3 if case.x.dtype == torch.float16 else 2e-6)
    )
    torch.testing.assert_close(
        outputs[1], expected, atol=tolerance, rtol=tolerance, equal_nan=True
    )
    assert_bitwise(states[1], expected_state)
    return binary
