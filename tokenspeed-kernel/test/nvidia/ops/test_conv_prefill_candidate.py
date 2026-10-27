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

"""Behavioral coverage for the unregistered token-parallel convolution."""

import itertools

import pytest
import torch
from conv_prefill_utils import (
    BASELINE,
    MAIN_LENGTHS,
    assert_bitwise,
    check_case,
    make_case,
    reference,
)
from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
    build_causal_conv1d_capacity_metadata,
    refresh_causal_conv1d_capacity_metadata,
)
from tokenspeed_kernel.ops.conv._gluon.prefill_candidate import (
    gluon_causal_conv1d_prefill_kernel,
)
from tokenspeed_kernel.ops.conv._triton.prefill_candidate import (
    PrefillCandidateConfig,
    _causal_conv1d_prefill_candidate_kernel,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

CONFIGS = [
    PrefillCandidateConfig(kind, m, n, warps)
    for kind, m, n, warps in itertools.product(
        ("window", "direct"),
        (8, 16, 32, 64, 128),
        (64, 128, 256),
        (4, 8),
    )
]


@pytest.mark.parametrize("config", CONFIGS)
def test_tiles_and_boundaries(config):
    torch.manual_seed(37)
    lengths = [
        1,
        2,
        3,
        7,
        8,
        9,
        0,
        config.block_m - 1,
        config.block_m,
        config.block_m + 1,
    ]
    case = make_case(
        lengths,
        257,
        4,
        torch.bfloat16,
        device="cuda",
        strided=True,
        with_bias=False,
        pad=True,
    )
    check_case(case, config, silu=True, enable_pdl=False)


@pytest.mark.parametrize("kind", ("window", "direct"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("width", (2, 3, 4))
@pytest.mark.parametrize("silu", (False, True))
def test_dtypes_bias_and_history(kind, dtype, width, silu):
    torch.manual_seed(37)
    case = make_case(
        [1, 2, 3, 63, 64, 65],
        65,
        width,
        dtype,
        device="cuda",
        strided=False,
        with_bias=True,
        pad=False,
    )
    check_case(
        case, PrefillCandidateConfig(kind, 64, 128, 4), silu=silu, enable_pdl=False
    )


@pytest.mark.parametrize("kind", ("window", "direct"))
@pytest.mark.parametrize("lengths", MAIN_LENGTHS)
def test_full_token_partitions(kind, lengths):
    case = make_case(
        lengths,
        257,
        4,
        torch.bfloat16,
        device="cuda",
        strided=False,
        with_bias=False,
        pad=False,
    )
    check_case(
        case, PrefillCandidateConfig(kind, 64, 128, 4), silu=True, enable_pdl=False
    )


@pytest.mark.parametrize("kind", ("window", "direct"))
@pytest.mark.parametrize("channels", (64, 128, 257))
def test_fp32_first_tap_rounding(kind, channels):
    torch.manual_seed(37)
    case = make_case(
        [1, 2, 3, 63, 64, 65],
        channels,
        3,
        torch.float32,
        device="cuda",
        strided=False,
        with_bias=True,
        pad=False,
    )
    check_case(
        case, PrefillCandidateConfig(kind, 64, 128, 4), silu=True, enable_pdl=False
    )


@pytest.mark.parametrize("kind", ("window", "direct"))
@pytest.mark.parametrize("lengths", ([], [0], [0, 0, 0]))
def test_empty_requests(kind, lengths):
    case = make_case(
        lengths,
        65,
        4,
        torch.bfloat16,
        device="cuda",
        strided=False,
        with_bias=False,
        pad=False,
    )
    check_case(
        case, PrefillCandidateConfig(kind, 64, 128, 4), silu=True, enable_pdl=False
    )


@pytest.mark.parametrize("kind", ("window", "direct"))
def test_chunk_continuation(kind):
    check_chunk_continuation(PrefillCandidateConfig(kind, 32, 128, 4))


def check_chunk_continuation(config):
    case = make_case(
        [73],
        129,
        4,
        torch.bfloat16,
        device="cuda",
        strided=False,
        with_bias=False,
        pad=False,
    )
    case.history.fill_(True)
    full_x = case.x
    expected_state = case.initial_state.clone()
    expected = torch.empty_like(full_x)
    case.launch(
        BASELINE,
        expected_state,
        expected,
        case.metadata(8),
        silu=True,
        enable_pdl=False,
    )
    state = case.initial_state.clone()
    pieces = []
    begin = 0
    for length in (1, 2, 3, 7, 8, 9, 17, 26):
        case.x = full_x[:, begin : begin + length]
        case.lengths = [length]
        case.boundaries.copy_(
            torch.tensor([0, length], device="cuda", dtype=torch.int32)
        )
        out = torch.empty_like(case.x)
        case.launch(
            config,
            state,
            out,
            case.metadata(config.block_m),
            silu=True,
            enable_pdl=False,
        )
        pieces.append(out)
        begin += length
    assert_bitwise(torch.cat(pieces, dim=1), expected)
    assert_bitwise(state, expected_state)


@pytest.mark.parametrize("kind", ("window", "direct"))
def test_graph_replay_changes_inputs_lengths_and_request_count(kind):
    check_graph_replay_changes_inputs_lengths_and_request_count(
        PrefillCandidateConfig(kind, 32, 128, 4)
    )


def check_graph_replay_changes_inputs_lengths_and_request_count(config):
    capacity = 193
    case = make_case(
        [64, 64, 64, 1],
        129,
        4,
        torch.bfloat16,
        device="cuda",
        strided=False,
        with_bias=False,
        pad=False,
    )
    metadata = build_causal_conv1d_capacity_metadata(
        case.boundaries, capacity, config.block_m
    )
    state = case.initial_state.clone()
    out = torch.empty_like(case.x)

    def launch():
        refresh_causal_conv1d_capacity_metadata(case.boundaries, metadata, capacity)
        case.launch(config, state, out, metadata, silu=True, enable_pdl=False)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    for lengths in ([1, 2, 190, 0], [193, 0, 0, 0], [3, 7, 31, 0], [0, 0, 0, 0]):
        case.lengths = list(lengths)
        case.boundaries.copy_(
            torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()],
                device="cuda",
                dtype=torch.int32,
            )
        )
        case.x.normal_()
        case.history.logical_not_()
        expected, expected_state = reference(case, silu=True)
        state.copy_(case.initial_state)
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2, equal_nan=True)
        assert_bitwise(state, expected_state)
        old = torch.full_like(out, float("nan"))
        old_state = case.initial_state.clone()
        case.launch(
            BASELINE, old_state, old, case.metadata(8), silu=True, enable_pdl=False
        )
        assert_bitwise(out, old)


@pytest.mark.parametrize("kind", ("window", "direct"))
@pytest.mark.parametrize("token_contiguous", (False, True))
def test_dynamic_shapes_do_not_compile(kind, token_contiguous):
    check_dynamic_shapes_do_not_compile(
        PrefillCandidateConfig(kind, 64, 128, 4), token_contiguous
    )


def check_dynamic_shapes_do_not_compile(config, token_contiguous):
    kind = config.implementation

    def launch(lengths):
        case = make_case(
            lengths,
            65,
            4,
            torch.bfloat16,
            device="cuda",
            strided=True,
            with_bias=False,
            pad=False,
        )
        if token_contiguous:
            case.x = case.x.contiguous()
        state = case.initial_state.clone()
        out = torch.empty_like(case.x)
        case.launch(
            config,
            state,
            out,
            case.metadata(config.block_m),
            silu=True,
            enable_pdl=False,
        )
        expected, expected_state = reference(case, silu=True)
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
        assert_bitwise(state, expected_state)

    launch([9, 7])
    kernel = (
        gluon_causal_conv1d_prefill_kernel
        if kind.startswith("gluon_")
        else _causal_conv1d_prefill_candidate_kernel
    )
    with assert_no_triton_compile(kernel):
        for lengths in ([1], [16], [17], [1, 1], [33, 15], [64] * 3, [5, 1, 300]):
            launch(lengths)


@pytest.mark.parametrize("kind", ("window", "direct"))
def test_token_addresses_past_int32(kind):
    if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
        pytest.skip("requires 6 GiB free for the wide-stride allocation")
    config = PrefillCandidateConfig(kind, 64, 128, 4)
    stride = 1 << 16
    length = (1 << 31) // stride + 64
    case = make_case(
        [length - 8, 8],
        16,
        4,
        torch.bfloat16,
        device="cuda",
        strided=False,
        with_bias=False,
        pad=False,
    )
    wide = torch.empty(length, stride, device="cuda", dtype=torch.bfloat16)[:, :16].T
    wide.copy_(case.x)
    case.x = wide
    check_case(case, config, silu=True, enable_pdl=False)
