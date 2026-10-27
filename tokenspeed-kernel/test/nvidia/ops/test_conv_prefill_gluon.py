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

"""NVIDIA behavior coverage for explicitly distributed prefill tiles."""

import itertools

import pytest
import torch
from nvidia.ops import test_conv_prefill_candidate as coverage
from tokenspeed_kernel.ops.conv._triton.prefill_candidate import PrefillCandidateConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
KINDS = (
    "gluon_serial",
    "gluon_direct",
    "gluon_window",
    "gluon_window_r64",
    "gluon_window_r48",
)
CONFIGS = [
    PrefillCandidateConfig(kind, m, n, w)
    for kind, m, n, w in itertools.product(
        KINDS, (4, 8, 16, 32, 64, 128), (128, 256), (4,)
    )
]
CONFIGS += [
    PrefillCandidateConfig("gluon_window_r64", 8, 512, 4),
    PrefillCandidateConfig("gluon_window", 64, 256, 8),
]


@pytest.mark.parametrize("config", CONFIGS)
def test_tiles_and_boundaries(config):
    coverage.test_tiles_and_boundaries(config)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("width", (2, 3, 4))
@pytest.mark.parametrize("silu", (False, True))
def test_dtypes_bias_and_history(kind, dtype, width, silu):
    coverage.test_dtypes_bias_and_history(kind, dtype, width, silu)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("lengths", coverage.MAIN_LENGTHS)
def test_full_token_partitions(kind, lengths):
    coverage.test_full_token_partitions(kind, lengths)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("channels", (64, 128, 257))
def test_fp32_first_tap_rounding(kind, channels):
    coverage.test_fp32_first_tap_rounding(kind, channels)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("lengths", ([], [0], [0, 0, 0]))
def test_empty_requests(kind, lengths):
    coverage.test_empty_requests(kind, lengths)


@pytest.mark.parametrize("kind", KINDS)
def test_chunk_continuation(kind):
    coverage.test_chunk_continuation(kind)


@pytest.mark.parametrize("kind", KINDS)
def test_graph_replay_changes_inputs_lengths_and_request_count(kind):
    coverage.test_graph_replay_changes_inputs_lengths_and_request_count(kind)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("token_contiguous", (False, True))
def test_dynamic_shapes_do_not_compile(kind, token_contiguous):
    coverage.test_dynamic_shapes_do_not_compile(kind, token_contiguous)


@pytest.mark.parametrize("kind", KINDS)
def test_token_addresses_past_int32(kind):
    coverage.test_token_addresses_past_int32(kind)


SELECTED = (
    PrefillCandidateConfig("gluon_serial", 8, 256, 4),
    PrefillCandidateConfig("gluon_window_r64", 8, 512, 4),
)


@pytest.mark.parametrize("config", SELECTED)
def test_selected_graph_replay(config):
    coverage.check_graph_replay_changes_inputs_lengths_and_request_count(config)


@pytest.mark.parametrize("config", SELECTED)
def test_selected_chunk_continuation(config):
    coverage.check_chunk_continuation(config)


@pytest.mark.parametrize("config", SELECTED)
@pytest.mark.parametrize("token_contiguous", (False, True))
def test_selected_dynamic_shapes_do_not_compile(config, token_contiguous):
    coverage.check_dynamic_shapes_do_not_compile(config, token_contiguous)


@pytest.mark.parametrize("config", SELECTED)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("width", (2, 3, 4))
@pytest.mark.parametrize("silu", (False, True))
def test_selected_numerics(config, dtype, width, silu):
    torch.manual_seed(37)
    case = coverage.make_case(
        [1, 2, 3, 7, 8, 9, 33],
        257,
        width,
        dtype,
        device="cuda",
        strided=True,
        with_bias=True,
        pad=False,
    )
    coverage.check_case(case, config, silu=silu, enable_pdl=False)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("silu", (False, True))
def test_signed_zero_bias(kind, dtype, silu):
    case = coverage.make_case(
        [9], 65, 3, dtype, device="cuda", strided=False, with_bias=True, pad=False
    )
    case.x.fill_(-0.0)
    case.weight.fill_(1.0)
    case.bias.fill_(-0.0)
    case.initial_state.fill_(-0.0)
    case.history.fill_(True)
    coverage.check_case(
        case, PrefillCandidateConfig(kind, 8, 512, 4), silu=silu, enable_pdl=False
    )
