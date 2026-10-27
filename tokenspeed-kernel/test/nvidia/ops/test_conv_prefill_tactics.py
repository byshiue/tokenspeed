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

"""Two-tactic selection, state continuity and capture behavior."""

import pytest
import torch
from conv_prefill_utils import assert_bitwise, check_case, make_case
from nvidia.ops import test_conv_prefill_candidate as coverage
from tokenspeed_kernel.ops.attention.gdn._triton.causal_conv1d_metadata import (
    build_causal_conv1d_capacity_metadata,
)
from tokenspeed_kernel.ops.conv._gluon.prefill_candidate import (
    gluon_causal_conv1d_prefill_kernel,
)
from tokenspeed_kernel.ops.conv._triton.causal_conv1d import _causal_conv1d_fwd_kernel
from tokenspeed_kernel.ops.conv._triton.prefill_candidate import (
    PrefillCandidateConfig,
    PrefillTacticPolicy,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
SERIAL = PrefillCandidateConfig("gluon_serial", 8, 256, 4)
POLICY = PrefillTacticPolicy(
    min_parallel_tokens=2048, min_tokens_per_request=128, max_parallel_tokens=8192
)
PARALLEL = PrefillCandidateConfig("gluon_window_r64", 8, 512, 4)


@pytest.mark.parametrize(
    "tokens,requests,expected",
    [
        (0, 0, SERIAL),
        (0, 4, SERIAL),
        (1, 1, SERIAL),
        (255, 1, SERIAL),
        (256, 1, SERIAL),
        (256, 8, SERIAL),
        (256, 9, SERIAL),
        (1023, 1, SERIAL),
        (1023, 32, SERIAL),
        (1024, 1, SERIAL),
        (1025, 1, SERIAL),
        (1024, 32, SERIAL),
        (1024, 128, SERIAL),
        (2047, 1, SERIAL),
        (2048, 1, PARALLEL),
        (2049, 1, PARALLEL),
        (2048, 16, PARALLEL),
        (2048, 17, SERIAL),
        (8191, 64, SERIAL),
        (8192, 8, PARALLEL),
        (8192, 64, PARALLEL),
        (8192, 65, SERIAL),
        (8193, 1, SERIAL),
        (8192, 1024, SERIAL),
    ],
)
def test_selection(tokens, requests, expected):
    assert POLICY.select(tokens, requests, exact_workload=True) == expected


@pytest.mark.parametrize("tokens,average", [(0, 1), (-1, 1), (1, 0), (1, -1)])
def test_invalid_policy(tokens, average):
    with pytest.raises(ValueError):
        PrefillTacticPolicy(tokens, average, max_parallel_tokens=8192)


@pytest.mark.parametrize("tokens,requests", [(-1, 1), (1, -1)])
def test_invalid_extents(tokens, requests):
    with pytest.raises(ValueError):
        POLICY.select(tokens, requests, exact_workload=True)


@pytest.mark.parametrize(
    "lengths",
    [
        [],
        [0, 0],
        [1],
        [2, 3, 7, 8, 9],
        [255],
        [256],
        [257],
        [1023],
        [1024],
        [1025],
        [2047],
        [2048],
        [2049],
        [8193],
        [128] * 16,
        [2033] + [1] * 15,
        [32] * 32,
        [8] * 128,
        [512, 1, 0, 3],
        *coverage.MAIN_LENGTHS,
    ],
)
@pytest.mark.parametrize("strided", (False, True))
def test_outputs_and_states(lengths, strided):
    case = make_case(
        lengths,
        257,
        4,
        torch.bfloat16,
        device="cuda",
        strided=strided,
        with_bias=True,
        pad=len(lengths) > 1,
    )
    check_case(case, POLICY, silu=True, enable_pdl=False)


def test_chunk_continuation_across_tactics():
    coverage.check_chunk_continuation(
        PrefillTacticPolicy(
            min_parallel_tokens=16, min_tokens_per_request=8, max_parallel_tokens=8192
        )
    )


@pytest.mark.parametrize("minimum", (128, 256))
def test_graph_replay_with_capacity_only_work(minimum):
    # Capacity maps must remain correct as live lengths and active requests shrink.
    coverage.check_graph_replay_changes_inputs_lengths_and_request_count(
        PrefillTacticPolicy(
            min_parallel_tokens=minimum,
            min_tokens_per_request=32,
            max_parallel_tokens=8192,
        )
    )


def test_dynamic_batches_do_not_compile():
    # Serving keeps a fixed state pool and model layout while batch sizes vary.
    pool = torch.randn(258, 257, 3, device="cuda", dtype=torch.bfloat16)

    def run(lengths):
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
        case.initial_state = pool.clone()
        check_case(case, POLICY, silu=True, enable_pdl=False)

    run([1])
    run([2048])
    with assert_no_triton_compile(
        _causal_conv1d_fwd_kernel, gluon_causal_conv1d_prefill_kernel
    ):
        for lengths in (
            [2],
            [3],
            [8],
            [63],
            [255],
            [256],
            [257],
            [513],
            [1023],
            [1024],
            [1025],
            [8] * 128,
            [32] * 32,
            [2048, 0, 1],
            [8193],
            [128] * 16,
            [0, 0],
        ):
            run(lengths)


@pytest.mark.parametrize(
    "lengths",
    ([1], [1024], [2048], [8193], [8] * 128, [16] * 64, [32] * 32, [128] * 16),
)
@pytest.mark.parametrize("exact_workload", (False, True))
def test_policy_launches_the_selected_binary(lengths, exact_workload):
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
    metadata = (
        case.metadata(8)
        if exact_workload
        else build_causal_conv1d_capacity_metadata(case.boundaries, sum(lengths), 8)
    )
    chosen = POLICY.select(sum(lengths), len(lengths), exact_workload=exact_workload)
    outputs, states, binaries = [], [], []
    for config in (chosen, POLICY):
        state = case.initial_state.clone()
        out = torch.empty_like(case.x)
        binary = case.launch(config, state, out, metadata, silu=True, enable_pdl=False)
        outputs.append(out)
        states.append(state)
        binaries.append(binary)
    assert_bitwise(outputs[0], outputs[1])
    assert_bitwise(states[0], states[1])
    assert binaries[0].asm["cubin"] == binaries[1].asm["cubin"]


@pytest.mark.parametrize("tokens,requests", [(2048, 1), (4096, 8), (8192, 64)])
def test_unknown_work_uses_rolling_tactic(tokens, requests):
    assert POLICY.select(tokens, requests, exact_workload=False) == SERIAL


@pytest.mark.parametrize("maximum", (0, 2047))
def test_invalid_calibration_range(maximum):
    with pytest.raises(ValueError):
        PrefillTacticPolicy(2048, 128, max_parallel_tokens=maximum)
