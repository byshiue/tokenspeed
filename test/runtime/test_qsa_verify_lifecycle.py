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

"""Check commit order, live acceptance and failure handling after eager or replay.

Exercise execution modes with all consumers, then each optional consumer alone.
The same runner entry covers width-one targets without a drafter. Fan-out spies
record hook calls; separate checks exercise real consumers with no staged state.
"""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.specific.qsa_indexer import (
    QSAIndexerBackend,
)
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    Qwen4ExpBackend,
)
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple import (
    Qwen4ExpPLEBackend,
)
from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
from tokenspeed.runtime.layers.attention.backends.state.mamba import MambaAttnBackend


def _runner(
    *,
    use_graph: bool,
    has_drafter: bool,
    consumers: tuple[str, ...],
    fail_forward: bool,
):
    events = []
    commits = {"recurrent": [], "ple": [], "qsa": []}
    wrapper = object.__new__(ForwardStepRunner)
    wrapper.config = SimpleNamespace(spec_algo="MTP", max_req_pool_size=8)
    wrapper.device = "cpu"
    full_backend = object.__new__(AttentionBackend)
    full_backend.device = "cpu"

    def commit_recurrent(accepted_lengths):
        events.append("recurrent")
        commits["recurrent"].append(accepted_lengths.tolist())

    def commit_ple(accepted_lengths):
        events.append("ple")
        commits["ple"].append(accepted_lengths.tolist())

    def commit_qsa(accepted_lengths, *, num_extends):
        events.append("qsa")
        commits["qsa"].append((accepted_lengths.tolist(), num_extends))

    def config(is_draft):
        return SimpleNamespace(
            device="cpu",
            dtype=torch.bfloat16,
            is_draft=is_draft,
            speculative_num_draft_tokens=4 if has_drafter else 1,
            component=lambda component_type: SimpleNamespace(
                num_attention_heads=1, num_kv_heads=1, attn_tp_size=1, head_dim=8
            ),
        )

    wrapper.attn_backend = Qwen4ExpBackend(
        config=config(False),
        attention_backend=(
            HybridLinearAttnBackend(
                full_backend,
                SimpleNamespace(commit_verified_state=commit_recurrent),
                [1, 3],
            )
            if "recurrent" in consumers
            else full_backend
        ),
        ple_backend=(
            SimpleNamespace(commit_verified_state=commit_ple)
            if "ple" in consumers
            else None
        ),
        indexer_backend=(
            SimpleNamespace(commit_after_mtp_verify=commit_qsa)
            if "qsa" in consumers
            else None
        ),
    )
    draft_backend = Qwen4ExpBackend(
        config=config(True),
        attention_backend=full_backend,
        ple_backend=None,
        indexer_backend=SimpleNamespace(
            commit_after_mtp_verify=lambda *args, **kwargs: events.append("draft_qsa")
        ),
    )
    wrapper.drafter = (
        SimpleNamespace(attn_backend=draft_backend) if has_drafter else None
    )
    wrapper.max_tokens_per_req = 4 if has_drafter else 1
    wrapper.input_buffers = SimpleNamespace(
        req_pool_indices_buf=torch.tensor([2, 3, 0, 0], dtype=torch.int32),
        seq_lens_buf=torch.tensor([10, 10, 1, 1], dtype=torch.int32),
        state_write_req_pool_indices_buf=torch.zeros(4, dtype=torch.int32),
    )
    wrapper.token_to_kv_pool = SimpleNamespace(
        arena=SimpleNamespace(cache_group_specs=())
    )
    wrapper._can_use_graph = lambda bs, ctx: use_graph
    wrapper._padded_bs = lambda bs, ctx: 4
    wrapper._prepare_decode_metadata = lambda *args, **kwargs: events.append("metadata")
    wrapper._init_forward_metadata = lambda *args, **kwargs: events.append("metadata")
    wrapper._cuda_graph_key = lambda bs: bs
    wrapper._graph_debug = False
    wrapper.deepep_adapter = SimpleNamespace(replay=lambda: None)
    result = (
        torch.arange(16, dtype=torch.int32),
        torch.tensor([3 if has_drafter else 1, 1, 99, 99], dtype=torch.int32),
        None,
    )

    def execute():
        events.append("execute")
        if fail_forward:
            raise RuntimeError("forward failed")
        return result[0][: 2 * wrapper.max_tokens_per_req], result[1][:2], None

    wrapper._forward_func = lambda **kwargs: execute()
    wrapper.graphs = {4: SimpleNamespace(replay=execute)}
    wrapper.output_buffers = {4: result}
    return wrapper, events, commits


def _run(wrapper, mode):
    ctx = SimpleNamespace(
        attn_backend=wrapper.attn_backend,
        bs=2,
        num_extends=1 if mode.is_mixed() else (2 if mode.is_extend() else 0),
        forward_mode=mode,
        global_num_tokens=None,
        all_decode_or_idle=mode.is_decode(),
        capture_hidden_mode=None,
        input_num_tokens=2 * wrapper.max_tokens_per_req,
    )
    empty = torch.empty(0, dtype=torch.int32)
    result = wrapper(
        2,
        ctx,
        None,
        extend_with_prefix=False,
        extend_prefix_lens=empty,
        extend_prefix_lens_cpu=empty,
        extend_seq_lens=empty,
        extend_seq_lens_cpu=empty,
        positions=None,
        block_tables={},
    )
    assert ctx.bs == 2
    return result


@pytest.mark.parametrize(
    "mode,use_graph,has_drafter,consumers,qwen4",
    [
        (ForwardMode.DECODE, False, True, ("recurrent", "ple", "qsa"), True),
        (ForwardMode.DECODE, True, True, ("recurrent", "ple", "qsa"), True),
        (ForwardMode.MIXED, False, True, ("recurrent", "ple", "qsa"), True),
        (ForwardMode.EXTEND, False, True, ("recurrent", "ple", "qsa"), True),
        (ForwardMode.DECODE, False, False, ("recurrent", "ple", "qsa"), True),
        (ForwardMode.DECODE, True, False, ("recurrent", "ple", "qsa"), True),
        (ForwardMode.DECODE, False, False, ("recurrent",), False),
        (ForwardMode.DECODE, True, False, ("recurrent",), False),
        (ForwardMode.MIXED, False, False, ("recurrent", "ple", "qsa"), True),
        (ForwardMode.EXTEND, False, False, ("recurrent", "ple", "qsa"), True),
        (ForwardMode.DECODE, False, True, ("recurrent",), True),
        (ForwardMode.DECODE, True, True, ("ple",), True),
        (ForwardMode.DECODE, True, True, ("qsa",), True),
        (ForwardMode.DECODE, False, True, (), True),
        (ForwardMode.MIXED, False, True, ("ple",), True),
        (ForwardMode.DECODE, False, True, ("recurrent",), False),
        (ForwardMode.DECODE, True, True, ("recurrent",), False),
        (ForwardMode.MIXED, False, True, ("recurrent",), False),
        (ForwardMode.DECODE, False, True, (), False),
    ],
)
def test_runner_commits_live_acceptance_once_after_execution(
    mode, use_graph, has_drafter, consumers, qwen4
):
    wrapper, events, commits = _runner(
        use_graph=use_graph,
        has_drafter=has_drafter,
        consumers=consumers,
        fail_forward=False,
    )
    if not qwen4:
        wrapper.attn_backend = wrapper.attn_backend.attention_backend
        wrapper.config.spec_algo = "DSPARK"
    _run(wrapper, mode)
    accepted = [3, 1] if has_drafter else [1, 1]
    expected_qsa = (
        [(accepted, int(mode.is_mixed()))]
        if "qsa" in consumers and mode in (ForwardMode.DECODE, ForwardMode.MIXED)
        else []
    )
    assert commits["qsa"] == expected_qsa
    assert events[:2] == ["metadata", "execute"]
    assert "draft_qsa" not in events
    verifies_decode = mode.is_decode()
    expected_commits = []
    for consumer in ("recurrent", "ple"):
        active = verifies_decode and consumer in consumers
        assert commits[consumer] == ([accepted] if active else [])
        if active:
            expected_commits.append(consumer)
    if expected_qsa:
        expected_commits.append("qsa")
    assert events[2:] == expected_commits


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("has_drafter", [False, True])
def test_failed_execution_does_not_commit_stale_staging(use_graph, has_drafter):
    wrapper, events, commits = _runner(
        use_graph=use_graph,
        has_drafter=has_drafter,
        consumers=("recurrent", "ple", "qsa"),
        fail_forward=True,
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        _run(wrapper, ForwardMode.DECODE)
    assert events == ["metadata", "execute"]
    assert commits == {"recurrent": [], "ple": [], "qsa": []}


def test_unstaged_consumers_ignore_width_one_commit():
    accepted = torch.ones(2, dtype=torch.int32)
    for cls in (MambaAttnBackend, KdaAttnBackend, Qwen4ExpPLEBackend):
        backend = object.__new__(cls)
        backend._verify_commit_ctx = None
        if cls is KdaAttnBackend:
            # Both the ordinary GDN delegate and the existing KDA replay route
            # must be no-ops when forward already wrote the final state.
            backend._replay_active = False
            backend.commit_verified_state(accepted)
            backend._replay_active = True
        backend.commit_verified_state(accepted)
        assert backend._verify_commit_ctx is None
    qsa = object.__new__(QSAIndexerBackend)
    qsa._verify_state = None
    qsa.commit_after_mtp_verify(accepted, num_extends=0)
    qsa.commit_after_mtp_verify(accepted, num_extends=1)
    assert accepted.tolist() == [1, 1]
