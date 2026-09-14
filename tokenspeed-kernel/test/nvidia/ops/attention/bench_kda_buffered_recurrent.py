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

"""Prepared/native-input recurrence microbenchmark; NOT a serving comparison."""

import argparse
import importlib.metadata
import json
import platform
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.kda._triton.buffered import (
    triton_kda_buffered_recurrent,
    validate_recurrent_blocks,
)
from tokenspeed_kernel.ops.attention.kda._triton.buffered_metadata import (
    commit_positions,
    prepare_positions,
)
from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
    fused_recurrent_kda_pool,
)


def measure(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(32):
            fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(50):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (32 * 50))
    return {"median_us": statistics.median(times), "samples_us": times}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-kind", choices=("prepared", "native"), required=True)
    args = parser.parse_args()
    native = args.input_kind == "native"
    torch.manual_seed(93)
    results = []
    for batch in (1, 8):
        for width in (1, 4):
            heads = 12
            dim = 128
            q = F.normalize(
                torch.randn(batch, width, heads, dim, device="cuda"), dim=-1
            )
            k = F.normalize(torch.randn_like(q), dim=-1)
            v = torch.randn_like(q) * 0.2
            g = -torch.rand_like(q) * 0.1
            d = g.exp()
            beta = torch.rand(batch, width, heads, device="cuda")
            state = torch.zeros(batch, heads, dim, dim, device="cuda")
            indices = torch.arange(batch, dtype=torch.int32, device="cuda")
            alog = torch.zeros(heads, device="cuda")
            bias = torch.zeros(heads * dim, device="cuda") if native else None
            lower_bound = -5.0 if native else None
            if native:
                # Serving split producers hand off packed BF16 conv Q/K/V and
                # raw f_b gate. These views must not need a contiguous FP32 copy.
                packed = torch.stack((q, k, v), dim=2).to(torch.bfloat16)
                q, k, v = packed.unbind(2)
                g = g.to(torch.bfloat16)
                d = g
                beta = beta.logit().to(torch.bfloat16)

            def baseline():
                return fused_recurrent_kda_pool(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    alog,
                    bias,
                    state,
                    indices,
                    indices,
                    scale=dim**-0.5,
                    cu_seqlens=None,
                    lower_bound=lower_bound,
                    use_qk_l2norm_in_kernel=native,
                    use_gate_in_kernel=native,
                    use_beta_sigmoid_in_kernel=native,
                )

            baseline_time = measure(baseline)
            for capacity in (2 * width, 16, 32, 64):
                # Fixed-round replay: end/c do not advance during timing.
                # A separate source/destination state block keeps a flush from
                # changing the next timing sample's checkpoint input.
                rows, start = 8, 8
                for phase, history_len in (
                    ("empty", 0),
                    ("no_flush", capacity - 2 * width),
                    ("flush", capacity - width),
                ):
                    end_value = start + history_len
                    columns = (end_value + width + rows - 1) // rows
                    count = batch * columns + 1
                    history_k = torch.zeros(count, rows, heads, dim, device="cuda")
                    history_u = torch.zeros_like(history_k)
                    history_d = torch.ones_like(history_k)
                    stamps = torch.zeros(count, rows, dtype=torch.int64, device="cuda")
                    table = torch.arange(
                        1, count, dtype=torch.int32, device="cuda"
                    ).view(batch, columns)
                    # State G=1 isolates source c from the flush destination e.
                    state_table = torch.zeros(
                        batch, end_value, dtype=torch.int32, device="cuda"
                    )
                    state_ids = torch.arange(
                        1, batch + 1, dtype=torch.int32, device="cuda"
                    )
                    state_table[:, start - 1] = state_ids
                    if history_len:
                        state_table[:, end_value - 1] = state_ids + batch
                    checkpoint_pool = torch.zeros(
                        1 + 2 * batch, heads, dim, dim, device="cuda"
                    )
                    stamps[
                        table[:, (end_value - 1) // rows].long(), (end_value - 1) % rows
                    ] = (start + 1)
                    endpoint = torch.full(
                        (batch,), end_value, dtype=torch.int32, device="cuda"
                    )
                    valid = torch.full_like(endpoint, width)
                    accepted = torch.full_like(endpoint, width)
                    checkpoint = torch.empty(batch, dtype=torch.int64, device="cuda")
                    length = torch.empty_like(endpoint)
                    flushed = torch.empty(batch, dtype=torch.bool, device="cuda")
                    ok = torch.empty_like(flushed)
                    materialized = torch.zeros_like(flushed)
                    out = torch.empty_like(v)

                    def buffered():
                        prepare_positions(
                            stamps,
                            table,
                            endpoint,
                            valid,
                            checkpoint,
                            length,
                            flushed,
                            ok,
                            capacity=capacity,
                            max_window=width,
                            for_handoff=False,
                        )
                        validate_recurrent_blocks(
                            table,
                            state_table,
                            endpoint,
                            checkpoint,
                            length,
                            valid,
                            flushed,
                            ok,
                            history_blocks=history_k.shape[0],
                            state_blocks=checkpoint_pool.shape[0],
                            history_block_tokens=rows,
                            state_block_tokens=1,
                            capacity=capacity,
                            max_window=width,
                            for_handoff=False,
                        )
                        triton_kda_buffered_recurrent(
                            q,
                            k,
                            v,
                            d,
                            beta,
                            checkpoint_pool,
                            history_k,
                            history_u,
                            history_d,
                            table,
                            state_table,
                            endpoint,
                            checkpoint,
                            length,
                            valid,
                            flushed,
                            ok,
                            out,
                            capacity=capacity,
                            state_block_tokens=1,
                            transform_inputs=native,
                            A_log=alog if native else None,
                            dt_bias=bias,
                            lower_bound=lower_bound,
                        )
                        commit_positions(
                            (stamps,),
                            table,
                            endpoint,
                            valid,
                            accepted,
                            checkpoint,
                            length,
                            flushed,
                            ok,
                            materialized,
                            for_handoff=False,
                        )

                    timing = measure(buffered)
                    if not bool(ok.all()):
                        raise RuntimeError("invalid paged benchmark backing")
                    row = {
                        "batch": batch,
                        "T": width,
                        "L": capacity,
                        "h": history_len,
                        "phase": phase,
                        "heads": heads,
                        "dim": dim,
                        "input_kind": args.input_kind,
                        "baseline": baseline_time,
                        "buffered": timing,
                        "buffered_over_baseline": timing["median_us"]
                        / baseline_time["median_us"],
                    }
                    results.append(row)
                    print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "fixed-round recurrence, full acceptance, single layer/GPU, graph replay; paged path includes position prepare, backing validation, recurrence and stamp commit. Native inputs include in-kernel Q/K normalization and gate/beta transforms on both sides. Endpoints do not advance during timing; not an amortized rollout or serving comparison. Excludes model, conv/gate producers, scheduler, endpoint publication and baseline speculative replay commit",
                "input_kind": args.input_kind,
                "environment": {
                    "python": sys.version,
                    "machine": platform.machine(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "tokenspeed_triton": importlib.metadata.version(
                        "tokenspeed-triton"
                    ),
                },
                "timer": {
                    "calls_per_graph": 32,
                    "graph_replays_per_sample": 50,
                    "samples": 5,
                    "warmup_replays": 5,
                    "seed": 93,
                },
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
