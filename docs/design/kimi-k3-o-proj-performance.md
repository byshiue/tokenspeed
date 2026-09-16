# Kimi-K3 projection redistribution measurements

This note separates standalone projection measurements from full-model workflow
checks. Fused packing leaves NCCL, projection arithmetic, weights and quantization
unchanged. The symmetric-memory prototype is a separate experiment.

## Packing change

The original implementation clears the send buffer and then copies a permuted
input into it. The new kernel writes the channel-sharded layout and padding in
one pass. For one contiguous input token and a one-token communication envelope,
the input already has the required layout: a view replaces both packing launches.
Empty ranks still participate whenever another rank in their subgroup has tokens.
Communication buffers remain persistent and shared across sequential layers.

## Matched standalone results

Hardware: 16 GB300 GPUs, four independent contiguous TP4 groups. The harness
loads real layer 0 (KDA) and layer 3 (MLA) output projections from the NVFP4
checkpoint, not the full model. These particular weights use FP8 E4M3 with
128-by-128 scales; both projections have shape `[7168, 12288]`.

The reference is the projection helper at `944c0081`. Each comparison uses the
same loaded sharded Linear and inputs. Numbers below are complete-operation
CUDA-graph latencies, including packing, all-to-all, GEMM and reduce-scatter.
Each sample has five warmups and 200 timed replays. Five paired samples alternate
reference/optimized order; each sample uses the maximum rank latency. The table
reports the median for each variant and the resulting latency reduction.

| Tokens per rank | KDA old → new (µs) | Reduction | MLA old → new (µs) | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 1, balanced | 45.410 → 40.259 | 11.34% | 48.169 → 41.827 | 13.17% |
| 8, balanced | 51.090 → 50.684 | 0.79% | 52.867 → 49.887 | 5.64% |
| 64, balanced | 66.177 → 62.832 | 5.05% | 66.879 → 63.748 | 4.68% |
| Only global rank 0 has 7 | 53.380 → 49.741 | 6.82% | 52.526 → 49.500 | 5.76% |
| 0/1/2/3 within each group | 51.235 → 48.354 | 5.62% | 51.532 → 48.312 | 6.25% |
| 256, balanced | 133.783 → 124.093 | 7.24% | 131.694 → 123.586 | 6.16% |
| 512/1/1/1 within each group | 197.454 → 178.156 | 9.77% | 197.449 → 178.207 | 9.75% |

The 0.79% result is small relative to run-to-run variation; treat it as
inconclusive, not a reliable gain. The uneven cases still pay for padded traffic.
Packing fusion does not remove that communication cost.

A four-GPU NSYS capture with 20 graph replays per variant confirms the launch
change: the reference has one fill and one copy kernel per projection; the
optimized path has one packing kernel. For eight tokens, their median kernel
durations were 0.768 + 2.144 µs versus 0.864 µs. GEMM and NCCL collectives remain
present. Profiled collective times include startup skew and are not used for
the performance table.

All tested optimized outputs matched the original TP4 helper bit for bit.
The distributed harness also checked empty groups and graph replay with changing
input values and valid token counts. Fourteen packing-kernel cases passed,
including strided inputs, zero padding, one-token aliasing and graph replay.
This is projection-level validation, not full-model accuracy validation.

## Reproducing and retaining evidence

Export the original helper before running the current harness:

```bash
git show 944c0081:python/tokenspeed/runtime/models/kimi_k3_o_proj.py > /tmp/reference_o_proj.py
```

Launch one process per GPU through the deployment runbook's persistent allocation
and container environment. On each of four nodes, use its corresponding node rank:

```bash
python -m torch.distributed.run \
  --nnodes=4 --nproc-per-node=4 --node-rank=<node-rank> \
  --master-addr=<master-host> --master-port=<port> \
  test/runtime/distributed/test_kimi_k3_o_proj.py \
  --model <real-model-path> --benchmark --iterations 200 --repeats 5 \
  --reference-module /tmp/reference_o_proj.py
```

Preserve the JSON lines containing `matched_graph_us`, not just the summary.
Compute `100 * (1 - median(optimized) / median(reference))` for latency reduction.
Retain the tested diff, environment versions and raw logs alongside the results.
The initial run's isolated component timings used the old packing callback;
only its complete-operation matched timings support the table above. The harness
now measures the production packing callback for subsequent component runs.

## Full-model packing comparison

A full Kimi-K3 NVFP4 test used the same sixteen GB300 GPUs for all variants:
attention TP1/DP16, MoE EP16 with FlashInfer transport, CUDA graphs and no
speculative decoding. The production TP4 variants both retained NCCL. The
symmetric-memory prototype was not included.

Five warm rounds per workload generated 32 tokens per request. Concurrent rows
are complete client batch wall times. Each measured suite followed at least one
full workload pass and an explicit prefix-cache flush, then reused prefixes
within the suite. Baseline additionally had a partial pass during client recovery.

| Workload | Unsharded TP1 (s) | Original TP4 (s) | Packing TP4 (s) | Packing latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Single request | 1.001151 | 1.139827 | 1.126397 | 1.18% |
| Balanced, 16 ranks | 1.253882 | 1.433128 | 1.416801 | 1.14% |
| Uneven, 3 ranks | 1.270039 | 1.434263 | 1.398540 | 2.49% |

These are small observed median reductions, not robust speedup guarantees.
Optimized balanced samples ranged from 1.411627 to 1.641366 seconds. Batch wall
time includes client dispatch and artifact writes; all samples were retained.
The test used one measured server launch per variant, in fixed order, and direct
gRPC rank affinity rather than the HTTP gateway. It is not a saturation benchmark
or a client-side streaming ITL measurement. Single-request server decode rate
increased from 39.1 to 39.6 tokens/s; TTFT changed from 344.34 to 340.83 ms.

All 122 original-versus-optimized TP4 requests matched input tokens, cache counts,
generated tokens and returned token log probabilities exactly. TP1 versus TP4
matched only 97/122 generated sequences; sharding's numerical acceptance remains
open. This is not a dataset-level accuracy evaluation. Optimized TP4 was still
10–13% slower than unsharded TP1 in these workflows: packing alone did not offset
the redistribution overhead. Different generated tokens also limit how precisely
the TP1/TP4 timing difference can be attributed to projection implementation.

## Symmetric-memory experiment

A standalone prototype reuses the harness's real weight loading, sharded Linear
and timing helpers. It writes packed input directly into peers' persistent
symmetric receive buffers. After GEMM, each rank reads peer partials, sums in
FP32 and writes its own complete output. GPU barriers publish writes and prevent
the next call from overwriting buffers before peer readers finish. NCCL remains
the production backend; this prototype is not an automatic runtime selection.

The following comparison uses the same 16-GPU setup and five paired samples of
200 graph replays. Here the reference already includes fused packing. Do not add
these percentages to the earlier table: these are separate paired experiments.

| Tokens per rank | KDA NCCL → prototype (µs) | Reduction | MLA NCCL → prototype (µs) | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 1, balanced | 40.158 → 34.709 | 13.57% | 40.380 → 34.944 | 13.46% |
| 8, balanced | 49.997 → 35.017 | 29.96% | 49.885 → 34.995 | 29.85% |
| 64, balanced | 64.850 → 43.232 | 33.34% | 64.312 → 43.409 | 32.50% |
| Only global rank 0 has 7 | 49.475 → 34.946 | 29.37% | 48.736 → 34.921 | 28.35% |
| 0/1/2/3 within each group | 48.682 → 34.967 | 28.17% | 49.395 → 34.964 | 29.21% |
| 256, balanced | 123.080 → 90.347 | 26.59% | 123.278 → 88.977 | 27.82% |
| 512/1/1/1 within each group | 178.151 → 152.648 | 14.32% | 178.312 → 152.558 | 14.44% |

The prototype completed four- and sixteen-GPU tests, including empty groups and
100 graph replays with changing input values per active test case. Its FP32 sum
changes rounding: relative L2 difference from NCCL was approximately 0.3%, not
bitwise equality. Relative L2 error against a dequantized FP32 matmul reference
was slightly lower than NCCL in the measured sixteen-GPU cases. This does not
establish equivalence for arbitrary inputs or full-model generation.

At a 512-token capacity, the prototype explicitly reserves 47 MiB per GPU for
receive, partial-result and owner-output buffers, versus 24 MiB for NCCL send/receive
scratch. These are calculated workspace sizes, not measured peak process memory;
GEMM outputs, graph pools, pointer tables and signal pads are additional. The
benchmark retains both variants' workspaces for comparison. Reducing launch
latency therefore does not imply a workspace-memory saving.

Before promoting this experiment, implement a supported kernel API and topology
checks, validate repeated transitions between message sizes and workspaces, and
define acceptable reduction numerics. Measure larger messages and other subgroup
placements before choosing a byte-size threshold. Retain NCCL for unsupported
topologies and workloads. The current evidence supports further development, not
enabling a new backend by default.

## Optional Triton RSAG reduction

**Measurement caveat:** the historical results in this section time one
projection per graph and include inter-replay submission gaps. They are not
isolated collective timings. The harness now captures up to 20 operations
per graph, warms the captured graph, and divides CUDA-event time by the
actual operation count. Compare backends with this method before selecting
one; do not combine gains measured with different replay protocols.

The integrated comparison holds FlashInfer NVLink A2A and the quantized GEMM
fixed, replacing only NCCL ReduceScatter with existing Triton RSAG. RSAG
includes its input staging copy and returns a cloned output; these timings
do not rely on exposing a reusable communication-buffer view.

With 16 GB300 GPUs arranged as four TP4 groups, 16 rows per rank, real FP8
projection weights from an NVFP4 checkpoint, and CUDA graphs, six alternating
rounds of 500 iterations gave these median complete-projection latencies:

| Projection | NCCL reduction | RSAG reduction | Latency reduction |
|---|---:|---:|---:|
| KDA | 51.87 µs | 43.49 µs | 16.15% |
| MLA | 51.90 µs | 43.89 µs | 15.43% |

Each round reports the maximum rank time. These are standalone projection
results, not full-model decode latency or throughput. Do not multiply gains
from separate A2A and reduction experiments to claim an end-to-end speedup.

An initial wider policy exposed regressions: combining NCCL A2A with RSAG
was about 14–17% slower at 17–64 rows per rank in this run. Some uneven
batches also slowed down. The retained policy uses RSAG only when fused
FlashInfer A2A is active, with balanced physical batches of 1–16 rows per
rank. Larger and uneven cases retain NCCL reduction; paired fallback results
were within roughly 0–2.5% of the reference.

RSAG uses FP32 accumulation for its BF16 multimem reduction. Relative L2
difference from NCCL was about 0.36%; outputs are not bitwise interchangeable.
The test harness checks reduction error against an FP32 sum of the same GEMM
partials separately from projection quantization error. It also checks exact
eager/graph replay agreement and preservation of outputs across workspace
reuse. These checks do not replace full-model logits or generation validation,
so the backend remains opt-in.

## Copy-free FlashInfer A2A and peer reduction

The integrated `triton_peer` candidate removes two intermediate copies:
FlashInfer's receive-buffer copy and the copy from GEMM output into symmetric
reduction storage. The prepared FP8 GEMM writes into that storage directly.
The reduction retains its publication/reuse barriers and returns owned output.

The comparison below uses 16 GB300 GPUs, four TP4 groups, BF16 activations,
and real block-scaled FP8 KDA/MLA projection weights from an NVFP4 checkpoint.
Both matrices have shape [7168, 12288]. The reference uses FlashInfer A2A,
the same quantized GEMM, and NCCL reduction. Each sample captures 20 complete
projections per graph, warms the graph, and times 500 operations with CUDA
events. Values are medians of six alternating samples, each taking the
maximum across all 16 ranks. These are not full-model measurements.

| Projection | Rows/rank | Reference | Copy-free peer | Latency reduction |
|---|---:|---:|---:|---:|
| KDA | 1 | 34.66 µs | 30.68 µs | 11.5% |
| KDA | 8 | 36.85 µs | 31.99 µs | 13.2% |
| KDA | 16 | 39.11 µs | 35.19 µs | 10.0% |
| MLA | 1 | 34.65 µs | 30.77 µs | 11.2% |
| MLA | 8 | 37.01 µs | 32.05 µs | 13.4% |
| MLA | 16 | 39.24 µs | 35.25 µs | 10.2% |

Larger and uneven physical batches retained NCCL. Their paired latencies
were within 0.8% of the reference in this run. Fast-path numerical checks,
eager/graph equivalence, changing valid rows, inactive groups and output
lifetime checks passed. Across all tested shapes, relative L2 difference
from TP1 was below 0.7%; this includes quantized sharding effects and is not
a full-model accuracy result. At 16 rows/rank, independent TP1 projection
was still faster than the optimized TP4 operation. Communication optimization
does not establish that projection sharding is a latency win.

A short production-helper NSYS capture used five replays of 20 KDA
projections for each path. On each GPU of the inspected TP4 group, the
reference had 100 receive copies of 393216 bytes. The optimized path had
zero memcpy events and no staging-copy kernel: each projection ran A2A,
activation quantization, GEMM, two barriers, and owner reduction.

### Other candidates and limits

The earlier 41–44 µs comparison used one projection per graph. Capturing 20
operations reduced the apparent custom-reduction advantage to about 1.4–1.6%
before copy removal. NSYS showed that custom reduction kernels themselves
were not faster than NCCL; complete-path gaps and copies matter.

NCCL's default selected LL. Forcing LL did not help consistently; LL128 and
Simple were slower. FlashInfer fused UC/MC could not be measured because
communicator initialization failed, so no speedup or slowdown is assigned
to them.

Three experimental single-kernel synchronization/reduction variants passed
numerical checks but increased complete latency to roughly 57–62 µs, versus
about 35 µs for the copy-free two-barrier path. Remote atomic polling,
acquire-load polling, and padded signal locations did not improve this
workload. They are not integrated. Fewer kernel launches alone are not a
reason to select a backend.
