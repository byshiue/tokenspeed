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
