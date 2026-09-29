# Kimi-K3 O-projection weight prefetch

## Runtime integration

Kimi-K3 can select compute TP or full-weight prefetch from the local
O-projection GEMM's physical row count, M:

```bash
export TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE=1
export TOKENSPEED_KIMI_K3_O_PROJ_WEIGHT_TP_SIZE=4
```

| Local physical M | O-projection execution |
| --- | --- |
| M ≤ 64 | TP4 compute: activation A2A → quantization/sharded GEMM → ReduceScatter |
| M > 64 | N-weight TP4: prefetch full weights beside attention → original local GEMM |

The cutoff is strict: M=64 uses compute TP; M=65 uses weight prefetch.
It applies to both decode and prefill, including CUDA-graph padding, not to
global concurrency or cached context length. At one token per request without
padding, M equals requests per rank. Prefill uses the current chunk's token rows.

Unset the weight-TP variable, or set it to `1`, to retain the original path.
The weight-TP setting already enables the small-M compute route; do not also
set `TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE` above `1`. QKV and
shared-expert TP remain independent. See the [TP-sharding recipe](kimi-k3-tp-sharding.md)
for DEP16 launch settings, the shared environment and persistent allocation.

The runtime loads two layouts of each KDA/MLA O-projection: an output-channel
(N) shard for prefetch and an input-channel (K) shard for compute TP, each with
its FP32 128×128 scales. Both copy checkpoint codes verbatim, without
requantization or a retained full matrix. These attention weights are FP8 even
in the NVFP4 checkpoint. TP must divide world size and preserve whole K scale
blocks. N shards may have unequal valid sizes, with zero-padded trailing blocks.

For M > 64, an auxiliary stream pulls immutable peer weights during attention through
CUDA copy engines directly into the full contiguous `[N,K]` GEMM buffer.
Each layer's full canonical scales and prepared GEMM layout are gathered and
cached once at initialization, before memory profiling and graph capture.
Inference does not gather scales or launch a scale-layout kernel. After attention
and weight prefetch complete, the original local full-width GEMM runs. There is
no weight restoration kernel, activation A2A or output reduction for that
owner's large-M output. Copy order remains peer shards first, local shard last.
Activation quantization is unchanged.

For M ≤ 64, the existing `DPRowParallelLinear` performs compute TP without
refreshing the full-weight buffer. In mixed subgroups, only small owners send
tokens to this route, but every peer supplies its K-sharded partial results.
Large and empty owners therefore participate when another owner needs compute
TP. Only a subgroup with no small-owner tokens skips these collectives.
Immutable N shards remain readable by large owners independently. The same
row-count rule applies in eager execution and graphs; no scheduler or
cache-layout changes are needed. Projection backend settings are unchanged.

The feature currently requires Blackwell, the prepared FlashInfer block-FP8
backend (`dense_gemm_backend=auto`) and peer-accessible symmetric memory.
Unsupported precision or topology fails at startup. Weight updates during
serving are not supported.

## Lifetime and memory

Each layer publishes its immutable N shard once and retains its K shard locally.
All sequential KDA/MLA layers
share one full-weight buffer, sized for the largest projection. Scale layouts
and GEMM plans are keyed by layer, not shape: same-shaped layers have different
scale values, which must remain valid across eager calls and graph replays.
The prefetch stream waits for the preceding consumer before reusing
scratch, and GEMM waits for the current prefetch. Buffers have stable addresses
before capture; finishing GEMM releases their logical lifetime rather than
allocating or freeing GPU memory every token. Peer mappings remain alive
until every rank has finished all graph replays.

The attention wrapper keeps the prefetch and stream join inside the existing
graph boundary. Decode captures both streams; eligible KDA prefill stays
captured; an eager attention break includes the prefetch and join together.

For KDA `[N,K]=[7168,12288]`, replicated FP8 weights/scales occupy about
84.02 MiB per layer. Each TP4 layout uses about 21.01 MiB per layer per GPU;
retaining both costs **42.02 MiB per layer**, not 21.01 MiB. This is roughly
half the replicated weight/scale storage, rather than one quarter. There is
also about 84 MiB of full-weight scratch **per model**, not per layer, plus
shared compute-TP communication scratch capped at 64 rows per owner.
Both cached full scale layouts occupy
42 KiB per KDA layer per GPU and replace the former shared scale scratch;
neither full scale layout is refreshed during inference. The compute GEMM
retains its ordinary prepared shard-scale plan as well. Direct N-layout copies
avoid the former K-sharded prefetch path's additional 84 MiB transfer buffer.
Count scale plans, symmetric allocation granularity and graph reservations
separately when reporting total memory or cache capacity. TP16 requires
N-block padding, so its per-rank storage is not exactly one sixteenth.

## Correctness validation

```bash
python -m torch.distributed.run --nnodes=NODE_COUNT --nproc-per-node=4 \
  --node-rank=NODE_RANK --master-addr=HEAD_NODE --master-port=PORT \
  -m test.runtime.distributed.validate_kimi_k3_weight_prefetch \
  --model MODEL_DIR --tp-size 4
```

Repeat with `--tp-size 16` to cover partially filled and empty N shards.
The test uses the production Kimi loader, startup and two real KDA modules.
It checks bit-exact loaded K shards and reconstructed N-sharded weights/scales.
Large-M outputs must match replicated FP8 exactly; small-M outputs must match
the existing compute-TP route exactly, with its existing accumulation tolerance
against replicated FP8 checked through TP4. TP16 uses exact compute-TP16
equivalence: the unchanged NCCL compute path can exceed the TP4 1.5% relative-L2
bound at M=1. This is not a wider tolerance for the hybrid. Cases include
M=63/64/65, graph padding, mixed
`[0,64,65,128]` owners, changing activations, poisoned weight scratch,
per-layer scale isolation, delayed peers and all-idle subgroups. Small owners
must leave poisoned full-weight scratch untouched. Incremental prefill tests
the same boundary through both eager-attention breaks and inline capture.
These checks establish attention-module equivalence, not dataset accuracy.

## Hybrid component check

The M > 64 hybrid was checked on 16 GB300 GPUs with real checkpoint weights,
CUDA graphs, QKV TP1 and fixed decode metadata. The complete production KDA
module includes input projections, attention and O projection, but no MoE or
AttnRes. Ten alternating-order samples each time 15 graph replays of 20
forwards, using CUDA events without NSYS and the slowest rank per sample.

| M/rank | Replicated TP1 | Compute TP4 | Always-prefetch N TP4 | Hybrid TP4 |
| --- | ---: | ---: | ---: | ---: |
| 32 | 204.85 us | 209.39 us | 277.74 us | 210.03 us |
| 64 | 310.87 us | 310.81 us | 318.18 us | 312.03 us |
| 65 | 316.05 us | 321.60 us | 323.51 us | 323.66 us |
| 128 | 505.42 us | 524.74 us | 511.40 us | 511.17 us |

The hybrid avoids small-M prefetch cost and keeps the large-M copy schedule.
At M64 it is 1.93% faster than always-prefetch and 0.39% slower than standalone
compute TP4. M65 demonstrates routing, not a measured performance crossover:
compute TP4 is still faster there. The cutoff is the selected policy, not an
autotuned threshold. These component timings do not establish full-model
performance or cache capacity with the additional K-sharded weights.

## Earlier C64 component comparison: cached scales

These results predate the M > 64 cutoff: the weight-TP4 column forced prefetch
at C64. The current hybrid instead selects compute TP there. The cache change
removes the per-forward scale-gather kernel, but the C64 test
does not show a latency gain. The production KDA module still includes input
projections, recurrent attention and O projection; this is not a GEMM-only or
full-model measurement.

The comparison uses 16 GB300 GPUs, 64 rows per rank, real FP8 attention weights
from the NVFP4 checkpoint, and CUDA graphs without a profiler. QKV stays TP1.
Compute TP4 uses TokenSpeed Lamport A2A and TRT-LLM Lamport ReduceScatter.
The pre-cache implementation and cached implementation run alternately twice;
each run rotates the three variants over nine samples. Each sample times 15
graph replays of 20 forwards and takes the slowest rank. Medians below include
all 18 samples per implementation and variant.

| Complete KDA module | Before scale caching | After scale caching | Change |
| --- | ---: | ---: | ---: |
| DEP16 / TP1 | 310.94 us | 311.06 us | +0.04% |
| DEP16 / compute TP4 | 312.65 us | 313.10 us | +0.15% |
| DEP16 / N-weight TP4 | 318.87 us | 319.18 us | +0.10% |

The weight-TP4 increase is small and comparable to movement in the unchanged
controls; do not claim a speedup. Caching saves repeated scale work, not the
bulk weight transfers or their competition with attention for memory bandwidth.
TP4 and TP16 correctness checks pass bit-exactly against replicated FP8,
including per-layer scale isolation, padded/empty shards, changing inputs,
poisoned weight scratch, eager/graph decode and incremental prefill. Full-model
E2E performance has not been rerun for this scale-cache change.

## Earlier 48-layer E2E comparison

These measurements precede both initialization-time scale caching and the
M > 64 hybrid. They do not validate the new hybrid's full-model latency or memory.

The paired comparison uses real checkpoint weights, the first 48 text layers,
16 GB300 GPUs, DEP16 attention/cache ownership, EP16 MoE and O weight TP4.
QKV/shared-expert TP and speculation are disabled; CUDA graphs are enabled.
C128 means 128 requests per rank, or 2,048 total. Both variants use the same
hardware, serving settings and fixed-affinity inputs, without NSYS attached.

Each round flushes cache, sends 1,024-token cold prompts, then sends 1,280-token
prompts reusing the first 1,024 tokens. Cold generation uses 128 output tokens;
incremental generation uses 512. One full warmup round precedes three measured
rounds. Report serving completion time, TTFT and stream-observed decode intervals
separately; none is a projection-only or full-depth model measurement.
Medians of the three measured rounds:

| Metric | Former K-sharded hybrid | N-sharded weight prefetch | N change |
| --- | ---: | ---: | ---: |
| Steady decode after cold prefill | 44.409 ms/token | 43.119 ms/token | -2.91% |
| Steady decode after incremental prefill | 44.461 ms/token | 43.199 ms/token | -2.84% |
| Cold request completion | 18.372 s | 17.841 s | -2.89% |
| Incremental request completion | 27.262 s | 26.526 s | -2.70% |
| Cold TTFT | 6.088 s | 5.673 s | -6.82% |
| Incremental TTFT | 1.877 s | 2.017 s | +7.50% |
| Incremental output throughput | 38,463 token/s | 39,530 token/s | +2.78% |

The steady metric is the median per-request average streamed-token interval
while all 2,048 requests are active, excluding 250 ms at each end. It is not a
CUDA-event measurement. Incremental steady-decode samples span 44.395–44.577 ms
for K versus 43.149–43.206 ms for N. This consistent large-M gain is the reason
to keep N-sharding and remove the K-sharded hybrid. Request completion includes
smaller startup/draining batches, where the former hybrid used activation TP.
TTFT varies more with admission and prefill overlap; these results do not show
an incremental-TTFT improvement.

TP4 and TP16 GPU validation passed, including padded N shards, and the existing
Kimi configuration/MoE suite passed 640 tests plus six subtests. E2E cold and
incremental workflows completed with finite smoke-test logprobs. Across the
three measured rounds, 5,782/6,144 cold sequences and 5,779/6,144 incremental
sequences matched between variants. Repeated runs of the same implementation
also differ, and the former hybrid changes FP8 accumulation at small M.
Serving token identity is therefore not the correctness gate here: the direct
KDA tests require exact agreement with replicated FP8. No full-depth or dataset
accuracy claim follows from this reduced-layer performance experiment.

## Earlier always-prefetch runtime measurements (N-sharded weights)

These measurements precede the former M=128 hybrid and the current return to
N-sharding. They are historical results from different workloads, not substitutes
for the paired 48-layer comparison above.

### Integrated KDA measurements

The runtime component comparison below calls `KimiLinearKDA.forward` with real
checkpoint weights. It includes input projections, recurrent attention and the
O projection, but not AttnRes, MoE, scheduling or the rest of the model.
Activations are seeded; state reads and writes use separate pages. All 16 GB300
GPUs participate, with four independent groups for TP4.

| Requests/rank | Replicated | Compute TP4 | Weight-only TP4 | Weight-only TP16 |
| --- | ---: | ---: | ---: | ---: |
| 32 | 204.85 µs | 209.98 µs | 262.80 µs | 363.13 µs |
| 64 | 311.36 µs | 312.54 µs | 318.92 µs | 368.74 µs |
| 128 | 505.22 µs | 526.83 µs | 512.69 µs | 512.68 µs |

These are unprofiled CUDA-event measurements: nine alternating-order samples,
15 replays per sample, 20 forwards per graph, taking the slowest rank and then
the sample median. TP16 was measured in a separate paired run; its replicated
medians were 205.56/310.52/505.23 µs. The environment used PyTorch 2.14.0,
CUDA 13.0, NCCL 2.30.7 and FlashInfer 0.7.0. Compute TP4 used TokenSpeed Lamport
A2A and TRT-LLM Lamport ReduceScatter; weight-only TP used copy-engine reads.

At C128, either weight-only size is about 1.5% slower than replicated weights
while reducing persistent weight storage. Shorter attention windows expose more
transfer cost, especially with TP16. These measurements do not establish a
full-model speedup or include full-model memory usage.

A short C128 trace of the integrated module confirms that weight transfers
overlap attention. On one four-GPU subgroup, excluding the first two of twenty
forwards, the median copy span was 386.61 µs, with 386.29 µs overlapping
attention and 0.14 µs left after attention. The attention kernel itself changed
from 395.52 to 399.58 µs. These profiled diagnostics explain the overlap; they
are not substitutes for the unprofiled timings above.

### Full-model validation

The serving comparison uses the full 93-layer text model from the real NVFP4
checkpoint on the same 16 GB300s. Attention/cache ownership stays TP1/DP16,
MoE stays TP1/EP16, and QKV/shared-expert TP and speculation are disabled.
Both variants use BF16 activations, FP8 KV cache (the same default scale of
1.0), FlashInfer MoE transport, CUDA graphs at 1/2/4/8/16 requests per rank,
256 global request slots, maximum sequence length 4096, a 1024-token prefill
chunk limit and memory utilization 0.83. Only the weight-TP setting changes.

Each workload has one warmup and five measured rounds. Cold requests have
1024 input tokens and generate 32 tokens. Incremental requests reuse the same
1024 cached tokens, append 256 new tokens and generate another 32. Requests
keep their rank affinity, with a 10 ms launch interval between slots. Serving
wall time includes scheduling and transport; it is not a pure GPU decode
measurement. Full-model C128 was not tested here.

Median workload completion time (prefill plus 32-token generation):

| Load | Phase | DEP16 | Weight-only TP4 | Latency increase |
| --- | --- | ---: | ---: | ---: |
| One request total | Cold | 1.073 s | 1.458 s | 35.9% |
| One request total | Incremental | 1.102 s | 1.512 s | 37.1% |
| C1/rank | Cold | 1.353 s | 1.764 s | 30.4% |
| C1/rank | Incremental | 1.381 s | 1.799 s | 30.3% |
| C16/rank | Cold | 5.581 s | 6.118 s | 9.6% |
| C16/rank | Incremental | 2.686 s | 3.046 s | 13.4% |
| Uneven, 32 requests total | Cold | 5.273 s | 5.895 s | 11.8% |
| Uneven, 32 requests total | Incremental | 2.587 s | 2.997 s | 15.8% |

The uneven case sends 16 requests to one rank and eight each to two other
ranks; the remaining ranks are idle. All measured rounds are retained in the
median, including one 3.169 s candidate C1/rank cold round. At C16/rank, cold
TTFT changes from 2.414 to 2.585 s and observed time per output token from
98.31 to 110.48 ms; incremental TTFT changes from 0.979 to 1.063 s and time
per output token from 51.03 to 59.94 ms. These serving metrics can include
overlapping prefills, so they do not isolate a steady-state decode step.

All 3,050 measured output sequences, comprising 97,600 generated token IDs,
match the replicated baseline. Eighty fixed-context next-token probes, each
repeated twice, match in both token and logprob exactly. During the concurrent
workflow, generated-token logprobs differ by 1.46e-5 on average and 0.007064
at most. This validates numerical/workflow equivalence, not AIME or another
dataset accuracy score.

Per-GPU memory after removing unused activation-collective initialization:

| Metric | DEP16 | Weight-only TP4 |
| --- | ---: | ---: |
| Attention weight parameters | 33.85 GB | 28.13 GB |
| Free device memory immediately after weight loading | 109.89 GB | 115.24 GB |
| KV-cache allocation | 62.63 GB | 67.47 GB |
| Usable cache pages, excluding null page | 426 | 459 |

The 5.72 GB parameter reduction becomes 4.84 GB more cache, or about 7.7% more
usable pages, after scratch, symmetric allocation and communication overhead.
Final free memory stays near 46 GB because the cache planner spends the saved
budget. Symmetric allocations move bytes outside PyTorch's allocator, so
parameter bytes or `torch.cuda.memory_allocated()` alone are not total device
memory measurements. The table uses the runtime logger's GB units.

Weight-only TP therefore remains an opt-in memory tradeoff, not a recommended
latency optimization for these full-model loads. The nearly hidden C128 KDA
transfer does not imply that smaller KDA batches or MLA layers have enough
attention work to hide the same full-weight transfer.

## Earlier component experiments

The original component benchmark compares four ways to execute Kimi-K3's O
projection under DEP16. The experiments below predate the runtime integration;
their implementation details and measurements are historical, not current
serving results. The standalone scripts do not load the full model.
Use the [TP-sharding recipe](kimi-k3-tp-sharding.md) for the shared environment,
persistent allocation and existing runtime paths.

The compute-TP4 reference uses `DPRowParallelLinear` with a preallocated
`DPRowParallelCommunication`; weight-only variants still use the full GEMM.
The recorded measurements below predate the projection API rebase onto
`4d4c0357`. Rerun the paired benchmarks before treating them as performance
results for the rebased code.

| Variant | Persistent weight placement | Work done for each forward |
| --- | --- | --- |
| DEP16 | Full weight on every rank | Local full-width GEMM |
| DEP16 + TP4 compute | One input-channel shard per rank | Activation A2A → sharded GEMM → ReduceScatter |
| DEP16 + TP4-weight | One input-channel shard per four-rank subgroup member | Gather weights/scales → restore full layout → local full-width GEMM |
| DEP16 + TP16-weight | One input-channel shard per world rank | Gather weights/scales → restore full layout → local full-width GEMM |

The two weight-only variants gather on an auxiliary stream while the production
KDA decode kernel runs on the main stream. GEMM waits for both. The next gather
waits for the previous GEMM before overwriting scratch. No activation A2A or
output reduction is needed in those variants.

## Precision and memory

The real NVFP4 checkpoint's KDA O projection is FP8, not FP4:
`[N, K] = [7168, 12288]`, with FP32 scales for 128×128 blocks. Each rank loads
only its persistent input-channel shard in the weight-only variants. FP8 codes
and preformatted scales share one byte payload; NCCL AllGather moves it without
requantization. A restoration kernel writes the ordinary full GEMM layout.
The activation quantizer and GEMM match the replicated production path.

Weights and scales occupy 84.02 MiB per layer. Persistent storage becomes
21.01 MiB with TP4-weight or 5.25 MiB with TP16-weight. That early implementation
also needs 168.04 MiB of reusable scratch per rank: one gathered payload and
one restored weight/scale pair. That scratch must be shared by sequential
layers in any later model integration, not allocated once per layer.

CUDA graphs retain stable buffer addresses. Finishing GEMM releases the
weight's logical lifetime for reuse; it does not call `cudaFree` every step.
Count scratch, communicator allocations, allocator reservations and cache
capacity separately before claiming a model-memory improvement. The paired
benchmark keeps all variants resident and reports logical tensor bytes, not a
full-model memory measurement.

## Run

Use a fresh persistent 16-GPU allocation, one compatible NVLink fabric, the
existing shared environment, and four workers on each of four nodes:

```bash
python -m torch.distributed.run --nnodes=4 --nproc-per-node=4 \
  --node-rank=NODE_RANK --master-addr=HEAD_NODE --master-port=PORT \
  -m test.runtime.distributed.benchmark_kimi_k3_o_proj_weight_prefetch \
  --model MODEL_DIR --rows 32 64 128 \
  --steps 10 --samples 5 --replays 5 --output RESULTS_JSON
```

C32/C64/C128 mean requests **per rank**: 512/1024/2048 globally. Baseline and
weight-only GEMMs have `[M, N, K] = [C, 7168, 12288]`; compute-TP4 uses
`[4C, 7168, 3072]`. Its reference communication is TokenSpeed Lamport A2A
and TRT-LLM Lamport ReduceScatter. Weight-only gather uses NCCL.

Before timing, the harness checks exact FP8/scale reconstruction, weight-only
outputs against DEP16, and compute-TP4 error. Graph replay changes inputs and
overwrites gathered weights with invalid contents to detect missing refreshes.
KDA uses real convolution, decay and normalization weights with seeded
activations/states. Separate read/write state pages keep repeated measurements
deterministic while retaining state traffic.

After warmup, each sample measures graph replays with CUDA events, takes the
slowest rank and alternates variant order. Results include standalone projection,
gather, layout restoration, GEMM, attention and attention→projection latency.
Use the last measurement for overlap comparisons; subtracting or adding
standalone kernel times misses contention and cache effects.

Add `--profile` inside an NSYS CUDA-profiler-API capture to inspect ten steps
of each complete path. Cross-rank profiler-start skew can inflate early
collectives. Use unprofiled samples for performance and later trace steps to
check whether gather actually overlaps attention.

An auxiliary stream alone does not guarantee overlap. Inspect dependencies,
CTA register/shared-memory footprints and bandwidth contention. Any controlled
NCCL launch-tuning run must be labeled separately from the default comparison.
This experiment establishes neither full-model latency nor dataset accuracy.

## Copy-engine overlap experiment

To compare high-priority NCCL weight gathering with peer reads through CUDA
copy engines, use the same allocation, checkpoint and shared environment:

```bash
python -m torch.distributed.run --nnodes=4 --nproc-per-node=4 \
  --node-rank=NODE_RANK --master-addr=HEAD_NODE --master-port=PORT \
  -m test.runtime.distributed.benchmark_kimi_k3_weight_copy_engine \
  --model MODEL_DIR --rows 32 64 128 --tp-size 4 \
  --steps 20 --samples 7 --replays 5 --output RESULTS_JSON
```

Repeat with `--tp-size 16 --rows 128` to test one group across the NVLink fabric.
This requires peer-accessible symmetric memory across the selected ranks; the
benchmark fails if mapping is unsupported rather than falling back silently.
It changes only the benchmark, not model loading or serving defaults.

By default, all variants keep the original layout-restoration kernel and
full-width GEMM.
NCCL uses a high-priority process-group stream, and all prefetch streams have
high priority. The baseline is a replicated local projection on each rank.
Two copy-engine variants separate the synchronization cost:

- `ce_barrier` uses symmetric-memory barriers before and after peer reads.
  Data moves through copy engines, but the barriers themselves use SM kernels.
- `ce_immutable` publishes each weight shard once, then reads it without
  per-forward cross-rank barriers. The shard must remain unchanged and mapped
  until **all** ranks finish. This is valid for persistent inference weights,
  not arbitrary activations, training updates or a reused weight-staging buffer.

Both variants retain local stream waits: full-weight scratch cannot be
overwritten before its previous GEMM finishes, and GEMM cannot read it before
gather/restoration completes. The immutable path still fetches all shards on
every call; it does not cache the complete restored weight between calls.

Correctness checks require bit-exact reconstructed FP8 bytes/scales and outputs,
including changed activations, poisoned scratch and delayed readers. Inputs,
peer mappings and output buffers are prepared before graph capture. Logical
weight/scratch accounting is unchanged; symmetric-memory allocation granularity
and mapping metadata add overhead not covered by that accounting.

Add `--profile` for the short NSYS capture. Verify actual GPU memcpy events;
`non_blocking=True` by itself is not evidence of a copy-engine implementation.
Report attention duration and the complete attention-to-projection latency
separately. Copy engines remove transfer CTAs, but still consume memory
bandwidth, and the restoration kernel still uses SMs. Repeat a fixed graph
length: different capture layouts can change overlap and the measured gain.

### Direct-layout copy-engine experiment

Add `--direct-layout` to include `ce_direct` in the same paired benchmark.
It uses `cudaMemcpy2DAsync` to place each peer's FP8 input-channel shard into
the full row-major GEMM weight, then copies that peer's prepared scales into
their final contiguous slice. It uses the CUDA runtime already loaded by
PyTorch; no additional CUDA installation or environment is needed.

This removes the rank-major temporary and the restoration kernel. Logical
scratch falls from 168.04 MiB to 84.02 MiB per rank; persistent shard size is
unchanged. Allocator reservations and symmetric-memory metadata are not included.
The benchmark still holds several variants at once, so its process memory is
not a prediction of model-serving memory.

The direct variant has the same immutable-source contract and local stream
dependencies as `ce_immutable`. It must pass exact byte/scale reconstruction,
eager output checks and changing-input graph replays with poisoned scratch and
delayed peers. Profile it to confirm actual copy-engine transfers and absence
of restoration kernels. Removing a kernel does not guarantee a speedup:
strided destination writes may have lower transfer bandwidth, and the copies
still compete with attention for memory bandwidth.

On the measured 16-GPU C128 workload, this direct-layout variant regressed:
TP4-weight attention → projection took about 601–612 µs, versus about 491–492 µs
for contiguous immutable copies plus restoration and 436–437 µs for DEP16.
The short trace shows direct copies starting near the end of attention, with
only about 11 µs of overlap. TP16-weight also regressed. Keep the direct variant
as a scratch-memory/scheduling experiment, not a recommended serving path.

### Isolate weight and scale-copy scheduling

`benchmark_kimi_k3_weight_scale_copy` compares the original path with copied-
operand and graph-structure controls. It accepts one or more independent TP4
groups; report the actual world size rather than calling a four-GPU run DEP16.

```bash
python -m torch.distributed.run --nnodes=NODE_COUNT --nproc-per-node=4 \
  --node-rank=NODE_RANK --master-addr=HEAD_NODE --master-port=PORT \
  -m test.runtime.distributed.benchmark_kimi_k3_weight_scale_copy \
  --model MODEL_DIR --rows 128 --steps 20 --samples 7 --replays 5 \
  --modes contiguous interleaved weights_only scales_only \
    weights_then_scales scales_then_weights skip_local_scale \
  --output RESULTS_JSON
```

`weights_only` retains all scales, `scales_only` retains the full matrix, and
`skip_local_scale` retains the local scale slice. Those controls isolate copy
costs; they are not equivalent complete-prefetch backends. All copied operands
are poisoned before correctness replays, while explicitly retained operands
stay initialized. Outputs must match the replicated projection bit-exactly.

Adding `_marker` to a direct-copy mode inserts a one-store compute kernel after
the copies. `weights_only_marker_normal` uses a normal-priority auxiliary stream
to distinguish priority from graph structure. The marker changes no data or
required synchronization. DOT dumps allow inspection of actual capture edges.

At C128 on one four-GPU group, removing scale copies alone did not recover
overlap. A marker recovered weight-only overlap at either stream priority, but
interleaved full transfers still stalled at the local scale copy. The complete
`weights_then_scales_marker` path measured about 487 µs versus 491–492 µs for
contiguous copies plus restoration; replicated attention → projection was
about 438 µs. The small gain over the existing prefetch path and reduced scratch
do not establish model speedup or robust scheduling across other graph shapes.
Keep these controls in the benchmark, not in serving defaults.

### Complete-prefetch scheduling optimization

The same benchmark accepts `local_kernel_sm` for a hybrid copy-engine/SM
implementation. It copies the remote weight shards first, then uses one bounded
Triton grid to copy the local shard and gather all peers' scales. The grid uses
the device's SM count. This replaces the late local CUDA copy, small scale
copies and scheduling marker. Remote weight transfers still use copy engines;
the local copy and scale gather do use SM resources.

Activation quantization runs after attention without waiting for weights. The
stream join moves immediately before GEMM. Previous-GEMM scratch release and
immutable peer-source lifetimes are unchanged. Every forward refreshes all
weight bytes and scales; no layer's full weight remains cached as a shortcut.

For a paired C128 comparison, pass:

```bash
--rows 128 --steps 20 --samples 7 --replays 5 \
--modes weights_then_scales_marker remote_first_fused_scales local_kernel_32 local_kernel_sm
```

On one four-GB300 group, the complete attention → projection latency improved
from 487.65 to 447.86 µs, an 8.2% reduction. Replicated weights took 438.23 µs:
the candidate remains 2.2% slower, not faster than the replicated baseline.
Logical full-weight/scale scratch remains 84.02 MiB/rank, plus a 32-byte pointer
table; persistent TP4 weight/scale storage remains 21.01 MiB/rank.

NSYS shows the exposed startup and scale-copy tail removed. The replacement
local-copy kernel completes during attention, but attention itself grows by
about 7 µs. A four-CTA version was substantially slower; fewer CTAs do not
automatically reduce the total overhead. Keep capture length and hardware fixed
when comparing configurations, and use unprofiled timings for performance.

The `local_kernel_sm` check also alternates two real layers through shared
scratch in eager execution and CUDA graphs. Outputs, reconstructed FP8 codes
and scales must match exactly, including changed activations and poisoned
scratch. These checks do not establish full-model performance, dataset accuracy
or behavior with four TP4 groups active. The candidate remains benchmark-only.

Smaller-batch checks also passed. At C64/rank, the previous prefetch path took
293.03 µs and the candidate 255.81 µs, versus 241.69 µs replicated. At C32/rank,
the corresponding medians were 226.85, 210.52 and 135.37 µs; prefetch results
were more variable. Weight-only sharding remains substantially slower at C32.
Do not apply the C128 overlap result to shorter attention windows.

### Further copy tuning: choose by the measured workload

Two additional benchmark options preserve the same complete-prefetch contract:

- `local_asm_load_4096_sm` uses 4096-word local-copy tiles and explicit 128-bit
  weight loads. It retains serialized remote copies and uses one CTA per SM.
  Whole-tile and source-alignment checks prevent vector over-reads.
- `remote_streams_3_asm` additionally copies each remote peer on a separate
  stream, while the local copy and scale gathering run independently. All
  branches inherit the previous GEMM's scratch-release edge; all must complete
  before the next GEMM. Logical weight/scratch storage is unchanged.

On the same four-GB300 setup, paired repeated measurements were:

| Requests/rank | Replicated | Previous local-copy path | Vector copy | Parallel + vector copy |
| --- | ---: | ---: | ---: | ---: |
| 32 | 135.34 µs | 212.14 µs | 197.49 µs | 175.22 µs |
| 64 | 241.80 µs | 255.26 µs | 251.38 µs | 254.58 µs |
| 128 | 438.08 µs | 446.84 µs | 446.02 µs | 450.53 µs |

Parallel copies help C32 but regress at C64/C128. The C128 vector-copy gain is
less than 0.2%, not a substantial new speedup. These are explicit experimental
choices, not an automatic threshold or serving recommendation. All options
still run slower than replicated weights; no full-model gain is established.

In the measured NSYS trace, the remote transfers remained serialized despite
their independent streams. The C32 gain comes mainly from overlapping local
copy/scale work with the remote chain, not concurrent remote transfers or
tripled link bandwidth. At C128, the local kernel shortened from about 58 to
34 µs, but it was already hidden by attention, so total latency barely changed.

The compiler already emitted vector stores; a store hint alone did not help.
Alignment-hint-only modes (`local_vec_load_*`) still emitted scalar loads in
the tested compiler. The `local_asm_load_*` modes use actual vector loads,
verified in PTX. Streaming-cache hints and earlier joins did not improve C128.

Each candidate passes exact byte/scale reconstruction, changed-input and
poisoned-scratch graph replay, and two-real-layer shared-scratch checks. Match
the 20-forward graph length when profiling: a shorter capture can change copy
scheduling. Use repeated unprofiled measurements for performance claims.

### C128: contiguous weight shards and early scale gathering

`row_shards_ce_scales_first` keeps the full replicated GEMM but changes the
persistent weight partition from input channels (K) to output channels (N).
For `[N, K] = [7168, 12288]`, each TP4 rank holds `[1792, 12288]` FP8 codes
and its corresponding scales. These shards occupy contiguous regions in the
ordinary full GEMM matrix, so four CUDA copies replace strided destination
copies and the local SM weight-copy kernel. This is a weight-prefetch
experiment, not column-parallel projection computation.

A small kernel gathers all four scale shards into the prepared scale layout
**before** the weight copies. Scales are immutable and independent of the
weight transfer, so they need not wait behind it. Every forward still refreshes
all weight bytes and scales. The full matrix remains reusable scratch;
no complete layer weight or scale cache bypasses communication.

All four copies run on the auxiliary stream. The previous GEMM releases the
scratch before either scales or weights are overwritten. Attention and activation
quantization run on the main stream; the main stream waits for prefetch only
before GEMM. The numerical computation and output layout remain unchanged.

Compare the previous best path, contiguous copies with late scale gathering,
and the candidate using:

```bash
--rows 128 --steps 20 --samples 15 --replays 20 \
--modes local_asm_load_4096_sm row_shards_ce row_shards_ce_scales_first
```

On one four-GB300 group, with real checkpoint weights, C128/rank and CUDA
graphs, three fresh-process repetitions gave these medians. Each repetition
uses 15 samples of 400 forwards, alternating variant order and taking the
slowest of the four ranks per sample; the table reports the median of the
three run medians.

| Path | Attention → projection | Gap to replicated TP1 |
| --- | ---: | ---: |
| Replicated TP1 | 438.17 µs | — |
| Previous vectorized K-shard prefetch | 445.79 µs | +1.74% |
| Contiguous N-shards, scales last | 443.48 µs | +1.21% |
| Contiguous N-shards, scales first | 441.43 µs | +0.74% |

The final candidate reduces latency by 0.98% against the previous prefetch
path and closes about 57% of its remaining gap to TP1. It remains slower
than replicated weights. Run medians ranged from 441.40 to 441.46 µs, with
matched TP1 medians of 438.13–438.19 µs. A separate 100-forward graph measured
441.41 versus 437.90 µs (+0.80%), with the previous prefetch path at 446.08 µs.
These are component results on one TP4 group, not full-model or DEP16
throughput measurements, and they do not establish TP16-weight performance.

The final 20-forward trace confirms early scale gathering and removal of the
SM weight-copy kernel. Average attention duration falls from 405.52 to 403.28 µs
(TP1: 399.70 µs); quantization-to-GEMM gap falls from 2.04 to 1.65 µs
(TP1: 0.34 µs). The unchanged GEMM also measures faster, 33.19 versus 36.04 µs
for the previous prefetch path and 35.49 µs for TP1. These trace averages explain
the path, but do not replace the unprofiled measurements above. Copies still
contend with attention; the optimization has not eliminated every source of
overhead.

Also repeat with `--steps 100 --replays 4`, keeping the number of forwards
per sample constant. Report these graph lengths separately. Each candidate
must pass exact weight/scale reconstruction and output comparison, changed-input
and poisoned-scratch replay, delayed peers, and alternating two real O-projection
layers through one scratch allocation.

Logical storage stays at 21.01 MiB of persistent weight/scales and 84.02 MiB
of reusable weight/scale scratch per rank, plus the small peer-pointer table.
The N partition requires output width divisible by `128 * TP`. It is not a
drop-in loading change for the activation-TP runtime, which shards O weights
along K. Serving defaults and model loading are unchanged.

The local CUDA copy still starts late in the trace. Early scale gathering
removes the scale kernel from that tail; it does not prove that all copies
overlap attention perfectly. Smaller SM-copy grids, one-warp copies, an early
stream join and local-first copy order did not beat the copy-engine candidate.
CUDA 13's batched-copy overlap hint passed eager checks but was rejected during
graph capture with `cudaErrorStreamCaptureUnsupported`. It is not retained as
a selectable backend, and no environment upgrade was made for it.
