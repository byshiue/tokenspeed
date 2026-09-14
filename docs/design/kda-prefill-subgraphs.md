# Capacity-based KDA prefill graphs

KDA prefill launches many short kernels between the existing breakable graph
segments. Replaying that attention region as a CUDA graph reduces host launch
overhead while keeping the existing extend implementation.

## Implementation

### Merged outer capture

Eligible KDA layers now capture directly inside the outer prefill graph,
alongside their projections and post-attention operations. MLA remains an
eager break. This removes the separate per-layer KDA graph launch and its
output handoff copy. Padding is cleared inside the merged graph.

The backend prepares stable capacity metadata before capture and refreshes
live boundaries, convolution maps and state-page indices once before replay.
The merged capture accepts any positive live length within its token bucket,
with the same request count used at capture. The original outer graph remains
available for mixed batches and uncaptured request counts. Layerwise PD transfer
and data parallelism retain that original route.

Each merged capture reserves one checkpoint/tail slot per request and records
the same body scan, checkpoint write and tail scan used by eager prefill.
The graph key is just `(token bucket, request count)`: changing the number or
identity of checkpointed requests refreshes metadata, not the graph.
The body reserves the outer bucket's capacity. Tail storage reserves
`min(bucket, BS * max(1, prefix_granularity - 1))` tokens.

A request without an internal checkpoint runs entirely in the body. Its tail
slot contains one zero-input dummy token, keeping every native scan sequence
positive-length. Negative token indices suppress that slot's output, negative
checkpoint destinations suppress cache writes, and a negative state-update
row preserves the body's final state. Dummy scan results are discarded, not
assumed to be identity updates. This leaves native scan/GEMM arithmetic intact.
The tradeoff is a fixed tail launch even when no request has a checkpoint.

Replay refreshes both scans' boundaries and token maps, state-update rows and
scheduler-owned checkpoint page IDs once before the first layer. The metadata
refresh does not allocate request state or modify the scheduler's source
metadata. Startup uses only placeholder state pages.

`--prefill-graph-capture-batch-sizes 1 2` captures exact request counts one
and two. It is independent of the decode batch ladder. The token capacities
still come from `--prefill-graph-capture-sizes`. When the request-count option
is unset, each token bucket uses the minimum number of requests needed to fit
within the model context. Capture balances tokens across those requests;
it never inserts empty sequences. A configured count that cannot fill a
particular token bucket within the context limit is skipped for that bucket.

For example, two extends of 868 and 869 tokens with aligned cached prefixes
use the 2048-token, two-request capture. Their bodies contain
768 tokens each and their tails contain 100 and 101 tokens. The body reserves
2048 tokens; the packed tail reserves at most 254 at prefix granularity 128.
No-tail, one-tail and two-tail batches all reuse this capture, including when
the checkpoint moves between request rows.

Request-count coverage is opt-in because it multiplies startup work.
Configuring 1 and 2 creates two inline variants per token bucket, in addition
to the ordinary graph. With eight token buckets this is 16 inline variants,
down from 40 with exact checkpoint-count variants. Shared pools reuse scratch,
but retained metadata and
graph objects still cost memory; bucket count alone does not bound that cost.

All capture variants belong to the outer graph owner and share its pool;
they are never replayed concurrently. This adds capture work and retained
metadata. The added cost relative to single-request capture and the full-model
speedup still need controlled measurements.

### Separate KDA graphs for the original outer capture

Set `TOKENSPEED_KDA_PREFILL_GRAPH=1` with the `cutedsl_kda` backend and
prefill CUDA graphs enabled. The default is off. The cache uses the padded
token extent chosen by the outer prefill graph, following
`--prefill-graph-capture-sizes` or the default outer bucket ladder.
There is no separate KDA size list or schedule-count limit. Schedules are
created lazily for each encountered bucket, sequence count, stream and PDL
setting. Configuring more buckets can increase capture time and memory use;
the eight-bucket measurements below do not quantify larger configurations.

The first call warms native plans. The second captures the same callable and
replays it once. Later calls refresh stable metadata buffers and replay.
Changing tensor addresses, shapes, strides or scalar arguments uses the
ordinary callable instead. Cache-pool rebinding releases the graph cache.

The new graph covers state staging, causal convolution, QKV splitting, the
second gate projection, scan input preparation, KDA scan and state writeback.
QKV projection, gated RMSNorm and output projection remain in their existing
outer graph segments. Shared metadata refresh runs once per forward outside
the new graph.

`KdaPrefillCapacity` separates fixed launch capacity from live sequence
boundaries. The facade validates the CPU length mirror; the CuTeDSL adapter
uses capacity bounds for host planning while kernels consume live GPU bounds.
Inactive convolution programs receive the padding sentinel. Scan input padding
is cleared before native full-tile loads. Other backends retain exact-length
planning. The native scan and GEMM arithmetic are unchanged.

Layers on the same replay stream share a private graph pool. Different
streams have separate pools, and the outer graph pool stays separate because
its intermediates remain live across the attention break. See
[the execution invariants](unified_path.md#experimental-kda-prefill-subgraphs).

## Measured performance

These measurements describe the earlier separate-KDA-graph implementation,
not the merged outer capture described above.

The comparison used the same source with the KDA graph switch OFF and ON:
real 93-layer Kimi-K3 NVFP4 weights, TP8 on eight Blackwell GPUs in one NVLink
domain, BF16 computation, FP8 KV cache, CuTeDSL KDA, CUDA graphs and overlap
enabled. Timing runs did not use NSYS or per-request replay instrumentation.
The serving environment used PyTorch 2.13.0+cu130 and InstantTensor 0.1.9.

The workload used public SWE-smith trajectories
(`SWE-bench/SWE-smith-trajectories`) to build six serial conversations with
eleven requests each. Two conversations warmed the model; four supplied
44 measured requests, including four cold prompts and 40 incremental requests.
Each request generated 500 tokens. Baseline input token IDs were frozen for
ON. Cold prompts targeted 50,000 tokens; incremental prefill work ranged from
681 to 6,714 tokens. The eight capacities were
128, 256, 512, 768, 1024, 2048, 4096 and 8192.

| Metric | OFF | ON | Reduction |
|---|---:|---:|---:|
| Incremental request prefill, median | 243.570 ms | 201.850 ms | 17.13% |
| Incremental prefill, sum over 40 requests | 10.768 s | 9.148 s | 15.04% |
| Cold-prompt prefill, median | 2453.700 ms | 2471.890 ms | -0.74% |
| Client E2E, sum over 44 requests | 230.171 s | 229.422 s | 0.33% |

Prefill time is the request's scheduled-to-prefill-complete interval and can
include multiple forward steps. It is not a single kernel's duration.
An OFF repeat measured a 244.000 ms incremental prefill median and a
229.878 s E2E sum. The small E2E difference is not evidence of a stable
end-to-end speedup: 500-token decoding dominates this workload.

Populating all eight capacities added 3.440 GiB of live Torch allocation per
GPU. Graph-pool live allocation was 3.408 GiB and its reservation was
3.723 GiB. These are overlapping accounting views, not additive costs.
This measurement covers single-sequence serving; additional sequence counts
or streams can consume more memory.

## Validation

For the merged capture, GPU regressions check that state-attention breaks
disappear while full-attention breaks remain. They also check changing live
lengths and page IDs, exactly-once state writes, metadata restoration, and
fallback selection. A saved-real-activation check covered 18 complete KDA
regions at live lengths 58, 76, 94, 1536, 3584 and 6656. Each region captured
as one graph; 54 shared-pool replays in forward and reverse order matched
eager outputs and convolution/recurrent states bit-for-bit on each of two
GPUs. This is operator-level validation, not a new full-model benchmark.

Checkpoint regressions additionally replay an 837-token extend (768-token
body plus 69-token tail), changing lengths within the same 1024-token bucket,
fresh and resumed state, and different checkpoint/continuation pages. A
two-request test moves the checkpoint between request rows. Each replay must
match eager token outputs and the entire convolution/recurrent state pools
bit-for-bit, with zero output padding. Padded pack/scatter tests also cover
strided scan outputs and both KDA and GDN token-axis conventions.

Multi-request regressions exercise startup capture and selection for configured
request counts, balanced placeholder batches, and zero, some or all requests
with checkpoints. Native KDA tests compare valid outputs and whole state pools
against eager for request counts 1, 2 and 4, including the 868+869 example.
These are operator and graph-owner checks, not an AIME accuracy result.

A full-model correctness check used real 93-layer Kimi-K3 NVFP4 weights,
TP8 on eight Blackwell GPUs, CuTeDSL KDA, BF16 computation, FP8 KV cache and
overlap. Paired inputs were constructed from a frozen SWE-smith trajectory:
both requests shared 50,304 cached tokens, followed by extends of 868/869,
868/768 or 768/896 tokens. Each request generated eight tokens.
Two ordinary-graph passes, three merged-graph passes and two eager passes
produced identical output tokens for all 42 generated sequences. Actual replay
records on all eight ranks verified the two-request, two/one/zero-checkpoint
variants; HTTP request count alone was not used as evidence of batching.

A further 16-sequence cold/warm check consumed the internal checkpoints from
the 868/869 pair. Both requests then hit 51,072 cached tokens and extended only
100/101 tokens. Outputs matched between cold and warm and between ordinary and
merged graphs, including repeats. All-rank records verified the warm
two-request, zero-checkpoint replay in the 256-token bucket.

The paired requests were queued through the existing admission gate and
released together to make batch size two reproducible. These runs used local
path auditing, without NSYS. They establish correctness for the tested inputs,
not benchmark latency, general batch-invariant generation or an AIME score.

The following larger validation runs were performed on the earlier separate
subgraphs:

Regression tests cover capacity admission, inactive convolution programs,
metadata isolation, exactly-once cache writes, changed input addresses,
replay across more than eight buckets, metadata restoration on failure, stream-local pool sharing
and cache reinitialization.

GPU validation exercised 1,920 capacity cases across eight buckets and
sequence counts 1, 2 and 4. Real-activation checks compared 414 native-scan
cases and 414 complete KDA cases, including outputs and convolution/recurrent
state. Shared-pool checks covered 828 reordered replays and 207 replays with
different live lengths in the same bucket; these comparisons were bitwise equal.

A separate full-model correctness run replayed all 66 frozen requests in
OFF/OFF/ON/ON/OFF phases. All ten phase pairs matched all output tokens and
cache/scheduling lengths. Startup health checks were retained, with a
3600-second periodic health interval; logs confirmed no health request
overlapped the 330-request window. Earlier runs with periodic health traffic
showed output variation even within the same mode. This is controlled workflow
parity, not general batch-invariant generation or a task-accuracy score.

Separate NSYS captures verified graph replay on all eight ranks without new
captures in the measured interval. Their timings are not used in the table.
