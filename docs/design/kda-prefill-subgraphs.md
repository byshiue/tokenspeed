# Capacity-based KDA prefill graphs

KDA prefill launches many short kernels between the existing breakable graph
segments. Replaying that attention region as a CUDA graph reduces host launch
overhead while keeping the existing extend implementation.

## Implementation

Set `TOKENSPEED_KDA_PREFILL_GRAPH=1` with the `cutedsl_kda` backend and
prefill CUDA graphs enabled. The default is off. The cache uses the padded
token extent chosen by the outer prefill graph; it retains up to eight
schedules per sequence count, including stream and PDL variants.

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

Regression tests cover capacity admission, inactive convolution programs,
metadata isolation, exactly-once cache writes, changed input addresses,
schedule limits, metadata restoration on failure, stream-local pool sharing
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
