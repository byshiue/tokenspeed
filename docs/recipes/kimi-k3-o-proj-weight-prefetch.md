# KDA O-projection weight-prefetch experiment

This component benchmark compares four ways to execute Kimi-K3's O projection
under DEP16. It does not add a serving mode or load the full model.
Use the [TP-sharding recipe](kimi-k3-tp-sharding.md) for the shared environment,
persistent allocation and existing runtime paths.

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
21.01 MiB with TP4-weight or 5.25 MiB with TP16-weight. The current implementation
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
