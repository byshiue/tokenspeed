# Kimi-K3 DEP output-projection TP

## Contract

Attention and its caches remain TP1/DP-world; routed experts remain EP-world.
Only KDA/MLA output projections use a separate contiguous TP subgroup. Set
`TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE=4` for projection TP4/DP4 in a 16-rank
deployment. Unset or `1` retains the existing projection and communication.
The setting is agreed across ranks before loading weights.

Each rank starts with its local tokens and all attention-output channels.
An all-to-all transposes token ownership into input-channel ownership within
the subgroup. Each shard projects all subgroup tokens. A reduce-scatter sums
partial hidden states and returns each token to its original DP rank.
AttnRes, residuals, routing and shared experts therefore retain their existing
local-token contract. The MoE's `_forward_attn_dp` does not need a second
layout implementation or an additional world-wide transition.

KDA gating/normalization and MLA output gating precede redistribution.
The registered `o_proj` Linear retains the checkpoint parameter names and
uses ordinary row-sharded weight/scale loaders. A projection TP group must
divide the input dimension without splitting quantization blocks.

Equal-size padded messages handle uneven token counts; an empty local rank
still participates when a peer has tokens. An entirely empty subgroup can
skip both collectives. The scheduler's collective counts, including graph
padding, determine message shapes. No new device-to-host count synchronization
is introduced.

## Implementation boundaries

- A shared Kimi projection helper owns subgroup selection, exchange and local
  output restoration; it does not own request or cache state.
- KDA constructs the sharded Linear directly. MLA uses a construction hook
  so it never allocates a full output weight before replacing it.
- Communication groups initialize before model loading. Exchange scratch is
  shared across sequential layers and allocated during communication
  preparation, before cache budgeting and CUDA-graph capture.
- Attention TP reductions remain attached to the original attention mapping;
  they must not reduce the completed projection a second time.
- Eager and captured forwards invoke the same helper.

### Optional small-message A2A

`TOKENSPEED_KIMI_K3_O_PROJ_A2A_BACKEND` selects `nccl` (default),
`auto`, or `flashinfer`. All ranks agree on both projection settings at
startup. This does not change the MoE communication backend.

The optional FlashInfer Ulysses NVLink exchange fuses the token/channel
transpose with A2A. Its adapter stays under tokenspeed-kernel. A shared
workspace creates the communicator and its IPC/JIT resources before graph
capture; sequential layers reuse it on one stream. Repeated preparation
retains the workspace. IPC resources must outlive every graph that refers to
them; explicit close is collective and is only safe after those graphs die.

The initial measured envelope is balanced physical batches of 1–16 rows per
rank with aligned channel shards. Larger or unequal subgroup counts use the
existing padded NCCL exchange. This decision uses the same host counts on
every subgroup rank, not forward mode or local occupancy. Graph padding can
make physical counts equal even when valid request counts differ.

`auto` falls back at startup when the optional API or supported NVLink
topology is unavailable. `flashinfer` requires NVLink initialization to
succeed, but still uses the shape-based NCCL fallback outside the envelope.
Forward errors propagate; switching collectives after a rank-local failure
would risk a deadlock. GEMM, quantization, ReduceScatter and output ownership
are unchanged. NCCL remains the default until full-model validation passes.

Expected source areas are Kimi model integration, the shared projection
helper, the MLA construction hook and distributed tests. Scheduler, cache
geometry, attention input projections and expert placement do not change.
Implementation, tests and deployment documentation belong to one PR.

### Optional ReduceScatter

`TOKENSPEED_KIMI_K3_O_PROJ_RS_BACKEND=triton_rsag` selects the existing
token-aware Triton RSAG implementation; `nccl` remains the default. Rank
agreement covers TP size and both communication settings. The settings are
separate, but RSAG currently runs only with the fused FlashInfer A2A path.

Communication preparation allocates symmetric scratch once per distinct
projection output width and warms the reduction before capture. Capacity is
`TP * min(max_tokens, 16)` rows. BF16 and output widths divisible by eight
are required. On NVIDIA, this path needs NVLink multicast support; an explicit
request fails startup if initialization is unsupported rather than attempting
a rank-local recovery.

All ranks in the subgroup select the backend from the same physical counts.
Balanced batches of 1–16 rows use RSAG when FlashInfer A2A is active; larger
or uneven physical batches use NCCL for reduction as well as A2A. Graph padding
may make physical counts equal while valid request counts differ. An entirely
empty subgroup skips both collectives. The rule does not depend on forward
mode or whether execution is eager or captured.

This restriction follows complete-projection measurements: RSAG with NCCL A2A
regressed some 17–64-row and uneven cases, although RSAG with FlashInfer A2A
was faster. Do not extend the threshold based on isolated reduction timings.
If FlashInfer A2A is unavailable or disabled, initialization logs the fallback
and allocates no RSAG scratch.

RSAG stages the GEMM result into its symmetric buffer and clones the reduced
output. Do not expose a view: the next attention layer reuses that buffer.
With TP4, width 7168, and capacity 16 rows per rank, payload scratch is
896 KiB per GPU per output width, plus synchronization metadata and output
allocations. Graphs and in-flight operations must be released before closing
the shared workspace.

The NVIDIA multimem reduction accumulates BF16 inputs in FP32. Its outputs
need not match NCCL's reduction order bit for bit. Tests report error against
both NCCL and an FP32 reference; eager/graph equivalence and output lifetime
remain exact checks. This remains opt-in until full-model validation passes.

### Copy-free peer reduction

`TOKENSPEED_KIMI_K3_O_PROJ_RS_BACKEND=triton_peer` selects a TP4 peer-read
reduction with the same balanced 1–16-row FlashInfer A2A envelope. NCCL remains
the default and the fallback outside that envelope.

The A2A adapter borrows FlashInfer's persistent receive storage instead of
copying it into a second tensor. The following GEMM must consume that view
on the same serialized stream before the next exchange. The adapter retains
the communicator and its entry/exit barriers; removing the copy does not
remove peer synchronization. Its CUDA source ships with tokenspeed-kernel
and compiles against the installed FlashInfer headers before graph capture.

Prepared FP8 GEMMs write directly into persistent symmetric partial-result
storage when their output shape and layout permit it. Padded or incompatible
destinations retain a copy fallback. A publication barrier precedes the
peer-read reduction, which accumulates in FP32 and writes an owned local
output. In the complete projection path, the next A2A entry barrier supplies
the reuse fence: every peer has completed the preceding reduction before
any peer can launch its next GEMM into symmetric storage. NCCL fallback
shapes and empty groups do not write that storage. Standalone reduction
calls retain an explicit trailing barrier. This optimization requires the
same subgroup and serialized stream for A2A, GEMM and reduction.
Only internal intermediates are borrowed: returned model outputs
must survive the next layer's workspace reuse.

The TP4 partial buffer needs 896 KiB per GPU at width 7168 and 16 rows per
rank, excluding synchronization metadata, A2A storage and graph allocations.
Explicit initialization errors are fatal; never switch collectives after a
rank-local forward failure. Full-model accuracy remains a separate gate.

### Quantized borrowed exchange

Within the peer-reduction envelope, prepared block-FP8 projections with
16 physical rows per rank can quantize while redistributing. Each sender
quantizes complete 128-channel groups, then writes FP8 values and FP32 scales
to its peers. TP4 channel boundaries must preserve these groups. The GEMM
consumes the received MN-major scales explicitly; it must not reinterpret
them as canonical scales or quantize the values a second time.

Values and scales borrow disjoint regions of the existing communicator
allocation. Its entry and exit barriers, stream discipline and output
ownership contract remain unchanged. The native module initializes before
capture. BF16 projections, unsupported prepared plans, smaller batches and
NCCL fallback shapes retain their existing routes. Shared-buffer paired
measurements found no gain at 2/4/8 rows, so these shapes keep BF16 exchange.
This optimization adds no persistent communication allocation.

## Validation gates

1. Run a standalone four-GPU harness using small BF16 matrices, then production
   KDA/MLA dimensions. Compare against independent TP1 projections on the same
   tokens. Cover uneven traffic, empty ranks, token ordering and graph replay
   with changing inputs.
2. Load representative real checkpoint projection weights and scales, without
   loading the full model. NVFP4 model naming does not imply NVFP4 attention
   projections: mixed checkpoints may use FP8 block-scaled output weights.
   Test the actual projection arithmetic and scale sharding.
3. Repeat on 16 GPUs, including inactive subgroups. Benchmark the full exchange
   plus GEMM, not GEMM alone, using warmup and repeated unprofiled measurements.
4. In parallel, establish unchanged full-model DEP16 serving on a fresh
   persistent allocation. After the small tests pass, compare feature-on
   serving with the same baseline settings and request corpus.
5. Validate logits/generation, prefill/decode and multi-turn prefix reuse.
   Measure latency, throughput, memory and short NSYS captures with CUDA
   graphs enabled. Start without a speculative drafter.

BF16/FP8 partial reductions need not be bitwise identical to TP1. Numerical
reports must include error against TP1 and a higher-precision reference;
quantization error and additional sharding error are different measurements.
Failures are investigated rather than hidden by increasing tolerances.

## Expected results

Projection weight storage per rank falls approximately by the subgroup size.
This is not the reduction in total model memory. With balanced traffic,
each rank handles more tokens and fewer input channels, so its projection
FLOPs remain approximately unchanged. Larger GEMMs may be more efficient,
but packing, communication and unequal-batch padding can outweigh that gain.
No speedup is claimed until the complete operation and full model are measured.

See [the DEP runbook](../recipes/kimi-k3-dep-o-proj-tp.md) for commands.
Standalone redistribution measurements and their limits are recorded in
[the performance notes](kimi-k3-o-proj-performance.md).
