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
