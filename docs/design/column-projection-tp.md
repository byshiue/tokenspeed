# Column-sharded projection with token ownership restored

`runtime/layers/attention/column_proj.py` provides an experimental projection
module independent of model attention/cache parallelism. It is not enabled
by Kimi's output-projection environment variable and does not change any
model forward path automatically.

## Mapping

For TP4 and M local tokens, the operation is:

1. AllGather `[M,K]` inputs into subgroup-rank-major `[4*M,K]` rows.
2. A `ColumnParallelLinear` with `gather_output=False` computes a contiguous
   output-channel shard, giving `[4*M,N/4]` local outputs.
3. A2A sends each token owner's rows back to that owner. Source-rank channel
   shards are concatenated into `[M,N]`; tail padding is then hidden.

There is no cross-rank numerical reduction. Per-GPU weights are approximately
one-quarter of the unsharded projection, while aggregate-token batching keeps
per-GPU FLOPs approximately unchanged. Extra output padding changes FLOPs
slightly and is included in timings.

The module accepts an explicit `DenseLayerMapping`, capacities, output widths,
and backend choices. It reads no environment variables. Callers must prepare
the subgroup and agree on dimensions/backends before collective construction.
The Linear must use that same subgroup/rank, no bias, and no internal output
gather. This module currently requires TP4 and BF16 activations/outputs.

## Quantized weights

`column_projection_width(output_size, tp_size, block_rows)` rounds the total
output width to a multiple of `tp_size * block_rows`. For block-FP8, use
`block_rows=128`: each rank then owns complete 128-row scale blocks.
Load contiguous N shards with the ordinary ColumnParallelLinear weight and
scale loaders. Copy checkpoint codes/scales unchanged; fill extra weight rows
with zero and their scale blocks with one. Preserve each model's fused row
order and pre-existing internal padding before appending any new tail rows.

For KDA's fused QKV/g/f_a/b projection, 49,376 logical rows become 49,664
stored rows, or 12,416 per TP rank. K stays 7,168. At 64 tokens/rank the local
GEMM is `M=256, N=12416, K=7168`, compared with the DEP16 GEMM's
`M=64, N=49408, K=7168` (including its original 32 padding rows).

## Communication and lifetime

AllGather choices are `nccl` and `triton_multimem`. The latter wraps the
existing TokenSpeed NVIDIA multicast all-gather and returns borrowed input
storage directly to GEMM. It adds a pre-copy reuse barrier so a faster rank
cannot overwrite data that a slower rank's preceding GEMM still consumes.
That barrier has a separate symmetric-memory signal pad from the multicast
kernel's per-CTA protocol. Do not remove it based on equal problem sizes.

A2A choices are `nccl` and `flashinfer`. NCCL receives source-rank-major
channel slices and copies them into token-major order. FlashInfer uses the
inverse Ulysses `gather_heads` operation to produce token-major full-channel
outputs directly. Runtime imports go through `tokenspeed_kernel.ops`; the
FlashInfer call remains under the kernel package's third-party boundary.

Construction initializes persistent scratch and optional communication state
before CUDA-graph capture/cache budgeting. Warm kernel dispatch before capture.
Every operation and consumer must remain on one serialized stream per module.
Outputs own their storage, including the one-token case; logical tail slicing
can leave a padded row stride, so outputs need not be contiguous.
Close collectively only after all graphs and borrowed consumers are gone.
Explicit backend initialization failures propagate; there is no rank-local
collective fallback during forward.

`counts` describes agreed physical row counts across world ranks, including
graph padding. A non-empty subgroup pads every rank to its maximum count;
zero-input ranks still participate. An entirely empty subgroup skips both
collectives. This is the same execution path for eager and graph replay, with
no scheduler, cache ownership, or per-request state change.

## Validation and measurement

`test/runtime/distributed/test_column_projection.py` covers width padding,
settings, uneven-rank/channel restoration, empty batches and owned-output
lifetime, including one row. GPU validation must additionally exercise the
real collectives, FP8 scale shards, changing graph inputs and delayed peers.

Compare complete AllGather/GEMM/A2A latency against DEP16 for identical local
tokens and real weights. Include padding and output restoration. GEMM-only
and isolated collective timings are diagnostic, not an E2E speedup claim;
their sum need not equal complete-operation latency. Backend selection must
follow measured shapes: a faster GEMM or one faster collective is insufficient.
