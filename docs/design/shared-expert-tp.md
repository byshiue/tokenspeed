# Shared-expert TP with DP token ownership

Kimi-K3 may shard its BF16 shared-expert MLP independently of attention and
routed experts. `TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE=4` enables this for
attention TP1/DPworld, routed MoE TP1/EPworld, PP1. Unset or `1` preserves the
original shared expert. Other sizes/topologies fail at initialization; ranks
agree on raw settings before creating groups.

The existing merged-column loader shards corresponding intermediate rows of
gate and up; the row-linear loader shards matching down-projection columns.
No full shared-expert weight replica is needed on the optimized path.
The down Linear must not perform an additional all-reduce.

The forward chain is AllGather -> gate/up -> SiTU -> down -> ReduceScatter.
Inputs are padded to the maximum physical count in each contiguous TP4
subgroup. Every subgroup peer participates, including zero-token owners;
an entirely empty subgroup skips both collectives. ReduceScatter restores
local token ownership before shared/routed/residual addition. There is no
intermediate A2A and no scheduler/cache-layout change.

Shared AllGather starts in the early auxiliary branch, followed by shared MLP
computation. Main runs local routing, top-k, input projection and quantization
concurrently. An intermediate event recorded immediately after AllGather lets
main wait for gathering alone before routed dispatch (or its all-gather
fallback); shared GEMMs may still overlap dispatch. The input must already
contain this layer's attention/AttnRes/norm result, so gathering cannot move
ahead of that producer. Main joins shared compute before routed BMM, avoiding
competition between the two GEMM paths. A second auxiliary branch waits for
dispatch and performs shared ReduceScatter while routed BMM runs. Main joins
that reduction before MoE combine. Both event boundaries apply to empty
owners as well as active ones; an entirely empty subgroup skips its kernels,
not the ordering of the surrounding routed work.

Thus shared collectives never overlap routed dispatch/combine. Simultaneously
resident peer-polling kernels can otherwise exhaust SM resources and deadlock
despite using separate communication buffers. `StreamFork.join()` waits for
the latest branch without closing the scope; `branch_after_main()` records
the next fork boundary. `record_checkpoint()` inside the branch and
`join_checkpoint()` on main use a separate preallocated event; the branch-end
event cannot express the gather-only dependency. Record the checkpoint in
each scope before joining it, including empty owners/subgroups. Captured
graphs retain these event generations. Eager
execution without an auxiliary fork follows the same stages serially; graph
warmup also honors the existing non-overlap policy. Do not put the full TP
chain into one unconstrained concurrent branch.

MLP construction receives explicit TP rank, size and group from its owner,
matching upstream dense/shared MLP initialization. The optional shared mapping
must agree with that geometry and use deferred reduction. When MegaMoE owns
dispatch/combine inside the expert kernel, main also joins shared reduction
before that call; only backends with separate dispatch and compute can overlap
shared reduction with routed BMM.

One workspace is
prepared by the model before memory profiling and graph capture, shared by
sequential shared-expert layers. It is separate from other main-stream
scratch: concurrent main/auxiliary users must never alias IPC buffers.
The same count-driven path handles eager, prefill and captured decode.

For up to 128 padded rows/rank, use the tested TRT-LLM one-shot AllGather and
Lamport ReduceScatter. Larger shapes use existing NCCL collectives with the
same sharded weights; selection depends only on agreed host counts. Native
initialization errors are fatal, not rank-local fallbacks. BF16 hidden width
7168 is the currently validated contract. The kernel package owns native
bindings; runtime imports only the tokenspeed-kernel operation boundary.

AllGather output is borrowed and consumed by the MLP before the next gather.
Reduction returns owned output rows; captured graphs retain scratch for their
lifetime. Close collectively only after all graph and stream consumers finish.
Changing communication capacity requires rebuilding the prepared workspace.

Validation covers exact checkpoint shards, the original BF16 MLP reference,
uneven/empty owners, retained outputs, auxiliary-stream graph replay with
changing inputs and delayed peers, and the 128/129 fallback boundary.
Model orchestration tests pin AllGather inside the early branch, its checkpoint
before routed communication, shared compute before
routed BMM, and reduction between dispatch and combine, including empty
owners and eager/warmup/capture execution. The distributed validator exercises
the same event generations and auxiliary reduction during repeated replay.
Down-GEMM sharding changes BF16 rounding, so numerical tolerance is explicit.
Shared-MLP microbenchmark gains do not establish full-model speedup: measure
decode with real routed experts on the concurrent main stream and verify the
requested number of active requests is actually admitted.
