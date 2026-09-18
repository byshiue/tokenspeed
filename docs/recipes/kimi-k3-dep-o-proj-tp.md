# Kimi-K3 DEP16 and projection TP4

This runbook separates an unchanged DEP16 baseline from projection-only TP4.
Use a real mixed-precision NVFP4 checkpoint and the same source revision,
dependencies, GPU topology and MoE transport for both runs. Do not infer
attention-projection precision from the model name: inspect its layer metadata.

Treat projection TP as experimental until your checkpoint passes end-to-end
accuracy checks. Sharding changes low-precision accumulation order and can
change deterministic generation. Weight-memory savings do not imply a speedup;
include both collectives and uneven-traffic padding in the comparison.

## Persistent allocation and launch

Reserve 16 GPUs in one supported fabric. For four GPUs per node, use four
nodes and keep each contiguous projection TP4 subgroup within a node:

```bash
salloc --no-shell --account=<account> --partition=<partition> \
  --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 \
  --cpus-per-task=<cpus> --mem=0 --exclusive --time=<allowed-duration>
```

Run all work through a submit wrapper using `srun --jobid=<allocation>`.
Initialize the cached container synchronously on all nodes first. Activate the
serving venv, build the scheduler for this source revision, and verify imports,
all 16 GPU UUIDs and fabric health. Reuse configured image/package caches.
Do not change a shared venv or borrow another task's running server.

Inside that wrapper, launch one server process per node:

```bash
python -m tokenspeed.cli serve --model <real-model-path> \
  --language-model-only --trust-remote-code --load-format instanttensor \
  --dtype bfloat16 --quantization nvfp4 --kv-cache-dtype fp8 \
  --world-size 16 --nprocs-per-node 4 \
  --attn-tp-size 1 --data-parallel-size 16 --dense-tp-size 1 \
  --moe-tp-size 1 --expert-parallel-size 16 \
  --attention-backend tokenspeed_mla --kda-backend cutedsl_kda \
  --moe-backend flashinfer_trtllm --all2all-backend flashinfer \
  --dist-init-addr <head-node>:<rendezvous-port> \
  --max-model-len 65536 --max-num-seqs 32 \
  --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --prefix-granularity 128 --seed 1 --disable-autotune \
  --gpu-memory-utilization 0.75 --disable-kvstore --enable-prefix-caching \
  --cudagraph-capture-sizes 1 2 4 --max-cudagraph-capture-size 4 \
  --disable-cuda-graph-padding --enable-log-request-stats \
  --enable-output-logprobs \
  --policy cache_aware --dp-aware --host 0.0.0.0 --port <http-port>
```

Multi-node Slurm steps provide node count/rank automatically. Keep linear
attention TP at its inherited value of 1. Here DEP means attention DP16 plus
MoE EP16, not the DeepEP transport. FlashInfer requires a supported shared
fabric; `agrs` is the reference alternative. Never compare different MoE
transports while attributing a difference to output-projection TP.

For a raw-token gRPC harness, add `--skip-tokenizer-init` and tokenize in the
client. Install the repository-pinned dependency versions in an isolated venv;
also build or expose the matching native kernel objects in a new worktree.
A successful import check alone does not exercise native libraries first used
during full-model warmup.

For baseline, leave `TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE` unset. For feature-on,
export the following on **every node before model startup**:

```bash
export TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE=4
```

With projection TP enabled, omitted backend settings now select BF16
FlashInfer A2A (`flashinfer`) and custom symmetric reduction (`triton_peer`).
Backend overrides use `TOKENSPEED_O_PROJ_A2A_BACKEND` and
`TOKENSPEED_O_PROJ_RS_BACKEND`; replace the former Kimi-prefixed backend
variables in existing launch scripts. The old names are no longer read.
The TP-size switch remains Kimi-specific. All three settings and defaults are
registered in `runtime/utils/env.py` as raw strings: ranks agree on their values
before strict validation, so malformed input is rejected rather than silently
replaced by a default. Shared projection construction,
execution, and settings validation live in `layers/attention/o_proj.py`;
`models/kimi_k3.py` owns feature enablement and DEP topology restrictions.
The thresholds are independent and refer to physical tokens per rank, including
graph padding—not context length or subgroup-total tokens:

| Physical rows/rank | A2A | Reduction |
|---|---|---|
| 1–512, balanced subgroup | FlashInfer BF16 | Symmetric |
| 513–8192 | NCCL | Symmetric |
| Above 8192 | NCCL | NCCL |

Uneven subgroups use padded NCCL A2A, but can still use symmetric reduction
when their maximum count is at most 8192. Entirely empty groups skip both.
The actual prepared workspace capacity may be smaller than these limits.

FlashInfer requires the optional Ulysses API and supported NVLink topology.
The default `flashinfer` setting fails startup if initialization is unavailable.
Set A2A to `auto` for its startup fallback, or choose `nccl` explicitly.
Symmetric reduction requires TP4, BF16 partials and supported peer access;
non-TP4 groups fall back to NCCL reduction. Explicit initialization errors
are fatal. Never retry another collective after a rank-local forward failure.

To run the NCCL reference, set both backends explicitly:

```bash
export TOKENSPEED_O_PROJ_A2A_BACKEND=nccl
export TOKENSPEED_O_PROJ_RS_BACKEND=nccl
```

The FlashInfer path exchanges BF16 values and quantizes separately before FP8
GEMM. Supported A2A choices are `nccl`, `auto`, and `flashinfer`; reduction
choices are `nccl`, `triton_peer`, and experimental `trtllm_lamport`.

To evaluate native TRT-LLM one-shot Lamport reduce-scatter while retaining FI
A2A, set `TOKENSPEED_O_PROJ_RS_BACKEND=trtllm_lamport`. This opt-in requires
TP4, BF16 partials and working CUDA IPC/native TRT-LLM communication objects.
It supports up to 128 physical rows/rank, including padding; larger batches
use NCCL reduction. A2A selection is unchanged. Initialization errors are
fatal, not rank-local fallbacks. The default remains `triton_peer`.

Lamport uses local GEMM output plus a separately published IPC ring. Its
native protocol handles publication and ring reuse, replacing the explicit
symmetric-memory barrier and peer-read reduction. It does not add residual
or normalization math. IPC scratch is shared across sequential layers of
the same output width and allocated before cache sizing and graph capture.
At width 7168 and capacity 128 it reserves about 114 MiB/GPU of IPC scratch,
plus 7 MiB of local GEMM partials. Calls must stay on one serialized stream,
and collective close must occur only after all referencing graphs are gone.
The 128-row bound limits memory use; it is not a measured crossover point.

Run the standalone validator above with the same RS environment override and
`--large-tokens` to exercise the 128/129-row Lamport/NCCL transition as well as
the A2A boundaries. Projection microbenchmarks alone do not establish a
full-model speedup.

At output width 7168, the TP4 symmetric partial buffer reserves
`4 * min(workspace_capacity, 8192) * 7168 * 2` bytes per GPU: up to **448 MiB**,
shared across sequential layers of that width. A2A, generic packing scratch,
graph pools and synchronization metadata are additional. Buffers initialize
before cache budgeting and capture. Increasing this capacity can reduce the
memory available for KV cache.

The large-token backend sweep showed that BF16 FI + Sym is faster than NCCL
for some balanced shapes, but the table above is a routing policy, not an E2E
speedup claim. In particular, NCCL A2A + Sym at 513–8192 is a different
combination from the historical all-FlashInfer sweep and requires its own
measurement. TP1 can remain faster despite using more weight storage.

This setting changes neither attention cache ownership nor EP placement.
Unset/`1` preserves the original projection. The value must divide world size
and projection dimensions, respect quantization alignment, and agree on all
ranks. Projection TP requires attention TP1/DP-world, linear-attention TP1,
MoE EP-world and PP1.

## Execution and buffer contract

The reusable wrapper is `DistributedOutputProjection` in
`tokenspeed.runtime.layers.attention.o_proj`. It takes an explicit projection
mapping and per-rank physical token counts; workspace initialization takes
explicit A2A and reduction backends. It reads no Kimi environment settings and
does not require any particular MoE layout. Kimi configuration parsing, rank
agreement and attention/MoE layout validation stay in the model integration.

Inputs contain locally owned tokens and all attention-output channels. Within
each contiguous TP subgroup, A2A changes token ownership into channel-shard
ownership. `RowParallelLinear` projects the subgroup tokens, and reduce-scatter
returns complete outputs to the original token owners before residual,
normalization or AttnRes processing. The Linear's own reduction is disabled to
avoid reducing twice. Weight shards must preserve the resolved quantization
block boundaries. KDA/MLA gating remains before this exchange.

Every subgroup rank participates when any peer has tokens, including ranks
with empty local inputs. Uneven counts use equal-size padded NCCL messages;
an entirely empty subgroup skips communication. Selection depends on shared
physical counts, including graph padding, not on forward mode.

Initialize process groups, communication buffers and native modules before
CUDA-graph capture and cache budgeting. Sequential attention layers share
one workspace. Keep calls on one serialized stream: the next borrowed A2A's
entry barrier protects reuse of symmetric GEMM output after peer reads.
When using NCCL A2A with symmetric reduction, an explicit pre-write barrier
protects transitions from the FI path before GEMM touches symmetric storage;
the reduction also retains its trailing reuse fence. NCCL A2A alone is not a
symmetric-buffer reuse fence.
The peer reduction preserves the 16-byte alignment of symmetric allocation
bases through its indirect pointer loads, allowing vectorized BF16 reads.
Owner offsets and masked tails still determine the safe access width;
unaligned slices are supported. Accumulation remains FP32 in peer-rank order,
with the final result stored as BF16. This changes neither synchronization
nor buffer ownership.
Never replace or close buffers while captured graphs still reference them;
explicit close is collective. Returned outputs own their storage and survive
later workspace reuse. Initialization failures propagate for explicitly
requested backends; never switch collectives after a rank-local forward error.

The supported block-FP8 `apply_into()` path writes GEMM output directly into
caller-owned symmetric storage, removing the intermediate D2D copy before
reduction. Unsupported direct-output plans still compute a temporary result
and copy it into `out`; accepting an output buffer alone does not guarantee
copy elimination. FlashInfer A2A preserves the BF16 inputs; activation
quantization remains part of the ordinary FP8 GEMM path.

For each `[7168, 12288]` block-FP8 projection, weights and checkpoint scales
occupy about 84.02 MiB per GPU at TP1 versus 21.01 MiB at TP4. These figures
exclude prepared scales, communication buffers and the rest of the model.
Additional saved memory may increase serving capacity, but the maximum safe
concurrency still depends on cache length, graph pools and temporary kernels.

## Small tests first

Run from the repository root using the venv's interpreter, not a
container-global `torchrun` executable:

```bash
python -m pytest test/runtime/distributed/test_kimi_k3_o_proj.py -q
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --module test.runtime.distributed.validate_kimi_k3_o_proj
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  --module test.runtime.distributed.validate_kimi_k3_o_proj \
  --model <real-model-path> --large-tokens
```

The pytest file covers configuration and integration contracts. The validation
runner checks distributed numerics against TP1 and an FP32 reference,
CUDA-graph replay, empty ranks, and output lifetime across workspace reuse.
Use `--large-tokens` for backend-boundary checks, including fallback to NCCL.
Shared helpers provide mapping, weight loading, and projection construction.

The real-weight test loads only representative KDA/MLA output projections,
not the full checkpoint. For a 16-GPU test, replace `--standalone` with
`--nnodes=4 --node-rank=<node-rank> --master-addr=<head-node>
--master-port=<test-port>` on each node. Give tests their own rendezvous port.

The harness validates complete projection operations, including packing,
all-to-all, GEMM and reduce-scatter; it does not measure performance.
Small correctness tests may run while baseline deployment is being prepared
if memory headroom permits.

## Full-model checks

Wait for HTTP readiness before issuing requests. Keep speculation disabled
for the initial comparison. Save exact input IDs, generation parameters,
token counts, outputs and source/dependency versions.

- Check single requests, balanced multi-rank load and intentionally uneven
  load. Include a single active rank; empty peers still enter collectives.
- Pin each multi-turn conversation to the same attention DP rank and verify
  actual cached-token counts. Test prefill, decode and incremental prefill.
- For explicit rank coverage, use the scheduler gRPC `GenerateRequest` field
  `data_parallel_rank`. Do not assume an identically named HTTP JSON field is
  forwarded by the gateway. Check scheduler logs for actual rank ownership.
- Compare deterministic outputs and generated-token log probabilities before
  interpreting performance. Enable output log probabilities at server startup
  and request them in the client. This is not a full-logit comparison: verify
  which fields the runtime returns rather than assuming requested top-k or
  prompt log probabilities are implemented. Divergent generation needs a
  common-prefix comparison or a separate teacher-forced evaluation.
  The current gRPC validator requires `top_logprobs_num=0`.
- Measure repeated unprofiled TTFT, inter-token latency, throughput and memory
  with identical workloads. Separate loading/warmup from measured intervals.
- Capture short CUDA/NVTX NSYS ranges containing prefill and several decode
  steps. Label baseline and projection-TP4 reports clearly and package a ZIP.

Record measured results and limitations in the job's local runbook. This guide
does not claim that full-model validation or a speedup has already passed.

## Shared-expert TP4 experiment

Shared-expert sharding is independent of attention output projection. To compare
it against unchanged DEP16, keep attention projections unsharded in both runs:

```bash
export TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE=1
# Baseline: 1; shared-expert TP4: 4.
export TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE=4
```

Use attention TP1/DP16, routed MoE TP1/EP16, PP1. The shared MLP stays on the
existing auxiliary-stream path. Shared AllGather overlaps local routing;
dispatch waits for a gather-only event, not the following shared GEMMs.
Shared GEMMs may overlap dispatch. Routed BMM waits for shared GEMMs;
auxiliary ReduceScatter waits for dispatch, then overlaps routed BMM. Combine
waits for ReduceScatter. This prevents peer-polling shared and routed
collectives from starving each other of SM resources, without overlapping
the two GEMM paths. Gate/up and down weights are sharded directly;
AllGather and ReduceScatter restore token ownership without an intermediate
AllToAll. Up to 128 padded tokens per rank use TRT-LLM one-shot collectives;
larger shapes use NCCL. See [the design](../design/shared-expert-tp.md).

Run `test/runtime/distributed/validate_kimi_k3_shared_expert_tp.py` with 16
distributed workers and explicit `--model MODEL_DIR --layer 1` before measuring
the full model. It checks real BF16 shared-expert weights from the NVFP4
checkpoint, uneven/empty peers, auxiliary-stream CUDA graphs and the fallback
boundary. It does not validate full-model generation.

C64 per rank means 1,024 active requests across DEP16. First verify cache
admission at that capacity. A failed capacity check is not an E2E measurement;
neither a shared-MLP microbenchmark nor fewer active requests substitutes for it.
