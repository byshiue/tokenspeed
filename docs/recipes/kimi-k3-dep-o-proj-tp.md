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

To opt into fused NVLink A2A on supported single-node projection groups, also
set `TOKENSPEED_KIMI_K3_O_PROJ_A2A_BACKEND=auto` on every node. Use
`flashinfer` instead to fail startup when NVLink initialization is unavailable,
or `nccl` (the default) for the reference path. The FlashInfer installation
must provide `flashinfer.comm.ulysses.UlyssesCommunicator`; older versions
can use the NCCL fallback without an upgrade.

Only balanced physical batches of at most 16 rows per rank use the optional
path. Larger and uneven batches retain NCCL, including when `flashinfer` is
requested. No mode-specific prefill/decode branch is added. Graph-padded rows
count toward the limit. Startup logs report the selected backend and fallback
reason. Keep the same setting throughout graph capture and replay.

This changes projection A2A only: GEMM and NCCL ReduceScatter stay unchanged.
The earlier 16-requests/rank prototype reduced complete projection latency
by about 13.8%, not full-model latency. Full-model correctness and performance
validation are still required before changing the default.

To also test the optional ReduceScatter path, set:

```bash
export TOKENSPEED_KIMI_K3_O_PROJ_RS_BACKEND=triton_rsag
```

Set it on every node before startup. Unset or `nccl` keeps the existing
reduction. Enable FlashInfer A2A as above to use RSAG. The NVIDIA RSAG path requires
NVLink multicast support, BF16 projection outputs, and output widths
divisible by eight. Explicit initialization failures are fatal.

RSAG is used for balanced physical batches of up to 16 rows per rank when
FlashInfer A2A is active. Larger or uneven physical batches use NCCL for both
operations; padded graph rows may still contain inactive requests. If A2A
falls back to NCCL at startup, RSAG is also disabled and a warning is logged.
TP4 with output width 7168 reserves up to 896 KiB of symmetric payload scratch
per GPU, shared across layers of that width. Returned outputs are cloned so
subsequent layers cannot overwrite them.

The conservative cutoff avoids measured regressions when combining NCCL A2A
with RSAG at some batch sizes. Keep complete-projection timing in the
comparison when evaluating a wider threshold.

The earlier standalone TP4 experiment reported a 14–17% complete-projection
latency reduction, but timed single-operation graph replays and included
submission gaps. Do not treat that result as an isolated reduction speedup;
use chained-graph measurements for backend selection. Its output differed
from NCCL by about 0.36% relative L2,
despite slightly lower error against an FP32 reduction reference. Compare
full-model logits and generation before treating the two backends as
interchangeable; NCCL stays the default.

The copy-free TP4 candidate uses:

```bash
export TOKENSPEED_KIMI_K3_O_PROJ_A2A_BACKEND=flashinfer
export TOKENSPEED_KIMI_K3_O_PROJ_RS_BACKEND=triton_peer
```

For prepared block-FP8 projections, this path fuses quantization into A2A at
16 physical rows per rank. Smaller batches retain BF16 exchange because
shared-buffer measurements did not show a gain. The complete projection
also reuses the next A2A entry barrier as its reduction-storage reuse fence;
standalone reduction diagnostics still include an explicit trailing fence.
See the performance notes for paired timings and the TP1 memory tradeoff.

It borrows the A2A receive buffer and lets supported FP8 GEMMs write directly
into symmetric reduction storage. The final output is owned, not a workspace
view. The same balanced 1–16 physical rows/rank cutoff applies; larger or
uneven shapes retain NCCL. Standalone 16-GPU BF16 and real-weight projection
checks pass; full-model logits and generation still need validation, so the
backend remains opt-in.

This setting changes neither attention cache ownership nor EP placement.
Unset/`1` preserves the original projection. The value must divide world size
and projection dimensions, respect quantization alignment, and agree on all
ranks. Projection TP requires attention TP1/DP-world, linear-attention TP1,
MoE EP-world and PP1.

## Small tests first

Use the venv's interpreter, not a container-global `torchrun` executable:

```bash
python -m pytest test/runtime/distributed/test_kimi_k3_o_proj.py -q
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  test/runtime/distributed/test_kimi_k3_o_proj.py --benchmark
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  test/runtime/distributed/test_kimi_k3_o_proj.py \
  --model <real-model-path> --benchmark
```

The real-weight test loads only representative KDA/MLA output projections,
not the full checkpoint. For a 16-GPU test, replace `--standalone` with
`--nnodes=4 --node-rank=<node-rank> --master-addr=<head-node>
--master-port=<test-port>` on each node. Give tests their own rendezvous port.

The harness compares complete projection operations, including packing,
all-to-all, GEMM and reduce-scatter. Run final timing measurements without a
concurrent server workload on those GPUs. Small correctness tests may run
while baseline deployment is being prepared if memory headroom permits.

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
