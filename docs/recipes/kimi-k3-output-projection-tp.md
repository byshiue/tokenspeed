# Kimi-K3 attention-projection TP validation

This experimental path independently shards QKV and output projections. With DEP16,
attention/cache ownership remains TP1/DP16 and routed experts remain TP1/EP16.
Keep shared-expert TP disabled when measuring this feature independently.

```bash
export TOKENSPEED_KIMI_K3_SHARED_EXPERT_TP_SIZE=1
export TOKENSPEED_KIMI_K3_O_PROJ_TP_SIZE=4
export TOKENSPEED_KIMI_K3_QKV_PROJ_TP_SIZE=4
export TOKENSPEED_O_PROJ_A2A_BACKEND=flashinfer
export TOKENSPEED_O_PROJ_RS_BACKEND=triton_peer
```

Unset both projection TP variables or set them to 1 for the replicated baseline.
Each variable can also be enabled independently.
The O-projection path redistributes channel shards, runs a row-parallel Linear,
then reduces complete outputs back to the original token owners. It does not
change attention/cache sharding. Allocate persistent workspaces before graph capture.

QKV uses AllGather → column-parallel GEMM → A2A. The A2A restores complete
channels to the original token owners before convolution, normalization or
attention. KDA shards its fused Q/K/V/output-gate/decay-down/beta projection;
MLA shards its fused QKV-A/output-gate and Q-B projections. MLA's absorbed KV-B
weights and KDA's decay-up projection stay unchanged. Weights and 128×128
scales load as contiguous output-channel shards, with zero tail padding to
keep each shard block-aligned. No checkpoint codes are requantized.

The QKV path currently requires block-scaled FP8 attention weights and BF16
activations (including the NVFP4 model recipe with FP8 attention projections).
Attention and linear attention must remain TP1, attention DP and routed MoE EP
must equal world size, and pipeline parallelism must be 1. Projection TP must
divide world size. TP4 is the validated model configuration in this recipe;
other sizes also need compatible collective backends and checkpoint alignment.
Each subgroup must be CUDA-IPC/NVLink accessible. Startup agrees settings across
ranks before creating groups; unsupported checkpoint formats fail explicitly.

QKV gathers use TRT-LLM one-shot through 128 physical rows/rank, then NCCL.
Its inverse FlashInfer A2A handles up to 512 physical rows/rank, then NCCL.
Uneven and empty owners participate with padding to their subgroup's maximum
physical count; only wholly empty subgroups skip communication. CUDA graph
padding counts, not live request counts, determine collective sizes.
Sequential same-shape QKV projections share scratch allocated before cache
budgeting/capture, separate from O-proj and auxiliary-stream MoE buffers.

Backend alternatives:

- A2A: `nccl` or `flashinfer`. FlashInfer requires compatible NVLink peers,
  equal positive physical row counts, and at most 512 rows/rank; other batches
  use the padded NCCL path.
- Reduction: `nccl`, `triton_peer` (up to 8192 rows/rank), or
  `trtllm_lamport` (up to 128 rows/rank). Above the selected fast-path bound,
  use NCCL. The Lamport path uses the generic explicit-state TRT-LLM wrapper.

For standalone real-weight correctness, reserve a persistent 16-GPU allocation
with `salloc`, then launch four workers per node through `srun` or a site submit
wrapper, keeping each contiguous TP4 group within one node:

```bash
python -m torch.distributed.run --nnodes=4 --nproc-per-node=4 \
  --node-rank=NODE_RANK --master-addr=HEAD_NODE --master-port=PORT \
  -m test.runtime.distributed.validate_kimi_k3_o_proj --model MODEL_DIR

python -m torch.distributed.run --nnodes=4 --nproc-per-node=4 \
  --node-rank=NODE_RANK --master-addr=HEAD_NODE --master-port=PORT \
  -m test.runtime.distributed.validate_kimi_k3_qkv_proj --model MODEL_DIR --rows 128
```

Inspect checkpoint tensors rather than inferring projection precision from the
model name: the real Kimi-K3 NVFP4 checkpoint used by this harness stores these
attention output weights in FP8 with 128x128 block scales. The validator loads
representative KDA and MLA projections, compares against the replicated
projection and a dequantized FP32 reference, and exercises uneven/empty owners,
output lifetime, and repeated graph replay.

For performance, time the entire operation, including packing, quantization,
A2A, GEMM and reduction. Compare identical local token counts and real weights,
warm up both variants, alternate measurement order, and report repeated graph
timings without a profiler. A standalone projection result is not a full-model
latency or dataset-accuracy result.

### Experimental fused AllGather and FP8 quantization

The kernel-level `TrtllmAllGatherQuantState` and
`trtllm_allgather_fp8_quantize` APIs combine BF16 Lamport AllGather with
128-element FP8 activation quantization. Each ready group is quantized directly
into the FP8 values and MN-major FP32 scales consumed by the prepared GEMM.
Communication remains BF16. This removes the gathered-BF16 output write/read
and the separate quantization launch; it does not reduce network traffic.

This prototype is **not selected by the model runtime**. The normal QKV path
still gathers BF16, then quantizes it. There is no PDL or overlapping scale
consumer: the next GEMM waits for the fused kernel to complete normally.
The existing RMSNorm-fused AllGather is a different operation and is not used.

Create the state collectively before CUDA-graph capture. Pass an explicit
positive `num_blocks` no larger than the device's SM count. Inputs must be
contiguous BF16 with a width divisible by 128 and equal physical row counts
of 1..128 across the subgroup. Empty owners participate with zero padding.
Calls sharing a state must be serialized, and returned buffers are borrowed
until the next fused call. The state retains the ordinary BF16 output buffer
for reference testing, so this prototype does not yet reduce allocated scratch.

Quantization matches the active prepared-scale path, including the native
TRT-LLM small-value clamp for aligned rows and the existing Triton padding
contract otherwise. Check exact FP8 bytes and scales when changing quantizer
dependencies. TP2/TP4 have GPU correctness coverage; the template dispatch also
accepts TP8/TP16, which require their own multi-GPU validation before use.

Run the four-GPU test inside a persistent allocation:

```bash
python -m pytest -q \
  tokenspeed-kernel/test/nvidia/ops/communication/test_trtllm_allgather_quant.py
```

It covers eager calls, repeated graph replay with changing inputs, batch-size
changes, signed zeros, tiny/large finite values, padded/empty owners, subgroup
isolation, and interchange with ordinary AllGather on the same workspace.

### Full-model validation

For DEP16 with FlashInfer MoE transport, place all 16 GPUs within one compatible
NVLink fabric, not merely each projection TP4 subgroup. Reserve the nodes with
`salloc`, then launch one server process per node through `srun` or the site's
submit wrapper. Keep the allocation for both variants.

The following serving settings use the full checkpoint, no layer override,
no speculation, and a capacity of 16 requests/rank (256 total). They are a
smaller integration workload, not a substitute for full-model C128 admission.

```bash
python -m tokenspeed.cli serve \
  --model MODEL_DIR --load-format safetensors \
  --served-model-name kimi-k3-nvfp4 --trust-remote-code --language-model-only \
  --dtype bfloat16 --quantization nvfp4 --kv-cache-dtype fp8 \
  --world-size 16 --nprocs-per-node 4 \
  --attn-tp-size 1 --data-parallel-size 16 --dense-tp-size 1 \
  --moe-tp-size 1 --expert-parallel-size 16 \
  --dist-init-addr HEAD_NODE:PORT --host 0.0.0.0 \
  --attention-backend tokenspeed_mla --kda-backend cutedsl_kda \
  --moe-backend flashinfer_trtllm --all2all-backend flashinfer \
  --max-model-len 4096 --max-num-seqs 256 \
  --chunked-prefill-size 1024 --max-prefill-tokens 1024 \
  --prefix-granularity 128 --enable-prefix-caching --disable-kvstore \
  --max-cudagraph-capture-size 16 --cudagraph-capture-sizes 1 2 4 8 16 \
  --disable-cuda-graph-padding --disable-autotune \
  --gpu-memory-utilization 0.83 --seed 1 --enable-output-logprobs \
  --policy cache_aware --dp-aware
```

For the reference, set both projection TP variables to `1`. For the TP4
comparison, set both to `4`; select `trtllm_lamport` explicitly if measuring
the short-message reduction path instead of the default `triton_peer`.
Keep shared-expert TP at `1` and all other settings identical. Use explicit
request-to-rank affinity, fixed input token IDs, and greedy generation.
Exercise cold prefill, decode, and a subsequent request that reuses the cached
prefix. Record active counts, cache hits, complete latency, TTFT and inter-token
latency; separate warmup from measured rounds.

Numerical checks should compare identical contexts. If generated strings
diverge, score fixed reference prefixes as well: different generated tokens
change later inputs and cannot establish a projection error by themselves.
Row-parallel BF16 accumulation need not be bitwise identical to the replicated
GEMM. Report projection error against both TP1 and a higher-precision reference;
a serving smoke test is not a dataset-accuracy result.

Before increasing to C128/rank, confirm cache admission and actual active counts.
If the full model does not fit, report the memory limit rather than silently
substituting a reduced-layer model.
