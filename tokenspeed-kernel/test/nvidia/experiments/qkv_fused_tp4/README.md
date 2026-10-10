# Fused TP4 projection experiment

This experiment combines owner-side BF16 quantization, FP8/scales exchange,
blockwise GEMM, remote TMA output stores and a final local output copy in one
kernel per rank. It preserves caller-owned outputs and uses device-resident
64-bit epochs across CUDA graph replays.

The selected and frozen baseline kernels live in
[`thirdparty/cute_dsl/fused_tp4_projection`](../../../../python/tokenspeed_kernel/thirdparty/cute_dsl/fused_tp4_projection).
The selected kernel also has an opt-in
[registered runtime interface](../../../../python/tokenspeed_kernel/ops/communication/README.md).
The baseline retains its original zero-group quantization convention for timing
comparison; the selected kernel matches the aligned reference quantizer's
epsilon clamp and the one-shot AllGather's signed-zero handling.

The supported case is TP4 with equal 128-row owners, aligned widths, 128x128
GEMM tiles and a fully resident grid on isolated node-local GB300 GPUs.
Concurrent workspace use, uneven owners and arbitrary competing distributed
kernels are outside this experiment. The profile analysis expects four GPUs
with 152 SMs each.

## Scheduling changes

- Initialize independent GEMM pipelines before waiting for remote input.
  Only the TMA and scale producers acquire their CTA's required input owner.
  M-major scheduling keeps that owner unchanged across persistent tiles.
- Release scale buffers after their values reach registers. Release each
  partial accumulator buffer after its last tensor-memory read completes,
  before the remaining FP32 scaling and accumulation. The epilogue also
  releases its final accumulator buffer immediately after the last read.
- Use 64-bit BF16 input loads, a hardware FP32 warp-max reduction, and
  statically unrolled 32-column accumulator transfers. The epilogue keeps
  64-column transfers and subtiles.
- Use two scale stages and two epilogue stages to retain six A/B stages.
  Accumulator, epilogue and producer register budgets are 256, 128 and 64.
- Have four lanes in CTA zero acquire the independent final peer
  acknowledgements concurrently. Other warps can retire once their copies
  are published. The kernel remains active until those polling lanes finish,
  so stream ordering prevents the next call from reusing scratch early.

System publication fences, async-proxy fences and the final cross-rank
consumption acknowledgement remain part of the protocol. Shared memory and
tensor memory still limit the selected schedule to one CTA per SM.

These changes follow two ideas in DeepGEMM's MegaMoE implementation:
[restrict final NVLink synchronization to SM zero](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/comm/barrier.cuh)
and [release TMEM after its last read, before epilogue processing](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh).
The projection keeps its existing FP32 scale arithmetic and caller-owned
outputs. Register prefetch, remote completion notifications and wider output
copies were also tested; the selected schedule retains the simpler unrolled
accumulator loop and 128-bit output copies.

## Validation and timing

Use the repository's existing NVIDIA environment, including the runtime,
CuTe DSL and checkpoint loader dependencies. The checkpoint adapter loads
attention layers zero and three through the runtime loader; the benchmark
measures the layer-zero QKV/GB projection. Supply a compatible checkpoint
directory containing its configuration, safetensors index and weight shards.

Run from the repository root on an allocated four-GPU node:

```bash
source .venv/bin/activate
QKV_TEST=tokenspeed-kernel/test/nvidia/experiments/qkv_fused_tp4
QKV_KERNELS=tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cute_dsl/fused_tp4_projection
QKV_RESULTS=outputs/qkv-fused-tp4-new
QKV_CHECKPOINT=/path/to/compatible/checkpoint
mkdir -p "$QKV_RESULTS"
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$QKV_TEST/benchmark.py" --model "$QKV_CHECKPOINT" \
  --baseline-kernel "$QKV_KERNELS/baseline_kernel.py" \
  --kernel "$QKV_KERNELS/optimized_kernel.py" --tile-m 128 \
  --exchange tma --trace --output "$QKV_RESULTS/result.json"
```

The benchmark checks exact FP8 bits, FP32 scales and BF16 outputs on every
one of 16 changing-input graph replays, including zero groups, signed zeros,
very small and large finite inputs, and repeated BF16 mantissas. It checks
retained outputs, deliberate rank launch skew, epochs crossing 2^32 and one
kernel per call on every rank. Keep result files and profiler traces local.
An additional graph queues 32 calls with changing inputs and rank-dependent
device delays, then replays three times and checks every retained output.
This exercises scratch reuse without host synchronization between calls.

Timing compares the frozen fused baseline, selected kernel and current TP1
and TP4 runtime projections. Five alternating samples use 20 calls per graph
and ten graph replays per sample. Each sample takes the maximum across ranks;
the report takes their median. Compilation, validation and profiling are
outside these timing samples. This is projection validation, not end-to-end
model accuracy or serving performance validation.
For incremental optimization, snapshot the current selected kernel and pass
that snapshot as `--baseline-kernel`; record both source hashes. The bundled
baseline remains the older frozen schedule.

## Nsight Compute and IKET

For Nsight Compute, make `ncu` available on `PATH`. Its application replay
instruments rank zero while three uninstrumented peers run the collective.
Kernel replay cannot snapshot the symmetric peer allocations. The last node
of a three-call graph absorbs initial host launch skew.

```bash
CUTE_DSL_LINEINFO=1 python "$QKV_TEST/profile_pair.py" --tool ncu \
  --baseline "$QKV_KERNELS/baseline_kernel.py" \
  --candidate "$QKV_KERNELS/optimized_kernel.py" \
  --model "$QKV_CHECKPOINT" --output-dir "$QKV_RESULTS/profile"

python "$QKV_TEST/instrument.py" --source "$QKV_KERNELS/baseline_kernel.py" \
  --output "$QKV_RESULTS/coarse_baseline_kernel.py"
python "$QKV_TEST/instrument.py" --source "$QKV_KERNELS/optimized_kernel.py" \
  --output "$QKV_RESULTS/coarse_optimized_kernel.py"
python "$QKV_TEST/profile_pair.py" --tool iket \
  --baseline "$QKV_RESULTS/coarse_baseline_kernel.py" \
  --candidate "$QKV_RESULTS/coarse_optimized_kernel.py" \
  --model "$QKV_CHECKPOINT" --output-dir "$QKV_RESULTS/profile"

for variant in baseline optimized; do
  ncu --import "$QKV_RESULTS/profile/ncu-$variant.ncu-rep" --page raw --csv \
    > "$QKV_RESULTS/profile/ncu-$variant.csv"
  ncu --import "$QKV_RESULTS/profile/ncu-$variant.ncu-rep" --page source \
    --print-source sass --csv > "$QKV_RESULTS/profile/ncu-$variant-source.csv"
done
python "$QKV_TEST/analyze.py" --root "$QKV_RESULTS/profile"
```

Run the captures sequentially. Instrumentation adds nine markers and checks
that removing them recovers the kernel source. Analysis checks all 7,296
warp timelines per version for complete, ordered markers, and checks Nsight
for dropped samples or overflow. It emits per-warp CSVs with role labels,
combined phase spans, stall samples, sampled instruction addresses, launch
resources and static local-memory instruction counts.

The instrumentation regression check runs without GPUs:

```bash
python "$QKV_TEST/test_instrument.py"
```

IKET may log optional CUDA compatibility and eager-warmup instrumentation
warnings. Use the analyzed graph nodes only after the completeness checks
pass. Marked spans are elapsed intervals, not exact idle cycles; deferred
barriers can move waiting into the following interval. Compare setup plus
GEMM roles together and compare the complete output tail. Independent phase
medians do not add to a total median. Nsight sample fractions are not latency
fractions, and long-scoreboard samples in synchronization helpers do not
establish an HBM bandwidth bottleneck.
With early retirement, most warps finish before CTA zero's polling warp.
Compare that completion warp and each process's last exit as well as the
all-warp summaries; median warp lifetime is not kernel latency.
