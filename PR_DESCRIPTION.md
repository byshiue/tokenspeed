# feat: add TRT-LLM CuTe-DSL FP8 dense GEMM support

## Summary

Add a shared `--dense-gemm-backend auto|trtllm_cutedsl` option for standard
128x128 block-FP8 dense linears, including Kimi-K3 KDA QKV/gates and output,
and MLA QKV-a/gate, Q-b, and output projections.

```bash
tokenspeed serve /path/to/model --dense-gemm-backend trtllm_cutedsl
```

Omit the option or use `auto` to preserve existing selection. The option is
independent of `--moe-backend`. BF16/NVFP4 linears, MXFP8, per-tensor FP8,
specialized grouped projections, routed experts, and communication are unchanged.

Unlike the earlier DeepGEMM implementation, this backend preserves the original
FP8 weight codes and FP32 scales. There is no weight requantization or E8M0
conversion. Activations use the existing 1x128 FP8 quantization path; accumulation
is FP32 and output is BF16.

## Implementation

- Reuse the existing FP8 linear loading and execution path across models.
- Give Kimi-K3's merged KDA projection the same prepared-plan contract.
- Vendor TRT-LLM's blockwise CuTe-DSL kernel under `tokenspeed-kernel/thirdparty`,
  retaining upstream licenses and provenance; expose it through the registry.
- Compile three dynamic-shape variants during preparation before CUDA-graph
  capture. Select a tile by token count and pass the current stream at launch.
- Remove the Kimi-K3-specific environment flag and the added DeepGEMM
  requantization API. Existing DeepGEMM support elsewhere is unchanged.

## Previously collected GEMM-only performance

These are the earlier independent backend-benchmark results, **not measurements
of this commit's integrated runtime**. That harness selected the fastest of
three validated CuTe-DSL configurations for each shape; this implementation
uses a fixed token-count heuristic, so the numbers are not a performance promise.

Environment: NVIDIA GB300, PyTorch 2.13.0+cu130, FlashInfer 0.6.18,
CUTLASS DSL 4.7.1, real layer-0 checkpoint FP8 weights, and BF16 outputs.
Each number is the median of 14 samples across two fresh processes. Each sample
times a CUDA-graph replay containing 200 GEMMs.

**Quantization, scale preparation, compilation, and communication are excluded.**
C is requests per rank, one decode token per request. DEP16 uses M=C; TP4 uses
M=4C. These are local GEMM shapes, not distributed or end-to-end measurements.
QKV TP4 uses N-sharding after padding; output projection uses K-sharding.

### KDA QKV + gates

Latency in microseconds; reduction is relative to CUTLASS.

| C/rank | Mapping | Shape (M, N, K) | CUTLASS | TRT-LLM CuTe-DSL | Latency reduction |
|---:|---|---|---:|---:|---:|
| 32 | DEP16 | (32, 49408, 7168) | 67.12 | 57.04 | 15.0% |
| 64 | DEP16 | (64, 49408, 7168) | 66.19 | 57.67 | 12.9% |
| 128 | DEP16 | (128, 49408, 7168) | 67.75 | 62.14 | 8.3% |
| 32 | TP4 | (128, 12416, 7168) | 18.23 | 13.83 | 24.1% |
| 64 | TP4 | (256, 12416, 7168) | 33.37 | 26.38 | 20.9% |
| 128 | TP4 | (512, 12416, 7168) | 49.19 | 43.11 | 12.4% |

### KDA output projection

Latency in microseconds; reduction is relative to CUTLASS.

| C/rank | Mapping | Shape (M, N, K) | CUTLASS | TRT-LLM CuTe-DSL | Latency reduction |
|---:|---|---|---:|---:|---:|
| 32 | DEP16 | (32, 7168, 12288) | 22.02 | 20.08 | 8.8% |
| 64 | DEP16 | (64, 7168, 12288) | 27.21 | 20.35 | 25.2% |
| 128 | DEP16 | (128, 7168, 12288) | 27.32 | 19.67 | 28.0% |
| 32 | TP4 | (128, 7168, 3072) | 9.84 | 7.54 | 23.4% |
| 64 | TP4 | (256, 7168, 3072) | 10.11 | 7.76 | 23.2% |
| 128 | TP4 | (512, 7168, 3072) | 17.83 | 12.74 | 28.5% |

There is no universal backend winner: FlashInfer TRTLLM-Gen measured
17.99/18.44 microseconds for DEP16 output at C32/C64, faster than CuTe-DSL's
20.08/20.35 microseconds.

## Integrated projection measurements

The production prepared-linear path was also measured on GB300 with real
checkpoint weights and seeded synthetic BF16 activations. These timings
**include online activation quantization and scale-layout copies**, but exclude
communication, attention, and the rest of the model. They must not be compared
directly with the GEMM-only tables above.

Each result is the median of five CUDA-event measurements of a graph containing
50 projection calls, following eager and graph warmup. TP4 denotes representative
local sharded shapes, not a distributed test.

| C/rank | Mapping | QKV default → CuTe-DSL (µs) | Output default → CuTe-DSL (µs) |
|---:|---|---:|---:|
| 32 | DEP16 | 69.86 → 59.33 | 24.84 → 23.46 |
| 64 | DEP16 | 68.96 → 60.29 | 29.83 → 23.75 |
| 128 | DEP16 | 70.56 → 64.82 | 31.76 → 25.48 |
| 32 | TP4 | 21.43 → 17.47 | 12.31 → 9.75 |
| 64 | TP4 | 38.20 → 31.68 | 13.17 → 11.25 |
| 128 | TP4 | 57.01 → 46.97 | 21.56 → 17.97 |

All 12 cases matched the default output exactly for the tested inputs.
Latency reductions ranged from 8.1–18.5% for QKV and 5.5–20.8% for output.
This is a single paired run, not a full-model speedup or an exhaustive tuning study.

## Numerical validation and limitations

In the prior backend benchmark, CuTe-DSL outputs matched CUTLASS exactly for
all tested inputs in both runs. This establishes the tested numerical behavior,
not a bitwise-equivalence guarantee for every input.

Seven kernel tests and ten selected CLI/runtime tests passed. All five
real-weight projection families matched the default output exactly at M=128
and passed CUDA-graph replay. The integrated backend's tests cover unchanged checkpoint weights/scales,
FP32 references using the actual quantized operands, multiple token counts,
CUDA-graph replay with changing inputs, padding, and CLI configuration.
Full-model task accuracy and end-to-end performance remain unvalidated.
The broader MLA cache tests require a scheduler extension rebuilt against
current main. An existing default QKV test also fails intermittently in the
test environment, including without the new backend selected.
