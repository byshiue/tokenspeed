# Experimental fused TP4 projection

`optimized_kernel.py` contains the selected CuTe DSL prototype;
`baseline_kernel.py` is the frozen earlier schedule for comparison.
Both derive from NVIDIA CUTLASS's blockwise GEMM example and retain its BSD
license and source attribution.

The kernel fuses BF16-to-FP8 quantization, input exchange, column-sharded
blockwise GEMM, remote output stores and the copy into caller-owned outputs.
It requires equal 128-row owners, TP4, aligned dimensions and a fully resident
grid on isolated node-local GB300 GPUs. It has no runtime registration.

See the [validation and profiling runbook](../../../../../test/nvidia/experiments/qkv_fused_tp4/README.md)
for the launcher, exact numerical and replay checks, timing methodology and
Nsight Compute/IKET commands.
