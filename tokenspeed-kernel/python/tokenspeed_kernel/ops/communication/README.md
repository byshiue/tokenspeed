# Fused TP4 column projection

`cute_dsl_fused_tp4_projection` combines owner-side BF16-to-FP8 quantization,
input exchange, block-FP8 GEMM and inverse all-to-all in one GPU kernel per
rank. It writes a caller-owned BF16 output, which remains valid across later
layers and graph replays.

The registered schedule supports four node-local GB300 GPUs with 152 SMs,
128 physical rows on every rank, 7168 input channels and 49664 padded output
channels. Weights are contiguous E4M3 shards with FP32 128×128 block scales.
The grid must remain fully resident; this schedule is intended for isolated
GPUs and serialized collective calls. It does not provide a progress guarantee
for arbitrary competing distributed kernels.

`create_fused_tp4_projection_state(group, device)` allocates symmetric storage,
compiles and warms the kernel collectively before graph capture. Sequential
layers share one state and pass their weights at each call. Separate models
or concurrent streams require separate states. Destroy graphs before calling
`state.close()` collectively.

The runtime's `AutoBackend` prepares this optional state when
`TOKENSPEED_FUSED_TP4_PROJECTION=1`. The setting must agree across peers;
batch-invariant collective mode disables it. `DPColumnParallelLinear` uses
physical owner counts, including CUDA-graph padding, to select the fusion.
Unequal or empty owners, other row counts, other layouts, and other backends
use the existing projection operations. The flag defaults to disabled.

Run `test/runtime/distributed/test_fused_tp4_projection.py` on four GB300s to
check registered dispatch, exact comparison with the ordinary projection,
multiple layer weights sharing a workspace, changing-input graph replay,
retained outputs, delayed peers, 64-bit epochs and release/reprepare. These
tests validate projections; full-model workflow checks are separate.
