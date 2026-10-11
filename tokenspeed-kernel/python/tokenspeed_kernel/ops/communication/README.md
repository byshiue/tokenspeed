# Fused TP4 column projection

`cute_dsl_fused_tp4_projection` combines owner-side BF16-to-FP8 quantization,
input exchange, block-FP8 GEMM and inverse all-to-all in one GPU kernel per
rank. It writes a caller-owned BF16 output, which remains valid across later
layers and graph replays.

The registered schedule supports four node-local GB300 GPUs with 152 SMs,
7168 input channels and 49664 padded output channels. Physical row counts may
be arbitrary or unequal within the prepared capacity, including empty owners.
Weights are contiguous E4M3 shards with FP32 128×128 block scales.
The grid must remain fully resident; this schedule is intended for isolated
GPUs and serialized collective calls. It does not provide a progress guarantee
for arbitrary competing distributed kernels.

`create_fused_tp4_projection_state(group, device, max_rows)` allocates symmetric storage,
compiles and warms the kernel collectively before graph capture. Sequential
layers share one state and pass their weights at each call. Separate models
or concurrent streams require separate states. Destroy graphs before calling
`state.close()` collectively.

Each call supplies all four physical `owner_rows` in group-rank order. The
kernel uses a common power-of-two bucket, at least 32 rows, covering the largest
owner. Its input staging masks invalid loads and writes zero padding directly
to the exchanged FP8 storage. Remote TMA stores use the bucket layout; the
final copy writes only each owner's actual rows. No separate padding kernel is
launched. Empty owners still compute their weight shard and join every epoch.

All buckets needed to cover the capacity are compiled and warmed before capture,
sharing symmetric allocations and epochs. Actual local lengths are runtime
parameters, so changing counts within a bucket does not trigger compilation.
Smaller buckets share 128-row GEMM tiles; larger ones use several tiles per
owner. The fixed grid advances every CTA's epoch even when it has no GEMM work,
so graph calls can alternate buckets safely.

The runtime's `AutoBackend` prepares this optional state when
`TOKENSPEED_FUSED_TP4_PROJECTION=1`. The setting must agree across peers;
batch-invariant collective mode disables it. `DPColumnParallelLinear` uses
physical owner counts, including CUDA-graph padding, to select the fusion.
Selection uses the subgroup maximum, so every peer chooses the same path even
when some owners are empty. The conservative measured policy selects fusion
at 64 rows, from 88 through 256 rows, and from 384 through 1024 rows. For larger
unmeasured buckets, the largest owner must fill at least 75% of the bucket.
Smaller or heavily padded cases use ordinary TP4. Preparation skips fused state
when capacity is below 64. Direct registered-kernel calls support every nonempty
group within capacity. Other layouts and backends
use the existing projection operations. The flag defaults to disabled.
Eager forwards choose from the current counts. CUDA graph capture records the
choice for each physical row bucket, so replay adds no host dispatch.

Run `test/runtime/distributed/test_fused_tp4_projection.py` on four GB300s to
check registered dispatch, exact comparison with the ordinary projection,
multiple layer weights sharing a workspace, mixed-row changing-input graph replay,
retained outputs, delayed peers, 64-bit epochs and release/reprepare. These
tests validate projections; full-model workflow checks are separate.

The experiment's `benchmark_rows.py` compares explicit registered
fusion and automatic dispatch against TP1 and ordinary TP4 using real weights. See
[`qkv_fused_tp4`](../../../../test/nvidia/experiments/qkv_fused_tp4/README.md)
for invocation and timing methodology.
