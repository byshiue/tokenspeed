# KDA buffered replay: implementation record

This record tracks implementation of [the buffered replay plan](kda-buffered-replay-plan.md).
It separates reference results from kernel tests and real-model measurements.
Passing a reference test does not mean buffered replay is available in serving.

## Source and milestones

Baseline: `2e4b540743878959eb51793ce01d74b5898b0e38` (upstream main).
Implementation branch: `kda-buffered-replay`.
M1 source commit: `2619669e51d7eeb438069964d042fd21a5bc0736`
(`test(kda): establish buffered replay reference and GPU prototype`).
M2 source commit: `3f3615e11bef462b2c1390ef29b289b612dc2f38`
(`feat(cache): add bounded live-state retention`).
M3 source commit: `24c69bf25ca09ab4417b7134593c6f8bebbaf60b`
(`feat(cache): keep replay history request-local`).
M4 source commit: `ac74d6785ee921efcd2b4c061e74a562b82ac47f`
(`feat(cache): reserve only replay history decode tails`).
M5 source commit: `f4312f1ce25a0ff95c78a97fc70bada0013ce8af`
(`feat(kda): bind cache-owned replay history and positions`).
M6 source commit: `bb330bf0a86152388254af7263a59cff0b191098`
(`feat(kda): run buffered recurrence through cache block tables`).
M7 source commit: `5a7bb231617d4c5e3feab467d444ad7cf5009523`
(`perf(kda): tile accepted history reconstruction`).
M8 source commit: `f7e780f15c595eec34501644e612ef55b6eddf7e`
(`feat(kda): unify commit entry and fuse native recurrence inputs`).
M9 source commit: `eb55b1dac8195dd6aca06238c018dd959c0bd0ca`
(`perf(kda): share replay metadata and stamp commits across layers`).
M10 source commit: `014c7a6f6b17ceba6fbfce6ef52cc78d31d079e5`
(`feat(kda): compose buffered conv and recurrent workspace`).
No default capacity or serving performance benefit has been established.
The GPU implementation remains an unregistered prototype, not the complete
serving feature. Eagle3 must run the new path without a performance regression
against the frozen baseline before the work meets its completion gate.

| Milestone | Status | Evidence / exit condition |
| --- | --- | --- |
| A. Recurrence, history representation, conv and acceptance reference | M1 CPU/GPU numerical gates passed | 19 CPU + 8 GPU cases; serving equivalence is a separate gate |
| B. LCM ownership, retention and lifecycle contract | M2–M5 cache foundations verified; M9 adds the metadata owner | Live endpoint materialization and handoff still pending |
| C. Unified GPU forward and commit | M10 composes conv/candidate capture, gate GEMM, recurrence and conv/stamp commit in one experimental workspace | Endpoint materialization and serving dispatch remain pending; recurrence not registered |
| D. Graphs, overlap and lifecycle integration | M10 isolated producer-stream pipeline tests pass in eager/graph; serving integration pending | Full-path prefill transitions, prefix reuse, overlap and recovery |
| E. Real-model correctness and performance | Not started | Matched TP8 NVFP4 comparisons, AIME 2026, capacity sweep and traces |

## Recording a result

Each validation entry must identify:

- Baseline and implementation commits. For uncommitted work, say so and retain
  a source patch plus its SHA-256 in local artifacts; never label it with HEAD
  alone as if the implementation were committed.
- Hardware, Python/PyTorch/CUDA and kernel dependencies, model/draft revisions,
  TP, precision, backend, graph/overlap settings and buffer capacity.
- Exact command, input/dataset revision, seed, output budget, acceptance policy,
  tolerances/scorer, repetitions and raw artifact names.
- Correctness result and performance result separately, including failures,
  profiler overhead and anything not tested.

Machine-specific paths, allocation identifiers and logs belong in local ignored
`outputs/kda-buffered-replay/` artifacts, not in a public PR. This document
records portable commands and summaries.

## Initial code audit

- Ordinary KDA decode writes recurrent state every step. Speculative KDA
  captures a single-window payload and replays accepted inputs at commit.
- The runner only invokes the state commit hook when a drafter exists. The
  new hook must cover ordinary decode without changing unrelated consumers.
- Sampling's returned accepted length already includes the target input;
  the current KDA commit helper clamps that count, without adding one. The
  comment describing it as only draft matches must not define the new ABI.
- Current snapshot retention follows token progress, not a lagging materialized
  checkpoint. Current prefix publication may publish an aligned accepted
  endpoint without separate materialization evidence. Both assumptions require
  deliberate cache-contract changes before buffered state enters serving.
- A sliding-history group participates in prefix matching and transfer today.
  Adding an empty-on-prefill replay group without addressing those policies
  would prevent valid prefix hits or expose uninitialized entries.

## Validation results

### A1: CPU recurrence and ring reference

Source: M1 commit above (tested before committing). Python 3.12.13,
PyTorch 2.8.0+cpu, pytest 8.4.2, Linux aarch64, one OpenMP/BLAS thread.
No weights or GPU are used. Command, from the repository root after activating
the test venv:

```bash
PYTHONPATH=tokenspeed-kernel/test OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest --confcutdir=tokenspeed-kernel/test/ops/attention \
  tokenspeed-kernel/test/ops/attention/test_kda_buffered_reference.py -q -s
```

Result: **19 passed**. The confcutdir isolates this torch-only CPU reference
from the kernel suite's platform fixture, which imports tokenspeed-triton.
An initial run without that isolation failed at conftest import because that
package is absent from the CPU venv; it did not execute any tests.

Coverage includes T=1/4, minimum and non-power-of-two capacities, conv width
1/4, rejected suffix poisoning, idle rows, repeated flush and ring wrap,
materialized endpoint reseeding, reordered request references, and the actual
per-layer TP8 shape (12 local heads, K=V=128). Request-object reordering is a
semantic check, not a GPU request-slot lifecycle or graph test.

FP32 gates were fixed before execution: atol=2e-5, rtol=2e-4, allowing different
FP32 reduction order. In a 256-round, 640-accepted-token weak-decay stress case,
the maximum absolute state error was 1.1920929e-7 for FP32 history versus
0.0229887664 for BF16 history, at each tested capacity (8,32,64). The BF16 case
deliberately rounds decay factors close to one; this rejects that storage
choice, not all possible mixed-precision or log-decay encodings.

Initial layout decision: store normalized K, correction U and multiplicative
per-channel decay in FP32. Keep only the current raw-QKV candidate window for
conv commit, not the whole raw-QKV history. Q is not retained. Reduced-precision
history is deferred; no tolerance was relaxed to accept it.

Byte estimate (not measured allocation): at 12 local heads, K=V=128 and
69 KDA layers, L=64 costs 77.625 MiB per live request per TP rank for K/U/decay
alone. The materialized recurrent checkpoint costs another 51.75 MiB. Conv,
candidate staging, metadata, allocator padding and protected storage are extra.
These are component sizes, not an estimate of total serving-memory regression.

### GPU primitive scope

The private Triton prototype consumes caller-owned FP32 checkpoint/history,
prepared normalized Q/K, V, decay and beta. One forward handles T=1/T>1 and
GPU-side per-row flush decisions; a second small kernel commits ring pointers.
It is portable recurrence groundwork, not the final tuned NVIDIA solution.
It is deliberately **not registered for serving** while the LCM and publication
contract is unresolved. Conv/gate fusion, absolute endpoint publication and
LCM field binding are not implemented by this prototype.

### A2: GPU recurrence and captured commit

Source: M1 commit above (tested before committing). One NVIDIA GB300,
driver 580.167.08, Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0,
tokenspeed-triton 3.8.10.post20260906, pytest 9.1.1. Cached container and
serving venv were reused, with a fresh persistent allocation and submit/srun.
No target or draft weights were loaded for these primitive tests.

```bash
PYTHONPATH=tokenspeed-kernel/python:tokenspeed-kernel/test \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  tokenspeed-kernel/test/ops/attention/test_kda_buffered_reference.py \
  tokenspeed-kernel/test/nvidia/ops/attention/test_kda_buffered_recurrent.py -q -s
```

Result: **27 passed** (19 CPU and 8 GPU cases), with 15 dependency deprecation
warnings. GPU cases exercise T=1/4, minimum and non-power-of-two capacities,
eager/captured forward+commit, idle rows and partial/zero acceptance over 32
rounds. Every round checks outputs, checkpoint stores, reconstructed endpoint,
ring pointers and ignored padding. CPU stress error in this newer PyTorch
environment was 5.96046448e-8 (FP32) and 0.0229887664 (BF16).
An initial collection attempt failed on a duplicated docstring header in two
new files; it was corrected before this successful run.

### A3: serial reconstruction microbenchmark — regression, not a speedup

Environment as A2. Single layer/GPU, H=12 and K=V=128, prepared FP32 Q/K/V,
decay and beta, full acceptance, batches 1/8 and T=1/4. The baseline is the
unchanged `fused_recurrent_kda_pool` state-update kernel. It is **not** the
full speculative verify-plus-replay baseline. The prototype includes one
forward and one pointer-commit kernel per timed call; eventual model-wide
commit batching is not represented. Conv, gate projection, model layers,
LCM, communication and profiling are excluded.

```bash
PYTHONPATH=tokenspeed-kernel/python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tokenspeed-kernel/test/nvidia/ops/attention/bench_kda_buffered_recurrent.py \
  --output microbench.json
```

Timer: 32 calls per captured graph, 5 warmup graph replays, 50 timed graph
replays per sample, 5 samples; report median CUDA-event time per call. Seed 93.
Compilation and capture are excluded. Steady-state runs include capacity
flushes; they are not separate flush/no-flush latency measurements.

Repeated measurements on the final M1 source (microseconds; serial prototype):

| Batch / T | Existing recurrence | L=16 buffered | L=64 buffered |
| --- | ---: | ---: | ---: |
| 1 / 1 | 2.24 | 5.86 | 12.16 |
| 1 / 4 | 4.42 | 8.63 | 14.63 |
| 8 / 1 | 3.59 | 7.08 | 14.06 |
| 8 / 4 | 6.25 | 10.66 | 17.81 |

This prototype does not establish a useful default capacity. Serial history
reconstruction and commit overhead outweigh the avoided state stores in this
microbenchmark. A batched/fused commit and faster reconstruction must be
measured before serving activation; no full-model speedup is claimed.

The final formatted source repeated all **27 passing tests**. The mandatory
`pre-commit run --all-files` passed before the signed-off M1 commit. Earlier
hook runs reformatted only the new prototype/reference/test/benchmark files;
no existing serving source changed. Local artifacts retain both benchmark
runs, raw test logs, the environment, submit scripts and the source patch.

## B: cache integration work that remains

The current two-token state retention rule is unsafe for a lagging checkpoint:
with block span 128, a checkpoint at token 128 resides in slot 0. An accepted
endpoint at 140 with a four-token conservative scheduling lag can expire slot
0, although those twelve buffered tokens still depend on it.

The capacity rule bounds accepted history by `L - T_max`. The cache contract
must retain a state block covering that lag in addition to ordinary overlap
protection, or hold an explicit checkpoint reference until flush retirement.
The same bound must inform admission and victim/reclaim accounting; changing
only the final reclamation call leaves admission able to reclaim the source.

### M2: bounded live-state retention

Source: M2 commit above (tested before committing), based on M1 and its
validation record at `b9a7a5c39a1a4efeb1e43554bc228597414eac17`.

Added the explicit `max_state_lag_tokens` cache contract. The Python spec,
scheduler binding and C++ translation carry the same non-negative token bound;
history groups must pass zero. All current recipes explicitly pass zero, so
this milestone does not activate buffered replay or change serving state math.

Admission credit, eviction planning, in-place reserve checks and reclamation
share the updated expiry calculation. Prefix matching still needs one exact
snapshot, regardless of live-state lag. Capacity bounds add conservative
whole-block lag headroom before packing, retaining Kimi-K3's existing prefill
working-set allowance. Required-argument changes in other recipes and test
fixtures merely state their existing zero-lag policy.

For the proposed buffer policy, `L - T_max` is the maximum accepted history
length to declare at integration. This milestone adds only the retention
mechanism; no capacity CLI or history group has been connected yet.

The C++ build environment is Linux aarch64, GCC 13.3.0, CMake 4.4.3,
nanobind 3.0.1, tokenspeed-spdlog 1.15.1 and GoogleTest 1.14.0, Release build.
Runtime and GPU checks use the same GB300/CUDA 13.0/PyTorch 2.13.0 serving
environment as A2, with the newly built scheduler extension staged separately.
The existing persistent allocation and cached image/venv were reused; no
downloads or real target/draft weights were needed.

Validation results:

- Clean rebuild and complete C++ scheduler suite: **483 passed**. New cases cover retention at
  aligned/unaligned endpoints, large and int32-limit lags, unchanged snapshot
  matching, tight-pool admission with cached/uncached checkpoints, retirement
  after progress, invalid configurations and per-role capacity accounting.
- Scheduler bindings and runtime cache suites: **389 passed, 284 subtests**,
  repeated with the final clean-built extension.
  These include Kimi-K3 layout/budget/bridge tests, PD wire round trips and
  missing-field rejection, existing GDN GPU state-paging/oracle/continuation
  checks, Qwen cache groups and one-forward prefill scheduling.
- Buffered reference and GPU prototype: **27 passed**, including eager and
  CUDA-graph forward/commit. No numerical tolerances changed.

Build from the repository root in an activated build venv:

```bash
cmake -S tokenspeed-scheduler -B build/kda-buffered-replay-m2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DTOKENSPEED_SCHEDULER_BUILD_TESTS=ON \
  -DTOKENSPEED_SCHEDULER_BUILD_PYTHON=ON
cmake --build build/kda-buffered-replay-m2 -j 8
build/kda-buffered-replay-m2/tokenspeed_scheduler_tests --gtest_color=no
```

For runtime checks, import that freshly built extension with the current
`tokenspeed_scheduler` Python wrapper; do not use an older installed binding.
With its staging directory first on `PYTHONPATH`, append `python`,
`tokenspeed-kernel/python` and `test/runtime`, then run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest \
  tokenspeed-scheduler/python/tests \
  test/runtime/test_group_specs_from_layer_types.py \
  test/runtime/test_multi_window_page_counts.py \
  test/runtime/test_kimi_k3_cache_spec.py \
  test/runtime/test_kimi_k3_integration.py \
  test/runtime/test_cache_setup.py \
  test/runtime/test_cache_memory_plan.py \
  test/runtime/distributed/test_cache_pd_manifest.py \
  test/runtime/test_aligned_max_scheduled_tokens.py \
  test/runtime/test_state_checkpoint_prefill.py \
  test/runtime/test_event_loop_scheduler_stats.py \
  test/runtime/test_v4_sliding_window_groups_smoke.py \
  test/runtime/test_gdn_state_paging.py \
  test/runtime/models/test_qwen3_cache_groups.py -q --tb=short
```

The prototype command is unchanged from A2. Inputs are synthetic test cases
and the existing GDN oracle; these results are not an AIME or TP8 full-model
accuracy score. No performance comparison was run for M2, and the M1 serial
prototype regression remains unresolved. Current recipes still declare zero
lag, and no serving kernel or attention arithmetic changed.

Failures retained in the record: the first C++ build exposed two new test-code
errors (a const reference at registration and initializer order), both fixed.
The small CPU venv passed 200 tests but could not run Kimi recipe tests with
its missing runtime packages or a mixed PyTorch/torchvision installation.
The first container run passed 385 tests and failed four shared test-helper
imports; adding `test/runtime` to `PYTHONPATH` produced the passing run above.
Local ignored artifacts hold the commands, environment, successful logs and
the failed container log. The mandatory `pre-commit run --all-files` passed;
formatter changes are included in this milestone.

### M3: request-local history ownership and reuse

Source: M3 commit above (tested before committing), based on the M2 validation
record at `fcca5cabca832b42267db5e01a84e8b03264924d`.

Added `replay_checkpoint_group`: a sliding, per-token history group names the
state group that seeds it on resume. The Python declaration and scheduler
binding require the argument, with `None` preserving every current recipe.
Recipe/runtime and C++ validation reject missing or non-state dependencies,
ambiguous IDs and windows too short to cover the checkpoint's declared lag.

Prefix resume uses the existing zero-lookback matcher for this group, without
changing live retention. Device and Host hits install absolute-position holes;
the normal allocator supplies fresh suffix blocks owned by the resumed
request. Two requests sharing a prefix never share mutable replay entries.
History cannot constrain an otherwise valid exact-checkpoint hit, enter the
prefix index, participate in canonicalization or boundary residency, or be
queued for Host writeback. Direct Host publication is rejected as well.

This does not yet allocate KDA history fields or GPU request-position metadata.
Prefill uses ordinary sliding-group allocation; an empty-tail-only reservation
is not implemented. The new group contract is exercised with synthetic cache
consumers, not enabled by a model recipe. PD startup and wire decoding reject
these groups until endpoint materialization and handoff ordering are connected.
Ordinary PD and reusable groups keep their existing behavior.

Validation results:

- Complete C++ scheduler suite: **485 passed**, repeated after formatting and
  rebuilding. New cases cover Device/Host resume, both dependency orders,
  private suffix ownership, publication/canonicalization exclusion, retention,
  release, invalid dependencies and the explicit PD gate.
- Runtime and scheduler bindings: **391 passed, 297 subtests**, repeated with
  the final rebuilt extension. The final run took 18.20 seconds and reported
  27 dependency warnings. Existing GDN GPU, Kimi layout, prefill, ordinary PD
  and Qwen cache checks passed alongside the new contract cases.
- Buffered reference/GPU: **27 passed**, including eager/CUDA-graph execution,
  in 7.95 seconds. Numerical tolerances are unchanged.
- Minimal CPU spec suite: **40 passed, 2 skipped**. The two bridge imports are
  unavailable in that small venv; both cases passed in the runtime suite.

The environment and portable build/runtime commands are the same as M2.
A fresh persistent four-GB300 allocation was used under the same binding;
submit/srun reused the cached image and read-only serving venv. Tests ran on
one GPU, with no target/draft model or TP8 model execution. The initial runtime
run also passed; its extra warnings came from first-use compilation. The first
repository-hook run formatted the new edits; those changes are included.
The final `pre-commit run --all-files` passed before the signed-off source commit.
Exact commands, versions, source patch and raw logs are retained in ignored
local artifacts. No weights, packages or image were downloaded. No M3
performance, AIME or full-model NSYS result is claimed.

Required before runtime dispatch:

1. Bind KDA's history fields to the request-local LCM declaration. Initialize
   empty history on materialized resume and budget the actual field layout,
   candidates, overlap and prefill working set before choosing a capacity.
2. Own request-position metadata through the cache contract, with stable GPU
   views and reset/rebind rules. Runtime batch position is not request identity.
3. Separate accepted progress from materialized checkpoint progress. Resolve
   current physical blocks from the live tables, including canonicalization;
   do not retain stale physical addresses after prefix deduplication.
4. Order endpoint materialization before prefix publication/transfer and before
   retraction sources can be reused. Published checkpoints remain immutable.
5. Generalize target commit to ordinary decode while preserving GDN/PLE/QSA;
   integrate conv commit and prepared gate inputs into that same operation.

**Serving remains unchanged.** Capacity CLI, GPU cache binding, lifecycle tests,
real-weight TP8 performance, AIME 2026 and full-model NSYS are still pending.

### M4: empty-history prefill allocation

Source: M4 commit above (tested before committing), based on
`e6e99fbf8ab1665a7d0d9d1ea2ba05681768b66f` (M3 validation record).

This milestone closes an allocation prerequisite before KDA field binding.
Prefill materializes an exact state and starts empty replay history, so reserving
history rows for an entire prefill chunk would waste storage. Local admission
now advances replay tables with absolute holes and reserves only the final
chunk's decode window. Intermediate aligned chunks acquire no replay blocks.
Ordinary sliding attention still reserves and writes its prefill extent.

The existing sparse allocator now accepts an aligned, empty suffix without
inventing writable capacity in a null block. It advances the known-empty
prefix without rescanning earlier chunks. Allocation remains atomic across
groups, and decode uses the existing dense admission and reclamation path.

Python capacity planning and the C++ single-request bound exclude prefill rows
for replay groups. Both still account for retained history, candidates, overlap
and partial blocks. They also include the `T-1` tokens between the conservative
decode reclamation frontier and the accepted endpoint; this guard is needed
even at overlap depth zero. This is a storage bound, not a longer logical
history or a change to flush policy.

No model recipe enables replay history yet, and no attention kernel, weight
format or numerical tolerance changes in M4. GPU field binding, request-position
metadata, endpoint materialization and serving dispatch remain pending. The M1
serial reconstruction regression is still unresolved; this allocation change
is not evidence of a serving speedup.

Validation commands and logs are retained in ignored
`outputs/kda-buffered-replay/m4/`. The build and runtime environments follow M3;
the same persistent allocation, cached image and read-only serving venv are
reused, with the freshly rebuilt extension staged in M4's artifact directory.
No target/draft weights are loaded, and there is no TP8 model run, AIME result
or NSYS capture for this cache-only milestone.

Validation results:

- Full C++ scheduler suite: **487 passed**. The consolidated scenario covers
  replay block spans 1/2/4, decode widths 1/3, overlap-depth configurations 0/1,
  chunked/one-forward prefill, Device prefix reuse, aligned/unaligned endpoints,
  partial acceptance, bounded decode residency and finish-time release. This
  is cache scheduling evidence, not GPU metadata or in-flight overlap validation.
  The final rebuild also verifies underfunded startup rejection. Allocator
  checks cover all-hole advancement, failed-admission atomicity, partial-tail
  backing, rejection of an unbacked reserve and int32-limit geometry.
- Runtime and scheduler bindings: **392 passed, 313 subtests**, 27 dependency
  warnings. Initial/final runs took 28.99/30.30 seconds; these are test durations,
  not a performance comparison. The new budget case covers spans 1/2/4/128, T=1/4,
  both overlap depths and prefill budgets 128/8192; ordinary sliding budgets
  remain unchanged. Existing Kimi, GDN GPU, prefill, Qwen and PD checks pass.
- Buffered reference/GPU: **27 passed**, 15 dependency warnings, 6.97 seconds.
  These repeat the existing eager/captured prototype tests, not buffered serving.
- Small CPU page-budget suite: **7 passed**. This repeats the pure math cases
  without the serving-container dependencies.

The final formatted source was rebuilt and repeated all **487 C++** and
**392 runtime** tests with the restaged scheduler extension. The mandatory
`pre-commit run --all-files` passed before the signed-off source commit;
the earlier formatter changes are included. Raw logs, the source patch,
environment and checksums are retained in the local runbook.

The remaining field/metadata work is unchanged. In particular, none of these
tests proves that a lagging checkpoint may be published at the accepted
endpoint: materialization and lifecycle ordering must be integrated first.

### M5: KDA history fields and cache-owned positions

Based on `2a62f8b8c585689526a8ee61a5490ada35da7b1f` (M4 validation record).
This milestone binds real KDA history fields through the existing recipe,
packing and arena pipeline. It does not enable buffered serving: the factory
still passes no replay capacity, and the KDA backend explicitly rejects a
buffered layout until materialization and commit are integrated.

Each of the three state groups gains a request-local history group. Its
per-layer fields are FP32 normalized K, correction U, multiplicative decay,
and an int64 checkpoint-position stamp. The fixed physical block span is eight
tokens, independent of capacity and prefix identity. A typed pool accessor
returns zero-copy arena views, honors layer-load fences and validates complete
field ownership. Pipeline/draft views retain their existing layer windows;
replay groups do not become paged-attention router leaves.

The position stamp records `c+1` at the last committed input row. Resolving it
through the current block table gives `h=e-c`, without a persistent table keyed
by batch slot. Stamps are owned per layer, preserving cache lifecycle and PP
ownership; the future runtime refresh should compute batch positions once and
share those outputs across layers. Unregistered GPU prepare/commit primitives
now cover this contract with caller-owned buffers, explicit strides and no
host readback. They handle idle rows, partial acceptance, and restamping after
a flush with zero acceptance. Invalid rows report failure and suppress stores.
Zero stamps only seed empty history after exact-endpoint materialization and
fresh-page zeroing; they must not silently recover lost live history.

At TP8 the existing 24 planes and 20.25 MiB parent remain unchanged. Six
eight-token history blocks fit per parent. Stamps use the spare 24th plane;
placing them inline would reduce packing to five. TP16 also keeps its original
parent size. TP1/2/4 round to 96/48/24 MLA pages per parent plane instead of
89/45/23; each MLA page retains its exact byte stride. Existing non-buffered
layouts are unchanged. Initial 16-token packing exceeded TP4's padding budget;
an eight-token attempt then exposed a whole-plane divisibility constraint.
The final planner accounts for both, without relaxing the padding limit.

The logical K/U/decay payload at TP8, 69 KDA layers and L=64 is still 77.625 MiB
per live request per rank. Stamps add 34.5 KiB before packing. These are payload
sizes, not peak serving memory: the planner also reserves state lag, candidates,
overlap and rounded blocks. Existing verify-workspace budgeting remains in
place; the final buffered-forward workspace and commit descriptors are pending.

One planner example: TP8, B=1, T=4, L=64, 4096-token capacity and overlap depth
zero needs 24 usable parents, or 506.25 MiB including the null parent. The
non-buffered plan needs 15 usable parents, or 324 MiB. This bound includes the
additional state-lag reservation and rounded history storage, not just payload.
It is a reproducible planning result, not a measured serving-memory comparison.

Validation used one GB300 in the existing persistent allocation, the cached
container and read-only serving venv. Python 3.12.3, PyTorch 2.13.0+cu130,
CUDA 13.0, driver 580.167.08, tokenspeed-triton 3.8.10.post20260906 and pytest
9.1.1. No target/draft weights were loaded. The scheduler extension is the
unchanged M4 build, staged separately; there are no C++ edits in M5.

- Cache/runtime/scheduler regression selection: **501 passed, 317 subtests**,
  28 dependency warnings, 19.57 seconds initially and 18.45 seconds on the final
  source. The nine new cases cover TP1/2/4/8/16,
  L=8/37/64, T=1/4, budget inversion, unchanged prefill independence, actual
  arena aliasing/zeroing, PP/draft windows, fences, invalid layouts and the
  serving guard. Existing Kimi, GDN/Qwen, GLM, PD, pool and router checks pass.
- Buffered reference/GPU selection: **34 passed**, 15 dependency warnings,
  7.18 seconds initially and 7.75 seconds on the final source. Seven new cases
  check strided paged metadata, 64 reordered
  rounds, eager/CUDA graphs, idle/padding, acceptance, flush and invalid stores.
  An initial test pattern did not produce zero acceptance at a flush; the test
  now forces that case. Recurrence tolerances and kernels are unchanged.
- Complete unchanged C++ scheduler binary: **487 passed, 134 suites**.

These durations describe test execution, not a performance comparison. M1's
serial recurrence regression is unresolved. No M5 serving throughput, TP8
real-weight accuracy, AIME score or NSYS result is claimed. Full commands,
planner estimates, environment, source patch and logs are retained in ignored
`outputs/kda-buffered-replay/m5/`. The first repository-hook run formatted seven
files; those edits are included. The final `pre-commit run --all-files` passed
before the signed-off source commit, and both GPU/runtime selections passed
on that final source. Address arithmetic widens raw table offsets before
multiplication. The local runbook records the source patch and artifact hashes.

Next: integrate paged recurrence and shared runtime position refresh, then
materialization/publication ordering and ordinary/speculative commit. Capacity
CLI/defaults and real-model validation remain gated on that work.

### M6: paged recurrence and position commit

Based on `68ecf326dfd6ee9305d35a96a6470d4281ae30c9` (M5 validation record).
The unregistered GPU prototype now reads/writes strided LCM field views through
current raw history and state tables. It replaces the dense per-request ring
prototype; the independent CPU recurrence reference is unchanged. Standard and
multi-token input windows use the same forward and position-commit sequence.

The kernel reads state at `c`, reconstructs accepted history `[c,e)`, and writes
candidate K/U/decay at `[e,e+width)`. A capacity flush stores `S_e` before any
candidate update; a non-flush round stores no full state. Current block tables
resolve both checkpoint positions, including across state-block boundaries and
after physical remapping. The null checkpoint at `c=0` is implicit zero state,
not a read from the arena's null block. Rejected suffixes remain uncommitted.

A GPU backing check validates the complete history range, checkpoint source,
flush destination and position consistency before recurrence stores. Invalid
rows leave state/history/output untouched and clear validity. This deliberately
adds a launch: the prototype sequence is position prepare, backing check,
recurrence, then stamp commit. Validity handling in serving is still pending.
The caller must preserve LCM ownership and stream ordering; valid block ids do
not prove that two cache groups may occupy the same physical parent.

Validation runs on a fresh persistent four-GB300 allocation under the same
binding, using submit/srun and the cached container. One GPU is used, with no
target/draft weights. Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0,
driver 580.167.08, tokenspeed-triton 3.8.10.post20260906, pytest 9.1.1, aarch64.
The scheduler binary/extension remain the M4 build; there are no C++ edits.

Initial validation:

- **36 reference/GPU tests passed**: 19 CPU reference, 10 paged recurrence and
  invalid-backing cases, and seven position-metadata cases. The recurrence
  matrix retains T=1/4, minimum/non-power-of-two capacities and eager/graph
  coverage, expanded to 64 rounds with absolute holes, block reuse, physical
  state remapping, simulated request reuse and poisoned rejected suffixes.
  History spans 3/8 and state spans 16/128 are independent. These are isolated
  lifecycle simulations, not proof of scheduler overlap or prefix publication.
- **501 runtime tests and 317 subtests passed** after fixing an arena test
  fixture. Its first version assigned state and history to the same LCM parent,
  violating allocator ownership and causing overlapping writes. The corrected
  fixture assigns separate parents. No numerical tolerance was relaxed.
- **487 scheduler tests, 134 suites passed** using the unchanged binary.

The microbenchmark now measures a fixed round at empty, no-flush and flush
history lengths, including all four prototype launches. Endpoints do not
advance during timing, and source/destination state blocks are separate so a
flush cannot change the next sample's input. This is not the M1 rolling-ring
benchmark, an amortized rollout, or a serving comparison. Baseline timing is
prepared eager recurrence only; model, conv/gates, scheduler, publication and
baseline speculative replay-commit costs are excluded. The final isolated sweep
and formatted-source results are recorded below.

Final-source validation repeated **501 runtime tests and 317 subtests** (28
warnings, 19.46s) and **36 reference/GPU tests** (15 warnings, 13.34s). The actual
arena test now continues through a history-block boundary and a capacity flush.
The first repository hook run formatted four files; those edits are included.
The final `pre-commit run --all-files` passed before the signed-off source commit.

The isolated timing sweep ran only after correctness steps completed, without
a profiler. It covers 48 cases: B=1/8, T=1/4, L=2*T/16/32/64 and three history
phases. Each median uses five samples of 50 graph replays, with 32 calls per
graph, after four call warmups and five graph warmups. Full samples, source
patch, environment and hashes are retained in the local M6 runbook.

L=64 results, microseconds per fixed round:

| Batch | T | Prepared eager recurrence | Paged, h=0 | Paged, h=L-2*T | Paged flush, h=L-T |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 2.24 | 8.00 | 27.11 | 27.77 |
| 1 | 4 | 4.42 | 10.30 | 28.34 | 28.61 |
| 8 | 1 | 3.58 | 8.39 | 32.05 | 32.95 |
| 8 | 4 | 6.22 | 12.28 | 31.77 | 34.95 |

**This prototype regresses against prepared eager recurrence.** It avoids full
state stores on non-flush rounds, but the four launches and serial history
reconstruction outweigh that saving in this test. The timing split does not
isolate each launch's contribution. It is not a full-model comparison or an
amortized capacity recommendation, and it does not establish a speedup over M1.
Reducing launch overhead and reconstruction cost remains a gate before serving
activation, alongside the lifecycle work below.

Production recipe/backend guards remain unchanged. Paged recurrence is not yet
dispatched by the model. Shared runtime refresh, conv/gate preparation, unified
ordinary/speculative commit, exact-endpoint materialization/publication and
overlap/transfer/retraction integration remain pending. No real-weight TP8,
AIME or full-model NSYS result is claimed.

### M7: tiled accepted-history reconstruction

Based on `dbca788740b769e2ac8a0db94688250f0998f5d8` (M6 validation record).
Replace token-at-a-time replay of accepted history with tiles of up to eight
tokens (bounded by the capacity's maximum committed history).
For each tile, compute the exclusive reverse product of per-channel decay,
then reconstruct `S' = S * product(D) + U^T @ (K * suffix(D))` with FP32
reductions. Shift decay before the scan instead of dividing by it, so zero
decay remains valid. The formula still builds the full recurrent state;
this is not the deferred output-only route.

The compute tile is independent of the history block span and checkpoint
granularity. Raw block tables and field strides still resolve every history
load. Candidate tokens retain their sequential dependence. Flush, candidate
writes, acceptance, four-launch ordering and non-flush store suppression are
unchanged; the kernel needs no additional workspace. Standard and speculative
windows continue through the same operation.

The initial tensor-core version passed the numerical gate but regressed at
batch 8. A compiled 16-by-16 variant used 255 registers and spilled; increasing
warps or reducing pipeline stages did not resolve the timing tradeoff. The
selected FP32 reduction uses 32 value rows, up to eight history rows, four warps and
one pipeline stage. A local geometry sweep measured both batch sizes before
selecting it. No TF32/BF16 history conversion is retained.

The existing parameterized recurrence test now also covers L=2*T+56 over 256
rounds with weak decay, identity/zero-decay channels and repeated tile/flush
boundaries. T=1/4, eager/graph, reordering, simulated allocation reuse,
rejection poisoning and exact-state checks remain in that same test. The
FP32 gate stays atol=2e-5, rtol=2e-4; no tolerance is widened.

Validation uses the existing persistent GB300 allocation via submit/srun,
one GPU, no weights. Environment: Python 3.12.3, PyTorch 2.13.0+cu130,
CUDA 13.0, driver 580.167.08, tokenspeed-triton 3.8.10.post20260906,
pytest 9.1.1, aarch64. The M4 scheduler binary/extension is unchanged.
The final capacity-bounded source passed **40 reference/GPU tests** (15 warnings,
44.74s) and **501 runtime tests and 317 subtests** (28 warnings, 18.53s), with
unchanged tolerances. Its T=4/L=64 compiled kernel uses 128 registers, zero
spills and 5,120 bytes shared memory. The exact `pre-commit run --all-files`
passed before the signed-off M7 source commit. The local runbook retains its
source patch, environment, commands, complete samples and artifact hashes.

The isolated before/after sweep reloads M6's exact archived kernel and uses
the unchanged fixed-round benchmark on the same GPU, sequentially and without
a profiler. It covers 48 cases: B=1/8, T=1/4, L=2*T/16/32/64 and empty,
no-flush and flush history. Each median retains five samples of 50 graph
replays with 32 calls per graph, after warmup. Both sides include position
prepare, backing validation, recurrence and stamp commit.

L=64 results, microseconds per fixed round:

| Batch | T | Phase | M6 serial | M7 tiled | Latency reduction |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | no flush, h=62 | 27.10 | 17.61 | 35.0% |
| 1 | 1 | flush, h=63 | 27.76 | 17.98 | 35.2% |
| 1 | 4 | no flush, h=56 | 28.34 | 17.87 | 37.0% |
| 1 | 4 | flush, h=60 | 28.60 | 19.70 | 31.1% |
| 8 | 1 | no flush, h=62 | 32.05 | 22.69 | 29.2% |
| 8 | 1 | flush, h=63 | 32.96 | 23.52 | 28.6% |
| 8 | 4 | no flush, h=56 | 31.77 | 24.32 | 23.5% |
| 8 | 4 | flush, h=60 | 34.95 | 26.78 | 23.4% |

This is not an across-the-board improvement: 38 of 48 measured medians are
lower, while T=1/L=2 includes regressions up to 6.7% (batch 8 flush,
9.66 to 10.31 us). Empty-history results at L=64 range from 5.2% faster to
1.2% slower. All samples and exploratory configurations remain in the local
runbook; the table must not stand in for the full sweep.

The prototype still costs more than the benchmark's prepared eager recurrence,
which excludes the original speculative replay commit. These measurements
neither establish Eagle3 serving performance nor select a default capacity.

Runtime integration remains the next gate: shared position refresh, conv/gate
preparation and commit, ordinary/speculative commit dispatch, and exact endpoint
materialization before publication or handoff. Serving guards remain enabled.
Real NVFP4 TP8 Eagle3 performance, AIME 2026 and full-model traces are pending.

### M8: shared commit hook and native recurrence inputs

Source commit: `f7e780f15c595eec34501644e612ef55b6eddf7e` (signed off), based on
`96921c09c9cb0efb9cee7d3ded6e8b4c85df7339` (M7 validation record). Tests and
timing below used this source before committing. The exact
`pre-commit run --all-files` command passed after formatter changes.
The runner now calls `commit_state_after_verify` after successful decode/mixed
execution with or without a drafter. Graph outputs are sliced to live requests
before this call. Ordinary decode supplies acceptance one through the same
interface; no token is added. Failed forwards, pure prefill and idle execution
do not commit. Hybrid/Qwen consumer routing is unchanged, and existing
consumers without staged state return without issuing GPU work. The old
speculative-only hook name is removed, including misleading accepted-count
comments. This changes the shared runner, not the control-plane event loop.

The private recurrence can now read positive-stride BF16 Q/K/V views from a
packed conv(+SiLU) output, BF16/FP32 raw gate and beta logits, and FP32
`A_log`/`dt_bias`. Q/K normalization, gate-to-decay conversion and sigmoid beta
are computed inside the recurrence, without separate prepared-input tensors or
launches. Both softplus and bounded gates follow the existing KDA equations.
History and reconstructed state stay FP32. Prepared FP32 inputs remain a
reference/benchmark input contract, not a separate ordinary-decode path.

The runner hook is integrated; **buffered KDA serving remains disabled**.
Native inputs still need the runtime's shared metadata and conv producer/commit
wiring. The primitive does not grant checkpoint publication provenance or solve
endpoint handoff ordering. No real-model Eagle3 performance claim follows from
these changes.

The existing tests now cover width-one and speculative commit fan-out, live-row
slicing after graph replay, failure ordering, and real unstaged GDN/KDA/PLE/QSA
no-ops. Native inputs join the existing multi-round recurrence matrix rather
than adding separate tests for each transform: T=1/4, minimum/unaligned/larger
capacities, eager/graph, strided packed views, weak decay, rejection, reuse and
flush. The first native round also compares outputs and full state to the
existing GPU recurrence. FP32 state/history tolerances remain atol=2e-5,
rtol=2e-4. Native BF16 outputs add their predeclared half-ULP rounding allowance
(`torch.finfo(torch.bfloat16).eps / 2`) to relative error; FP32 tolerances are
not widened.

Initial results on the same persistent GB300 allocation, one GPU and no weights:
**104 commit/consumer/unified-path tests plus 62 subtests passed** (23 warnings,
24.53s), and **64 reference/GPU tests passed** (15 warnings, 119.08s), including
FP32 raw gates and old-kernel state parity. Environment remains Python 3.12.3,
PyTorch 2.13.0+cu130, CUDA 13.0, driver 580.167.08,
tokenspeed-triton 3.8.10.post20260906, pytest 9.1.1, aarch64. Scheduler sources
and the M4 binary/extension are unchanged. Final validation and timing follow
below once complete.

The microbenchmark now requires `--input-kind prepared` or `--input-kind native`.
Native timing includes in-kernel input transforms on both sides but excludes
conv/gate producers, model execution and the original batched replay commit.
Only prepared timings are directly comparable to M7's prepared-input sweep.
This remains an isolated four-launch prototype test, not the Eagle3 gate.

Final-source checks repeat **64 reference/GPU cases** (15 warnings, 125.92s),
**104 commit/consumer cases plus 62 subtests** (23 warnings, 11.68s), and
**501 runtime cases plus 317 subtests** (28 warnings, 19.44s). Input token
offsets are widened before stride multiplication, like cache addressing; the
invalid-backing cases also check offsets beyond signed int32 without allocating
a multi-gigabyte tensor; both focused eager/graph cases passed separately
(15 warnings, 0.45s). The unchanged runtime/consumer results precede this
kernel-only address refinement; the 64-case numerical rerun follows it.

A geometry sweep selected 8 value rows, up to 4 history rows and one warp for
native K=V=128, T=1/4 at minimum capacity L=2*T. Other cases retain M7's wider
tile. This is a compile-time geometry choice, not a separate decode/commit path.
Prepared/native timing sweeps ran sequentially after numerical checks, using
the same five-sample graph timer as M7, without a profiler.

Final native T=4 results, microseconds per fixed round:

| Batch | Capacity | Empty history | No flush, h=L-2*T | Flush, h=L-T |
| --- | --- | --- | --- | --- |
| 1 | 8 | 10.44 | 10.27 | 11.14 |
| 8 | 8 | 12.55 | 12.55 | 14.35 |
| 1 | 64 | 10.83 | 18.79 | 20.41 |
| 8 | 64 | 13.63 | 26.11 | 28.64 |

The native eager-recurrence reference takes 7.49 us at batch 1 and 9.29 us at
batch 8; it is **not** the original Eagle3 verifier plus batched commit.
Prepared-input timings range from roughly unchanged to 7.8% slower than M7
over the 48 cases. These results do not establish a no-regression result or a
default capacity. Shared per-group metadata must remove repeated per-layer
work, and the actual native pipeline must be measured after integration.

### M9: shared group metadata and multi-layer stamp commit

Source commit: `eb55b1dac8195dd6aca06238c018dd959c0bd0ca` (signed off), based
on `b47041e823e25252e8d94f60b658db318d413fad` (M8 validation record).
No serving or full-model result yet.

`KDAReplayMetadata` binds local replay groups from the cache pool and allocates
only execution scratch: raw table stacks, common endpoint/width vectors and
per-group positions. History and checkpoint stamps remain LCM-owned. All
layers in a group reuse one cached metadata view. Runtime capacity sizes the
buffers, independent of which smaller batches have graphs. Table refresh uses
the existing ratio-one fill and rejects malformed live delivery without making
temporary contiguous tables. Pool replacement requires a new owner/recapture.

Range validation is now an explicit preparation operation before recurrence;
it checks the entire group's read/write span before any layer stores. The
recurrence consumes the validated positions in one launch. One commit launch
stamps every local layer in that group after all data stores complete. Reading
one representative layer at the next prepare relies on this shared-commit
invariant, not on assuming independent layer states happen to agree. This
changes launch placement, not recurrence arithmetic or tolerance.

The new parameterized runtime fixture covers all 69 KDA layers in three groups,
width one/four, request reordering, padding/idle, group-specific flush decisions,
invalid acceptance, graph capture at batch two with eager batch five, stable
addresses, and a replacement pool. Distinct LCM parents back different groups;
zeroing uses owned child pages, not a whole overlaid field view. An initial
fixture incorrectly cleared other groups' stamps; it was corrected without
changing the implementation or weakening assertions. This fixture checks
metadata, not recurrent/conv payload execution or checkpoint publication.

Buffered serving remains disabled. Conv producer/commit wiring, exact endpoint
materialization, failure feedback and publication ordering still gate dispatch.
The original Eagle3 full-model performance, AIME and trace requirements remain
unchanged. Validation results follow below.

Initial checks on a new persistent GB300 allocation, one GPU, no weights:
**13 cache/metadata tests passed** (22 warnings, 13.76s), **64 reference/GPU
tests passed** (15 warnings, 136.04s), and **505 runtime tests plus 317 subtests
passed** (31 warnings, 53.46s). The software environment remains Python 3.12.3,
PyTorch 2.13.0+cu130, CUDA 13.0, driver 580.167.08,
tokenspeed-triton 3.8.10.post20260906, pytest 9.1.1, aarch64. C++ sources and
the staged scheduler extension are unchanged. The final fixture also checks
non-unit table strides, shorter live acceptance and PP-local field binding.

Isolated position preparation, backing validation and stamp commit for 69
layers in three groups take **13.82 us at batch 1** and **14.60 us at batch 8**,
down from **308.94 / 321.78 us** when repeating the prototype protocol per
layer. This reduces 207 metadata launches to nine. Timing excludes raw-table
refresh, recurrence, conv/gate producers and model execution; the per-layer
control is **not the original Eagle3 implementation**.

The unchanged four-launch recurrence timer also compares archived M8 kernels
against M9 on the same GPU, sequentially and without a profiler. Across 48
cases per input representation, current/M8 median ratios are 0.99965–1.00058
for prepared inputs and 0.99902–1.00109 for native inputs: no material isolated
recurrence regression observed. Both use five samples of 50 graph replays,
32 calls per graph after warmup. This does not establish the full Eagle3
no-regression gate or justify a default capacity. The native T=4/L=8 flush
case is 11.197 us at batch 1 and 13.905 us at batch 8, including all four
prototype launches, not just the recurrence kernel.

Final-source reruns passed **13 cache/metadata cases** (22 warnings, 15.97s),
**64 reference/GPU cases** (15 warnings, 125.44s), and **505 runtime cases plus
317 subtests** (28 warnings, 21.86s). These include the final PP binding and
shorter-live-acceptance checks. The exact `pre-commit run --all-files` command
passed after formatting. Validation preceded the signed-off source commit with
the same code; source patches, environment, raw samples and hashes are retained
in the local milestone runbook. There is still no real-model accuracy or
Eagle3 throughput result for buffered serving.

### M10: conv capture, accepted windows and the shared forward workspace

Source commit: `014c7a6f6b17ceba6fbfce6ef52cc78d31d079e5` (signed off),
based on `291935c9fd0cbe5c76b15836f77824df8d7fb5b0` (M9 record).

The four-tap BF16 conv producer now captures raw candidates in the same kernel
that produces conv outputs. The later commit gathers the accepted raw suffix
and previous window, then writes every local layer's conv window in one GPU
launch. It loads a complete channel window before an in-place store. Conv
addresses follow accepted endpoint `e`, independently of recurrent checkpoint
`c`; possible destinations are validated once per group before layer work.
This adds one preparation launch per group to M9's metadata protocol. M9's
nine-launch metadata-only timing does not describe this expanded protocol.

`KDAReplayWorkspace` composes conv/capture, low-rank gate GEMM, native-input
recurrence and accepted conv/stamp commit. Gate and conv producers use the
existing fork/join stream protocol; recurrence waits for both. Width one and
four use the same code. Raw candidates are per-layer transient storage;
conv/gate/output scratch is shared and must be consumed before the next
layer. Long-lived state and history still belong to LCM. Recipe budgeting
includes all owned tensor storage, even for ordinary width-one decode, and
tests compare that budget against the actual full-model workspace allocation.
PP-local workspaces count only their bound layers and groups.

Initial validation used one GB300, no weights, in the M9 software environment:
Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0, driver 580.167.08,
tokenspeed-triton 3.8.10.post20260906, pytest 9.1.1, aarch64. The persistent
allocation, cached container, commands and raw logs remain in the local
milestone runbook. No C++ source or staged scheduler binary changed.

The conv-only fixture passed **8 cases** (15 warnings, 4.76s): BF16 outputs
are bitwise equal to the original GPU conv producer, and accepted windows
are bitwise equal to a CPU suffix reference. It covers width one/four, small
and TP8 per-layer dimensions, eager/graph, 32 rounds, request reordering,
padding, zero acceptance, block crossings, strided storage, in-place writes
and invalid backing/acceptance.

The combined cache/runtime suite passed **19 cases** (22 warnings, 35.17s).
Its six new pipeline cases use TP8 per-layer geometry, capacities 8/37,
width one/four, two live requests plus padding, and six PP-local layers
crossing two cache groups. They run the real producer streams and packed
LCM fields for 32 rounds, compare outputs and reconstructed accepted state
against independent CPU recurrence, and poison rejected raw candidates before
commit. They also check zero-acceptance flushes, different conv/recurrent
block slots and stable workspace addresses. FP32 tolerances remain
`atol=2e-5, rtol=2e-4`; BF16 outputs retain M8's half-ULP relative allowance.
The first fixture chose a PP window wholly inside one group while expecting
three; correcting it to cross a real group boundary required no implementation
or tolerance change.

Serving is still gated. Exact endpoint materialization, publication/failure
feedback and prefill/transfer/retraction handoff are not implemented by this
workspace. These local tests do not establish model accuracy or the original
Eagle3 no-regression requirement. Matched full NVFP4 TP8 runs, AIME 2026 and
short before/after NSYS reports remain required.

Isolated conv timing uses the same GPU and five samples of 50 graph replays,
32 calls per graph, without a profiler. The following medians cover all 69
layers at TP8 per-layer dimensions; they are not full model-step timings:

| T=4 conv operation, microseconds | Batch 1 | Batch 8 |
| --- | ---: | ---: |
| Original conv producer, without raw capture | 100.03 | 127.48 |
| Original producer plus a separate raw copy per layer | 179.83 | 257.07 |
| New fused producer and raw capture | 116.09 | 159.80 |
| Original batched conv-window commit | 3.09 | 11.78 |
| New batched conv-window commit | 3.08 | 12.16 |

Fusion saves time against the producer-plus-copy composition, but adds work
relative to the producer alone. The new batch-eight commit is about 0.38 us
slower. Timing excludes table preparation, gates, recurrence, stamps and
endpoint handoff. Sources/destinations are fixed private same-block states
with full acceptance; this isolates the kernels, not the original Eagle3
round. Width-one samples are retained in the local raw report. No end-to-end
speedup is claimed. At context 65,536, batch four and width four, the complete
buffered workspace reserves **11,208,428 bytes per TP8 rank** (about 10.69 MiB),
excluding cache-owned state/history and CUDA stream/event overhead.

Final-source checks passed **72 reference/GPU cases** (15 warnings, 119.37s),
**19 cache/runtime pipeline cases** (22 warnings, 22.45s), and **511 runtime
cases plus 317 subtests** (28 warnings, 34.43s). The last suite includes the
pipeline cases. The exact `pre-commit run --all-files` command passed after
formatting. Final tests also verify stable pointers by rereading the workspace
attributes, so retaining old tensor references cannot hide a rebound buffer.
