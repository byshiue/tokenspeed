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
M11 source commit: `c86d483db192b4bb72c01dbe0c305ff6c2d19d11`
(`feat(kda): materialize accepted replay endpoints`).
M12 source commit: `931eeaf21110265593e38e6563b0c1981003f361`
(`feat(kda): wire buffered decode and validate state commits`).
M13 source commit: `265ca5b97b8c9776e1c9a4989e84311a3c7c8df1`
(`feat(kda): materialize quiescent replay endpoints`).
M14 source commit: `fd31239aa518c75a3dc7dcd0172d49313d6bc03a`
(`feat(kda): compose mixed prefill and buffered decode`).
M15 source commit: `fea0e94fbd816116d9eebd0ac6b833e05bc8054b`
(`feat(kda): expose experimental buffered decode`).
M16 source commit: `38a396c8f5b0dc474ecda4ab624ec7789e0a3ad2`
(`fix(cache): retain exact decode checkpoint publication evidence`).
M17 source commit: `61e7508033cec8022b7f1052fdda27a3d0970dc4`
(`perf(kda): tune small-batch capacity-16 recurrence tiles`).
M19 source commit: `324c796ea8e10ae9bc7480b19a15043a99f97e47`
(`perf(kda): tune guarded long-history recurrence tiles`).
No default capacity or serving performance benefit has been established.
M15 registers the GPU recurrence and adds an explicit experimental capacity;
it is not a production-validated default. Eagle3 must run the new path without a performance regression
against the frozen baseline before the work meets its completion gate.

| Milestone | Status | Evidence / exit condition |
| --- | --- | --- |
| A. Recurrence, history representation, conv and acceptance reference | M1 CPU/GPU numerical gates passed | 19 CPU + 8 GPU cases; serving equivalence is a separate gate |
| B. LCM ownership, retention and lifecycle contract | M2–M5 foundations; M9 metadata owner; M15 real scheduler/GPU prefix resume | L2/PD and arbitrary live-endpoint handoff need end-to-end validation/integration |
| C. Unified GPU forward and commit | M12 decode, M14 mixed batches, M15 registration and explicit startup capacity | No default capacity or full-model acceptance yet |
| D. Graphs, overlap and lifecycle integration | M10–M14 graph/pipeline checks and M15 prefix/finish/cancel/slot tests pass | Full-model overlap and recovery/transfer validation remain |
| E. Real-model correctness and performance | Matched traces and M17 L16 AIME complete: baseline 26/30; L16 official 28/30, completed final answers 27/30; M16 generated-prefix reuse passes; M18/M19 L32/L64 measurements include baseline restart checks | All measured capacities still regress; M19 AIME, further independent repeats and broader workload validation remain open |

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

### M11: accepted-endpoint materialization

Source commit: `c86d483db192b4bb72c01dbe0c305ff6c2d19d11` (signed off),
based on `d8014a642ad829c4648d2df8550ab67e3bb85f0f` (M10 record).
No serving or real-model result yet. The exact `pre-commit run --all-files`
command passed after formatting and before the source commit.

The workspace now makes its endpoint decision after acceptance. It selects
an aligned accepted endpoint, or a caller-supplied GPU handoff mask, only when
the existing checkpoint still lags. Capacity flush and endpoint materialization
remain distinct: the former writes only old accepted history before this
round's candidates; the latter can include this round's accepted prefix.
One cross-layer kernel performs the selected endpoint writes. It reuses the
same FP32 tiled reconstruction as forward, extracted without changing its
arithmetic or tuning. Conv windows and endpoint state complete before the
group stamp commits record the resulting checkpoint. Invalid acceptance
suppresses all commit stores; a zero-acceptance handoff can still advance the
checkpoint over old committed history.

Stable descriptors name the pool's state/K/U/D fields and raw tables. Their
bytes and per-group endpoint flags are included in workspace budgeting. The
handoff mask is an explicit caller-owned input, like acceptance, not a hidden
per-request state map. Prefix alignment comes from the arena's identity grain;
the state group's span is used only for addressing. Published snapshots remain
read-only, and destinations must be request-writable.

Initial GPU validation passed **19 cache/runtime cases** (22 warnings, 39.92s)
on the same persistent GB300 allocation and software environment as M10, no
weights. The existing six pipeline combinations now cover aligned endpoint
writes, forced unaligned handoff, zero acceptance, mixed capacity-flush and
endpoint decisions, eager/graph, and exact conv/recurrent endpoint state against
the CPU reference. They keep copies of aligned snapshots and require later
decode not to modify them. Tolerances are unchanged. A final assertion also
checks that invalid acceptance cannot partially commit state or stamps.

This completes the experimental GPU operation, not its serving lifecycle.
Scheduler-triggered handoff, failure feedback, prefill seeding and external
completion/provenance still gate dispatch. Original Eagle3 no-regression,
real NVFP4 TP8, AIME 2026 and short NSYS requirements remain unchanged.

Four additional GPU cases cover width one/four and eager/graph with an implicit
zero initial state, state span 8 versus prefix identity 16, forced zero-count
handoff, completed capacity flush and padding. Rejected K/U/decay entries are
poisoned with NaNs before materialization. A direct CPU affine reconstruction
checks every state element, including untouched blocks. This catches using
state alignment as prefix identity or reading beyond the accepted endpoint.

The first endpoint launch grid scheduled every layer/head/value tile even on
empty rounds. It now uses a startup-fixed persistent grid, bounded to twice
the device's SM count, with a live-flag check before striding over tiles. This
keeps the eager/graph operation sequence unchanged and needs no host decision.
The edge-case fixture caps the grid at eight programs to exercise multiple
tiles per program. Only launch scheduling changed, not arithmetic or tolerances.

Final GPU checks passed **76 reference/kernel cases** (15 warnings, 130.91s)
and **511 runtime cases plus 317 subtests** (28 warnings, 27.99s). The latter
includes the 19 cache/pipeline cases; their separate persistent-grid rerun
passed in 21.65s. An earlier recurrence-suite run failed because its new flag
variable reused the CPU oracle's `materialized` name. Renaming the flag fixed
the fixture without changing the kernel or tolerance.

Same-GPU isolated commit medians, in microseconds across 69 layers, T=4/L=37,
18 old history entries and four accepted inputs:

| Commit operation | Batch 1 | Batch 8 |
| --- | ---: | ---: |
| M10 conv/stamp commit control | 7.15 | 16.95 |
| M11, no endpoint required | 11.68 | 22.16 |
| M11, every endpoint forced | 113.91 | 803.91 |

Before the persistent-grid change, empty-round M11 timings were 14.98/44.31 us.
The new protocol still adds about 4.5/5.2 us to the M10 commit control, and
full-batch materialization is expensive. These synthetic, fixed-address
measurements exclude forward, gates, table refresh and model work; M10 is a
prototype control, **not original Eagle3**. Raw samples and the earlier grid's
results are retained in the local runbook. The complete real-model path must
include this cost in its no-regression comparison; no serving speedup or
default capacity is established here.

The four-launch native recurrence timer also compares exact archived M10 code
against the shared-helper version: 48 matching cases have current/M10 ratios
of 0.9701–1.0449. The T=4/L=8 flush case is 11.197/11.208 us at batch one and
13.907/13.634 us at batch eight (M10/current). This includes position prepare,
backing validation, recurrence and stamp commit, not the new endpoint protocol.
The range does not justify a blanket no-regression claim.

All timings use five samples of 50 graph replays, 32 calls per graph after
warmup, without a profiler. At context 65,536, batch four and width four, M11
adds 3,324 bytes per TP8 rank to M10's tensor workspace: **11,211,752 bytes**
total, excluding LCM fields, the caller-owned handoff mask and CUDA stream/event
overhead. Exact source, environment, commands and hashes are retained with the
milestone artifacts.

### M12: backend dispatch and rank-agreed commit validity

Source: `931eeaf21110265593e38e6563b0c1981003f361`, based on `648c41bd`
(M11 record). The repository-wide pre-commit hooks passed before committing.
No new serving performance or real-model correctness result.

An explicitly planned buffered pool now binds `KDAReplayWorkspace` in the KDA
backend. Ordinary decode and verify share its refresh, forward and accepted
commit; neither uses a per-position state tape or eager recurrent replay.
The hybrid wrapper's existing commit hook supplies live accepted input counts
after execution, including after CUDA graph replay. Graph metadata exposes
the cached group views to the pointer guard. Pool replacement rebuilds the
workspace and descriptors; a changed replay layout is rejected before binding.
Refresh arms one commit; repeated commits fail before issuing stores. Validity
is unavailable until the commit is issued, so preparation alone cannot produce
a successful result. Rebind also resets the legacy replay capability latches.
The recipe accounts for the immutable no-handoff mask (one byte per runtime
request) alongside the workspace's existing tensors.

Metadata preparation reads only request-local, non-transferred stamps. Each
layer fences cache access before consuming the workspace's cached state views;
the batched commit follows every layer. The executor then snapshots live group
validity in its normal output D2H sequence. After the copy event completes,
`StateCommitValidator` agrees on failures over CPU TP and PP groups before
output post-processing or scheduler feedback. A rank with valid local data
must still reject if another rank failed. Missing/malformed decode flags also
fail through this agreement, so no rank exits the collective early. These
cache-invariant failures stop execution; they are not treated as recoverable
NaN output. Other cache contracts perform no extra collective.

The factory still enables no replay layout. Mixed buffered batches fail
explicitly until their handoff is integrated; they cannot read a lagging
checkpoint through the old prefill scan. Exact lifecycle handoff, public kernel
registration/configuration, real full-model NVFP4 TP8 Eagle3 no-regression,
AIME 2026 and NSYS remain required. In particular, this milestone's additional
CPU agreement and D2H must be included in the eventual end-to-end timing.

The first integration pass passed 102 cases plus 62 subtests and failed ten
legacy KDA fixtures that omitted the now-explicit refresh arguments. Fixing
those callers, including the production runner, passed **112 cases plus 62
subtests** (23 warnings, 43.28s). No kernel math or tolerance changed. A further
pass exercises the real hybrid forward/commit, pointer guard and pool rebind.
The suite includes a real two-process Gloo check: a failure present only on
rank one, missing flags and malformed flags are rejected by both ranks; a
stage without local buffered groups still participates successfully.

Environment: a fresh persistent four-GB300 allocation, driver 580.167.08,
Python 3.12.3, PyTorch 2.13.0+cu130 and tokenspeed-triton 3.8.10.post20260906.
Tests use TP8 per-layer dimensions and packed local-layer cache views, not an
eight-GPU model run. The cached image, serving venv and M4 scheduler extension
are unchanged; exact allocation, paths, commands and logs are retained in the
local milestone runbook. No weights or packages were downloaded.

The final integration suite passed **156 cases plus 62 subtests** (23
warnings, 36.70s), including the actual hybrid entry, duplicate/missing commit
guards, pointer guard, pool rebind, device/control-plane tests and NaN-guard
regressions. Three old cases were skipped: two require optional FLA and one
tests an AMD-only indexed-decode contract. Native CuteDSL prefill staging
checks did run. The final scheduler/cache/GDN suite passed **517 cases plus
317 subtests** (28 warnings, 34.13s). Counts overlap and must not be summed.
The expanded legacy suite initially exposed a mock pool missing its published
group specs and a capture call missing placeholder tables; both fixtures now
honor the real interface. No test was removed and no tolerance was widened.
The invalid-acceptance check refreshes the actual next endpoint and first
asserts valid backing, so an unrelated metadata failure cannot hide a missed
acceptance check. The static TP8 tensor budget is **11,211,756 bytes per rank**
at physical context extent 65,536, batch four and width four, four bytes above
M11. Serving adds speculative overshoot headroom to the logical context limit;
the real-model validation below records its actual allocation separately.

### M13: quiescent endpoint materialization

Source: `265ca5b97b8c9776e1c9a4989e84311a3c7c8df1`, based on `7f4ee451`
(M12 record).

`KDAReplayWorkspace.materialize_current` accepts fresh request tables and exact
accepted endpoints without running a candidate forward. It reuses the shared
position, backing-validation, endpoint and stamp operations with explicit
handoff semantics. Handoff has zero width and acceptance, validates only
committed history and the exact state destination, and never treats a planned
capacity flush as completed. It skips conv preparation and commit: the short
window already represents the accepted endpoint. All local payload fences
precede the batched endpoint writer; stamps follow all layer stores. The
operation returns borrowed live validity for the owner's completion check.
Already-exact endpoints perform no pool stores. Workspace tensor bytes are
unchanged; no persistent request map or independent storage pool was added.

The first focused GPU run passed **6 cases** (22 warnings, 28.36s): T1/L8,
T4/L8 and T4/L37 in eager and captured execution, using TP8 per-layer dimensions
across six local KDA layers and two cache groups. It checks missing candidate
capacity, a checkpoint that would trigger the next forward's capacity flush,
an implicit zero seed, reordered rows, exact recurrent output, unchanged conv,
duplicate handoff and invalid state backing. Subsequent tests add an exact
nonzero prefill seed and a missing middle history block. Final regression
results are recorded below once complete. Tolerances remain FP32 state
`atol=2e-5, rtol=2e-4`; conv and no-op pool comparisons are exact.

Environment is the same persistent four-GB300 allocation, cached image,
serving venv and scheduler extension as M12. This is not a real-model run.
Scheduler-triggered handoff, mixed-batch integration, registration/configuration
and the full-model NVFP4 TP8 Eagle3/AIME/NSYS gates remain pending. In particular,
this operation does not change retraction's stream-ordered writeback policy,
authorize page reuse or enable PD transfer of request-local history.

Final results: **76 kernel/reference cases passed** (15 warnings, 134.51s);
**162 integration cases plus 62 subtests passed** (23 warnings, 41.65s), with
the same three optional/vendor-specific skips as M12; **523 scheduler/cache/GDN
cases plus 317 subtests passed** (28 warnings, 36.74s). The final integration
also covers smaller and empty handoff batches. Counts overlap across suites.
All-files hooks formatted eight files; the final all-files run passed before
the signed-off source commit.

A same-GPU native-input microbenchmark compares the normal buffered decode
operations against frozen M12 `931eeaf2`, not the original Eagle3 implementation.
It covers 48 configurations: B1/B8, T1/T4, four capacities per width and
empty/no-flush/flush history. Two pairs run in before/after/after/before order;
each configuration uses five event samples, 32 calls per graph and 50 replays
per sample. New/old latency ratios range **0.998666–1.001094** across both pairs
(within ±0.14%). T4/L8/flush stays approximately **11.21 µs at B1** and
**14.26 µs at B8**. No distinguishable regression appears in this graph
microbenchmark. It excludes model execution, conv/gate producers, CPU feedback,
scheduler and quiescent handoff latency; it cannot satisfy the full-model
Eagle3 no-regression gate. Raw samples, source extraction, comparison script
and environment are retained in the local M13 artifacts.

### M14: mixed prefill and buffered decode

Source: `fd31239aa518c75a3dc7dcd0172d49313d6bc03a`, based on `58d42fa4`
(M13 record).

Mixed KDA metadata now describes prefill only for leading extend requests and
uses the ordinary buffered refresh for the decode suffix. Within each layer,
zero-copy producer slices feed the existing prefill scan and buffered decode;
the outputs rejoin in request/token order, with projection padding zeroed.
There is still one model forward. The common accepted-state hook delegates
through the hybrid wrapper; KDA commits only live decode counts. Legacy Mamba
retains its existing pure-decode commit behavior. Result validity likewise
covers only the live decode suffix, and the CPU check uses that row count.
No new persistent tensor workspace or request state is allocated.

This does not complete lifecycle handoff. Prefill still requires an exact
input snapshot from the owner; materializing after admission has recycled its
history would be too late. An eventual prefill handoff must also preserve its
borrowed failure flags before a later decode refresh overwrites metadata.
Neither shortcut is installed here. Scheduler-triggered handoff, factory and
kernel registration, real-model correctness and Eagle3 performance remain open.

Validation uses the same persistent four-GB300 environment as M13. Synthetic
TP8 per-layer dimensions span six local KDA layers and two cache groups. The
new parameterized test compares a real CuteDSL-prefill/buffered-decode mixed
batch against separate forwards on identical pools, then resumes all requests
through eager or captured decode. It covers fresh and cached prefill, an
internal aligned checkpoint, lagging decode state, partial/zero acceptance,
flush, projection padding and mode transitions. The comparison requires
bitwise-equal outputs and complete pool bytes; existing kernel tests retain
their independent CPU references. An initial fixture used the wrong backend
name and failed before kernels ran; it now selects `cutedsl_kda` explicitly.
The native kernel requires the real model's gate lower bound **-5.0**, so this
test uses that value rather than the prototype's synthetic -0.3. The first
actual composition run also caught the existing output-rank difference:
prefill returns `[tokens,H,D]`, while buffered decode returns `[1,tokens,H,D]`.
The join now adds a zero-copy leading dimension to prefill output. With that
fix, the focused suite passed **4 cases** (22 warnings, 18.48s), including
bitwise pool/output comparisons after mixed execution and graph decode.
The integration suite passed **169 cases plus 62 subtests** (23 warnings,
39.23s), with the same three optional/vendor-specific skips as M13. It also
checks invalid mixed acceptance before pool stores and successful/failed CPU
feedback for both pure decode and a mixed decode suffix. The scheduler/cache/GDN
suite passed **527 cases plus 317 subtests** (28 warnings, 36.46s). Counts overlap.
All-files hooks corrected imports and formatted two files; final validation and
source hashes are retained with the local milestone artifacts.

The real CuteDSL prefill kernel ran in the new test; only the older optional
FLA reference cases were skipped. No numerical tolerance was widened and no
kernel math changed. No new latency measurement is claimed for M14: M13's
recurrence-only timing is not evidence for this runtime change or the final
full-model no-regression gate. The plan's introduction now summarizes current
status rather than repeating superseded per-milestone statements; historical
details remain in this record and the full acceptance requirements are unchanged.

After formatting, the final integration run again passed **169 cases plus
62 subtests** (three skips, 23 warnings, 38.37s), and the shared regression run
passed **527 cases plus 317 subtests** (28 warnings, 37.68s). All applicable
all-files hooks passed before the signed-off source commit. Native prefill used
`tokenspeed-cutedsl-kda 0.1.0.post20260830`; binary hashes, exact commands,
environment, raw logs and the source patch are retained in the local M14 runbook.

### M15: experimental serving configuration and cache-owner resume

Source: `fea0e94fbd816116d9eebd0ac6b833e05bc8054b`, based on `f54c642d`
(M14 record).

`--ssm-replay-buffer-capacity` now reaches the Kimi-K3 recipe before memory
planning. A positive value selects buffered fields and the common backend path;
omitting it preserves the existing deployment. Zero is invalid, not an off
mode. Startup rejects other model families, non-FP32 recurrent state and PD.
The first registered recurrence covers Blackwell BF16, 128-dimensional heads,
width one/four and capacities through 64 with `L >= 2*T`. This is a bounded
experimental dispatch contract, not a recommended capacity. Selection happens
once at pool binding; the per-layer loop calls the resolved implementation
directly. Runtime imports now use the public kernel solution module. The
recurrence arithmetic, GPU launch sequence and workspace tensor budget are
unchanged. Startup logs identify capacity, width, kernel and workspace bytes.

The lifecycle audit resolves a previously ambiguous part of the plan. The
existing FSM has no live `Decoding -> Prefilling` transition: agentic turns
submit new requests and match immutable, materialized prefix checkpoints.
Retraction already exports reusable checkpoints and recomputes the suffix;
it does not transfer an arbitrary current endpoint. M15 preserves those
boundaries rather than adding a late materialization after sparse admission
has discarded history. A future direct live-endpoint consumer still requires
ordered materialization and checked completion before reuse. PD remains
explicitly rejected, and end-to-end L2 recovery/transfer validation is open.

On a fresh persistent eight-GB300 allocation (two nodes, same resource binding),
the startup/cache/backend suite passed **61 cases plus 15 subtests** (22 warnings,
102.61s), and the kernel/reference suite passed **77 cases** (15 warnings,
141.16s). Software and cached dependencies match M14; no weights or image were
downloaded. These tests use one GPU and are not TP8 model execution.

Five new real-scheduler/GPU lifecycle cases passed (22 warnings, 13.46s).
They use six local KDA layers, actual allocator-produced block tables, page
zeroing, native prefill, registered buffered decode and accepted feedback.
Coverage includes an aligned accepted endpoint, a skipped/unwritten boundary,
lagging live state behind an existing prefill checkpoint, capacity flush,
finish/cancel and reuse of the request slot. Resumed prefill reads the exact
saved checkpoint; its first decode starts empty, and published snapshots stay
bitwise unchanged. The initial fixture incorrectly expected an empty operation
list instead of an explicit idle batch. Its next run exposed a legitimate
prefix-promotion chunk when no state checkpoint matched; the fixture now sends
empty feedback for that intermediate chunk and executes its remaining tail.
Neither correction changes scheduling, arithmetic, tolerance or the expected
cache-hit boundary.

The model recipe now documents the experimental option and removes an obsolete
claim that K3 prefix granularity depends on the memory budget. Full-model
EAGLE3 performance, capacity choice, AIME and NSYS remain pending; this milestone
does not establish an end-to-end speedup or accuracy score.

After formatting, the final integration suite passed **200 cases plus
77 subtests** (three optional/vendor-specific skips, 23 warnings, 50.07s).
The shared runtime suite passed **527 cases plus 317 subtests** (31 warnings,
48.86s). These counts overlap with the focused suites above. All applicable
all-files hooks passed before the signed-off source commit. The local M15
runbook retains the environment, exact commands, source patch and raw logs.

### Real-model validation: first baseline run

The frozen original `2e4b540743878959eb51793ce01d74b5898b0e38` completed a
real-weight agentic smoke and its first unprofiled timing run. The buffered
comparison uses the independent M15 source archive; results below describe
the baseline only, not an optimization benefit or a no-regression pass.

Environment: eight GB300 GPUs across two nodes in one healthy, full-bandwidth
NVLink fabric, under a fresh persistent allocation. Driver 580.167.08,
Python 3.12.3, Torch 2.13.0+cu130, FlashInfer 0.6.18 and
tokenspeed-triton 3.8.10.post20260906. Both archives use the same cached
dependencies/native kernel objects and their own matching scheduler binaries.
The full 93-layer NVFP4 model uses TP8 attention/MoE, BF16 activations, FP8 KV,
MLA attention, CuteDSL KDA prefill and FlashInfer TRT-LLM MoE. EAGLE3 uses the
MLA draft, three draft steps, four target tokens and top-k one. CUDA graphs
and overlap remain enabled, with decode capture sizes 1/2/4 and prefill graphs.
Context is 65,536, prefix granularity 128, maximum live requests four,
prefill chunk/budget 8,192 and memory utilization 0.80. Autotuning is disabled
equally in both cases; no communication override is applied.

The smoke reuses a frozen, rendered real-content agentic conversation: 51,936
input tokens, 51,328 cached tokens and 608 new tokens, followed by 16 outputs.
All 16 output IDs match the prior frozen-baseline smoke. This is execution
evidence, not an AIME or agent task-resolution score.

Unprofiled timing uses that same continuation with 256 outputs, greedy sampling
and seed one. Each batch flushes the test cache, primes the identical frozen
parent, then submits either one or four simultaneous identical continuations.
There are three rounds per concurrency, each with one warmup and five measured
batches: 15 C1 requests and 60 C4 requests in total. All samples completed with
the expected cache/token counts and no preemption. C1 client latency has a
median of **1,070.642 ms**, with median decode throughput **287.9 tokens/s**;
C4 per-request client latency has a median of **1,631.700 ms**, with median
decode throughput **218.7 tokens/s**. C4 admission spans multiple prefill
batches, so those per-request rates are not aggregate batch throughput.
C1 reproduces one exact output sequence across all samples; C4 produces two.
All sequences, engine statistics, raw timings and warmups are retained.

Host-side Slurm/process audits confirm all eight workers belong to the test
and no Nsight injection is present. PyTorch's ordinary CUDA13 CUPTI library
is mapped even without profiling; its presence alone is not a trace session.
The local runbook records the corrected audit assumptions and initial failed
audit attempts. No failed model request was replaced or filtered. Buffered
timings, independent restart repeats, the capacity sweep, AIME and fresh NSYS
remain pending.

### First buffered model measurements and publication follow-up

The M15 L8 server completed its real-weight smoke, graph capture and all
75 measured requests with the same eight-GPU setup and protocol as the baseline
above. Each rank selected `triton_kda_buffered_recurrent`, width four, capacity
eight. Actual tensor workspace was **11,211,852 bytes per rank**: the 65,536
logical context includes 12 speculative overshoot tokens in physical metadata,
adding one raw-table column (96 bytes) to the static M12 example.

The 16-token smoke matches baseline through the first 11 tokens, then diverges.
Exact token parity therefore fails; successful execution and passing kernel
references do not by themselves establish full-model accuracy. AIME is pending.

The first L8 timing run has median client latency **1,168.941 ms at C1**,
**9.18% slower** than the first baseline run, and **1,623.153 ms at C4**, about
**0.52% faster**. Median acceptance length changes from 3.59 to 3.27 at C1;
it is 3.67 in both C4 runs. C1 median decode throughput is 259.1 tokens/s,
C4 221.25 tokens/s. All samples have the expected token/cache counts and no
preemption; C1 has one exact output sequence and C4 two. These are first-run
observations, not a restart-controlled performance pass. The completion gate
has not passed, and no default capacity is recommended.

The frozen M15 L16 server also completed all 75 measured requests. Median
client latency was **1,082.545 ms at C1** and **1,830.034 ms at C4**, respectively
**1.11%** and **12.16% slower** than baseline. Median decode throughput was
283.5 and 185.0 tokens/s; acceptance length was 3.64 and 3.10. Its 16-token
smoke matched L8, including the difference from baseline. No request failed or
was preempted. Neither tested capacity passes the end-to-end no-regression
gate. The remaining capacity sweep, independent restart repeats, AIME and
fresh NSYS are still open.

A separate raw-token lifecycle diagnostic generated 1,024 tokens, then reused
that generated prefix. It hit only the original prefill snapshot at 51,840,
so its assertion requiring a decode-produced checkpoint failed. The original
responses and failure are retained; the test was not weakened or retried.
This is not an AIME or rendered agent task-resolution test.

The follow-up scheduler test deterministically reproduces a missing publication
watermark: accepted progress reaches 128, but four-token verify's conservative
frontier is 125. After accepted progress moves to 129/133, the old scheduler
forgets that state 128 was materialized before the hash frontier can publish it.
The test fails for exact-boundary cases with and without replay history, at
overlap depths zero/one; skipped-boundary cases remain correctly unhittable.
The fix preserves only known exact materialization evidence across admission
rounds. This fix is being validated on top of `de69d598`; both capacity
experiments above used frozen M15, not the fix.

### M16: preserve exact decode checkpoints until publication

Source: `38a396c8f5b0dc474ecda4ab624ec7789e0a3ad2`, based on `de69d598`.

The scheduler now retains the last known exact materialization boundary in
`CacheProgress` when scheduling decode. It still rejects a boundary merely
crossed by a verify window. There is no new state allocation, GPU work or
kernel arithmetic change.

The new C++ scenario fails all four exact-boundary combinations with the old
implementation. With the fix, the full scheduler suite passes **488 tests in
134 suites**, including after all-files formatting and rebuilding. A matching
real-scheduler/GPU lifecycle case also fails against the preserved old binary:
**one failed, five passed**. It continues decoding beyond an aligned endpoint
before finish and prefix reuse, covering the gap in the earlier lifecycle
tests. With the fixed binary, GPU integration passed **201 tests plus
77 subtests** (three optional/vendor-specific skips, 23 warnings, 47.48s).
The shared scheduler/cache/runtime suite passed **527 tests plus 317 subtests**
(28 warnings, 38.44s), including GDN paging coverage. The test launch initially
waited for Slurm step creation after communication timeouts, then proceeded
without resubmission. No numerical tolerance was changed. No fixed-source
full-model result or performance improvement is claimed yet.

All applicable all-files hooks passed before the signed-off source commit.
The matching scheduler was rebuilt and staged separately; the source patch,
binary hashes, environment, exact commands and raw results are retained with
the local M16 artifacts. Full-model reruns use a separate source archive so
the earlier M15 measurements keep their original provenance.

### M16 real-model follow-up

The separate `38a396c8` archive completed the real-weight TP8 L8 smoke,
generated-prefix diagnostic and short NSYS capture with the same full model,
EAGLE3, graph/overlap and dependency settings recorded above. The scheduler
fix restores reuse of a decode-produced checkpoint: the resumed request hits
**52,864 tokens**, versus the old buffered source's prefill-only **51,840**.
All **1,024 parent output IDs** match the old buffered run. The smoke also
matches M15, including its difference from the original baseline.

After flushing, the cold replay of that same resumed prompt completes with
64 outputs. Warm and cold outputs first differ at zero-based index **20**.
Checkpoint reuse is therefore established, but exact warm/cold parity is not.
This raw-token lifecycle diagnostic is not an AIME or agent task-resolution
score; full-model accuracy remains open.

The short trace captures one frozen continuation with **608 new prefill tokens**
and **96 generated tokens**, after priming and warmup. Each of the eight GPUs
contains **one prefill forward and 39 decode graph replays**, with 2,598 kernel
nodes per decode graph. The profiled output matches both its 96-token warmup
and the first 96 tokens of the earlier unprofiled L8 timing sample. The two
node reports, source/binary provenance, complete graph samples and analysis
scripts are retained locally and packaged with clear filenames. Captured
durations are not unprofiled performance measurements. The matching new
baseline trace is reported below.

### M17: matched traces and bounded launch tuning

Source: `61e7508033cec8022b7f1052fdda27a3d0970dc4`, based on `a6d3e3fa`.
Kernel/runtime validation passed; the full-model results below do not pass the performance gate.
Only native four-token, capacity-16 recurrence with
12 local heads and batch sizes one through four changes its value tile from
32 to 16. Other shapes retain their previous configuration. Four warps, the
history tile, arithmetic, flush policy and L8 behavior stay unchanged. The existing parameterized recurrence
test adds capacity 16 alongside minimum, odd and long capacities; its width-one
cases also cover the corresponding intermediate capacity.

The matched original `2e4b5407` trace now covers all eight GPUs, using the same
environment and frozen request as M16. Each GPU has one prefill and **31 decode
graph replays**, compared with buffered L8's **39**, for the same 96-token output
budget. Per-GPU graph medians range from **12.141–12.152 ms** originally and
**12.197–12.210 ms** buffered. Generated tokens and acceptance differ, so the
additional rounds are not a kernel-only performance comparison. Rank 0's
recurrent/verify kernel medians are 5.376 us and 8.480 us. Original accepted
replay is already one batched recurrent call per round across KDA layers,
outside the graph (31 calls, median 42.368 us), not one replay launch per layer.
Both two-node captures and complete analysis are packaged together locally.

On an idle GPU in the same allocation, **144 isolated launch measurements**
compared dynamic versus unrolled loops, value tiles 4/8/16/32, one/two/four/eight
warps and selected pipeline depths at B1/B4, L8/L16, empty and flush histories.
All configurations passed the existing numerical tolerance; 88 were not
bitwise identical. No tolerance changed. Unrolling did not consistently help
L8 flush, so it was not adopted. The selected dynamic L16 tile-16 candidate
was bitwise identical to the current output, state and history in all four
measured cases. Its median kernel times changed as follows:

| L16 case | Original tile 32, bracketed medians | Candidate tile 16 |
| --- | ---: | ---: |
| B1, empty | 6.404–6.412 us | 5.638 us |
| B1, flush | 9.222–9.227 us | 8.202 us |
| B4, empty | 7.423–7.428 us | 6.656 us |
| B4, flush | 11.139–11.144 us | 10.331 us |

This is a kernel-only candidate, not an EAGLE3 no-regression pass. Each sample
replays a 16-call CUDA graph 25 times, with five event-timed repetitions;
configurations are shuffled and bracketed by the current configuration.
The source variants, all timings, numerical differences and script are retained.
The first expanded kernel suite passed **89 tests** (15 warnings, 132.85s),
integration passed **201 tests plus 77 subtests** (three optional/vendor skips,
23 warnings, 49.48s), and shared runtime passed **527 tests plus 317 subtests**
(28 warnings, 37.51s). The CPU reference passed **19 tests** in 7.73s. These
runs preceded the final small-batch guard. On the final guarded source, the
same suites passed **89 kernel/reference tests** (134.20s), **201 integration
tests plus 77 subtests** (three skips, 48.11s), and **527 shared runtime tests
plus 317 subtests** (36.61s). Warning counts were unchanged. A separate actual
dispatch check passed **14 cases**, confirming the selected tile and bitwise
results at B1/2/3/4/8/16/32 with six or twelve heads. Large batches and other
head counts keep tile 32. All applicable all-files hooks passed before the
signed-off source commit. Serving uses a separate archive of that commit and
the unchanged matching M16 scheduler binary. Full-model results follow below.
The earlier isolated scaling/FMA experiments did not establish exact
agreement with the original native verify kernel; no such math change was
adopted.

The first baseline AIME attempt was stopped at **27/30**, with every completed
answer retained. Its short-profile graph ladder captured 1/2/4 with padding
disabled, leaving the final three requests on slow eager execution. This is
an incomplete run, not a reported score. Fresh complete evaluations use graph
sizes **1/2/3/4 for both sources**, with all dataset, sampling and output-budget
settings unchanged. Short timing and trace protocols retain their original
ladder; timing on the new AIME servers is a separately matched comparison.

A second isolated sweep checked **56 cases**: B1/2/3/4/8/16/32, history lengths
0/3/8/12 and two seeds. All outputs, state and history were bitwise identical
between value tiles 16 and 32. Paired 32/16/16/32 timings show improvements for
B1/B4 and most B2/B3 cases; B2/B3 long-history flush is effectively tied.
Larger batches regress: roughly 31–39% at B8, 13–18% at B16 and 8–13% at B32.
That evidence narrows the production change to the measured small-batch,
12-head geometry. It is not a universal L16 configuration change.

The corrected frozen-baseline AIME run completed all **30 questions** with
**26 correct (86.67%)**, zero request errors and one 63,488-token budget stop.
All prompts were cold, with 203,117 output tokens in total. Official scoring
and independent full-text grading agree on every question. A separate audit
grades the 29 explicit response channels and treats the budget-truncated text
without a response channel as incomplete; it also yields 26/30. Both audit
versions and every prediction remain in the artifacts. This is the original
source's result, not accuracy evidence for the buffered candidate.

The same unprofiled baseline server completed the 75-request timing protocol
with the new 1/2/3/4 graph ladder. Median client latency is **1,058.916 ms at C1**
and **1,597.885 ms at C4**; median decode throughput is 288.4 and 214.5 tokens/s,
and acceptance length is 3.59 and 3.67. No request failed or was preempted.
The candidate must use this same ladder for its paired timing; these results
are not mixed into the earlier 1/2/4 comparison.

### M17 full-model AIME and matched performance

Both complete evaluations use the frozen original `2e4b5407` and M17
`61e75080`, the same eight GB300 GPUs, full 93-layer real NVFP4 weights, TP8,
EAGLE3 and CUDA graphs/overlap. L16 is explicitly enabled only for M17.
The software, target and draft revisions are unchanged from the environment
record above. This pair captures graph sizes **1/2/3/4**, with padding disabled.
It is separate from the earlier 1/2/4 profile and timing pair.

AIME uses EvalScope 1.11.1 and all 30 frozen `math-ai/aime26` questions,
revision `79037aebdb6580008fb960d17cb21fd3099083e3`, with temperature 1,
seed 42, a 63,488-token output budget, concurrency four and one attempt per
question. There are no prompt cache hits, retries or replacement samples.
All 30 actual input messages, dataset hashes, normalized evaluation commands,
GPU identities, dependencies and native objects match between sources.

| AIME result | Original | M17 L16 |
| --- | ---: | ---: |
| Official scorer | 26/30 (86.67%) | 28/30 (93.33%) |
| Completed final-response audit | 26/30 (86.67%) | 27/30 (90.00%) |
| Request errors | 0 | 0 |
| Output-budget stops | 1 | 1 |
| Total output tokens | 203,117 | 263,275 |

The candidate's budget-truncated question at zero-based index 14 has no final
response channel. The official scorer extracts the correct number, 83, from
its unfinished text; the final-response audit excludes that point. The
baseline also truncates that question but receives no point. M17 corrects
indices 21 and 29, misses index 23 that baseline answered correctly, and both
miss index 6. Independent full-text regrading agrees with the official scorer
on every question. This is one matched dataset run, not evidence of a general
accuracy improvement or bitwise equivalence. Different generation lengths
make AIME wall times unsuitable as a controlled speed comparison.

The unprofiled timing protocol is unchanged: the identical frozen continuation,
51,936 input tokens, 51,328 cached tokens, 608 new prefill tokens and 256
generated tokens per request. Each source completes three rounds at C1/C4,
with one warmup and five measured batches per round: **75 measured requests**
and six warmup batches. Host audits exclude active profiler injection.

| Metric | Original | M17 L16 | Change |
| --- | ---: | ---: | ---: |
| C1 median client latency | 1,058.916 ms | 1,074.716 ms | +1.49% |
| C4 median client latency | 1,597.885 ms | 1,816.588 ms | +13.69% |
| C1 median decode throughput | 288.4 tokens/s | 285.7 tokens/s | |
| C4 median decode throughput | 214.5 tokens/s | 187.7 tokens/s | |
| C1 median acceptance length | 3.59 | 3.64 | |
| C4 median acceptance length | 3.67 | 3.10 | |

No measured request fails or is preempted. Each source has one exact output
sequence at C1 and two at C4. C4 is 15 concurrent batches, not 60 independent
timing observations; median whole-batch latency is 1,609.466 ms originally
and 1,883.348 ms for M17. Acceptance differs, so these are end-to-end results,
not an isolated measure of recurrence kernel cost. They also do not isolate
the tile change from M16's scheduler fix or the older graph configuration.

The comparison harness initially hashed completed conversations, including
assistant replies and generated message IDs, when checking prompt identity.
The corrected check compares every input-message field except the generated
ID and records input-only hashes. All 30 comparisons pass; no model input,
raw prediction, score or timing sample was changed to resolve the check.

Raw predictions, both scoring audits, per-question inputs/results, manifests,
all request and batch timings, comparison scripts and the matched provenance
are retained in local artifacts. This is one server run per source; independent
restart repeats, the remaining capacity sweep and broader workload coverage
remain open. **The EAGLE3 no-regression gate still fails. Buffered replay stays
opt-in, with no recommended default capacity.**

### M18: new GPU cohort and remaining capacity checks

Sources remain frozen at original `2e4b5407` and candidate `61e75080`; no
production code changes accompany this validation. A new persistent allocation
provides eight GB300 GPUs under the same resource binding. Both nodes pass
the idle, common healthy/full NVLink fabric, source and dependency checks.
This is a separate timing cohort, with graph sizes 1/2/3/4 throughout.

Its first baseline completes the smoke and all **75 measured requests** plus
six warmup batches, with no errors or preemption. C1 median client latency is
**1,065.635 ms**, C4 **1,619.871 ms**; decode throughput is 287.5 and
216.25 tokens/s, and acceptance is 3.59 and 3.67. C1 has one exact output
sequence and C4 two; the smoke matches the earlier baseline cohort. These
measurements are retained separately, not substituted into the M17 comparison.

Before loading L32, the frozen recurrence test is invoked at that exact
intermediate capacity: **12 combinations pass in 20.57s**, covering T1/T4,
eager/graph and prepared/softplus/bounded inputs. The existing sequential
reference, multi-round rejection/flush/padding/page-reuse checks and numerical
tolerances are unchanged. This reuses the parameterized test function; it is
not 12 new tracked test functions or a full-model accuracy result.

The bounded follow-up sequence runs L32, L64 and a second independent baseline.
Each case must complete smoke, source/worker ownership checks and all timing
samples before the controller stops its exact owned server step. Both GPU
nodes must be idle before the next model starts. Failures stop the sequence
and retain partial artifacts; no retry or sample replacement is automatic.
L32 and L64 have now completed on this cohort. Each finishes the same 75
measured requests and six warmup batches, with no request errors or preemption.
All ranks select the buffered recurrence at the requested capacity. Each run
has one exact output sequence at C1 and two at C4; neither capacity has a
full-model AIME result. The L16 score above must not be attributed to them.

| Metric | First baseline | L32 | L64 |
| --- | ---: | ---: | ---: |
| C1 median client latency | 1,065.635 ms | 1,090.406 ms (+2.32%) | 1,074.513 ms (+0.83%) |
| C4 median client latency | 1,619.871 ms | 1,799.633 ms (+11.10%) | 1,751.396 ms (+8.12%) |
| C4 median whole-batch latency | 1,628.462 ms | 1,887.201 ms (+15.89%) | 1,848.089 ms (+13.49%) |
| C1 median acceptance length | 3.59 | 3.64 | 3.75 |
| C4 median acceptance length | 3.67 | 3.28 | 3.405 |

The whole-batch figures retain the 15 concurrent batches as the timing units.
From the source's rounded acceptance statistic, the corresponding integer
request-accounted verify counts are uniquely determined here: C1 is 71 rounds
for baseline, 70 for L32 and 68 for L64. C4 is split evenly between 69/70,
71/86 and 69/82 rounds, respectively. These are inferred request counts, not
direct observations of GPU graph replays; the helper retains ambiguity for
rounded values with more than one integer solution.

Fewer rounds do not by themselves make C1 faster. Its median request decode
window is 887.020 ms originally, 908.710 ms at L32 and 896.140 ms at L64.
Dividing each request's window by its accounted rounds gives medians of
12.493, 12.982 and 13.179 ms, respectively. These include host scheduling and
overlap, not just recurrence kernels. At C4, changed outputs and concurrent
request interactions prevent interpreting such averages as kernel timings.

The second baseline restart also completes all 75 measured requests and six
warmup batches without errors or preemption. Its C1/C4 median client latencies
are **1,061.186 / 1,604.796 ms**, and median C4 whole-batch latency is
1,615.668 ms. The 30 measured batches have exactly the same output multisets
as the first baseline, including multiplicities but not request arrival order;
acceptance and inferred round counts also agree. Both restarts use the same
eight GPU UUIDs, source, native objects, model, graph ladder and input protocol.

Against this second baseline, L32 client latency regresses **2.75% / 12.14%**
at C1/C4, and L64 **1.26% / 9.14%**. C4 whole-batch latency increases 16.81%
and 14.39%, respectively. Comparisons against both baseline restarts are kept
separately; there is no pooled or selectively chosen baseline. Each candidate
still has only one restart in this cohort. **Neither capacity passes the
no-regression gate.**

After validating and stopping the last owned server, both model nodes pass
another idle check. The isolated experiment compares value tiles 32/16 at
L16/L32/L64, requiring bitwise output, state and history equality before timing.
It also records compiler register, spill and shared-memory metadata. No
production code or default changes have been made for that experiment.

### M19: bounded launch tuning for longer histories

Source: `324c796e`, based on `4f477bee`; the tests below ran before the signed-off
source commit, with unchanged kernel and test hashes. The required
`pre-commit run --all-files` completed successfully. The same GB300
allocation and software environment as M18 are used; the kernel experiment
runs alone after both serving nodes pass an idle check. The frozen M17 source
is the numerical and timing reference. No model weight, sampling parameter,
state representation, flush policy or arithmetic operation changes.

An initial sweep compares value tiles 32/16 with four warps and one stage at
L16/L32/L64, B1/2/3/4/8/16/32, four history lengths and two seeds. **All 168
cases are bitwise equal** for output, materialized state and all history
fields. Paired CUDA-graph timings use 32/16/16/32 order, with 16 calls per graph,
25 replays per sample and five event samples. All cases, including regressions
and baseline-anchor drift, are retained.

Tile 16 reduces compiled registers per thread from 130 to 96 in these shapes;
both variants report zero spills and 8,192 bytes of shared memory per program.
It doubles the number of value-tile programs. This is compiler evidence, not
a measurement of achieved occupancy or proof of a single bottleneck.

The promising shapes are checked again with two new seeds at **every reachable
history length**: 0–28 at L32/B1 and L32/B4, and 0–60 at L64/B1. **All 238
additional cases are bitwise equal.** The paired isolated kernel latency
changes across those lengths are:

| Shape | Tile-16 latency change versus tile 32 |
| --- | ---: |
| L32, B1 | −12.99% to −8.34% |
| L32, B4 | −9.62% to −2.62% |
| L64, B1 | −13.11% to −6.59% |

The candidate extends the existing static tile choice only to those three
shapes, with native T4, 12 local heads and 128-dimensional keys/values. L16
keeps its prior B1–4 tuning. B2/B3 at longer capacities, L64/B4, larger batches,
other capacities/head counts, T1 and prepared-input execution are unchanged.
The existing parameterized recurrence test adds exact L32/T4 coverage; its
assertions and tolerances are unchanged. Validation of the actual candidate
passes **112 strict dispatch/numerical cases** across L16/L17/L32/L64,
B1/2/3/4/8/16/32, H6/H12 and two seeds. This includes shapes deliberately left
on tile 32, not just the new tile-16 cases.

The full suites then pass: **101 kernel/reference tests** in 171.68s;
**201 integration tests plus 77 subtests** in 92.05s, with the same three
optional/backend-specific skips; and **527 shared runtime tests plus 317
subtests** in 48.82s. The GPU dispatch artifact's kernel hash matches the
workspace source held unchanged during these tests. No tolerance was relaxed.
These microbenchmarks alone do not establish an end-to-end speedup or a
default capacity; the separate real-model results follow below.

The new source is frozen separately from M17 before model startup. Serving
uses the same eight GB300 GPUs, 93-layer real NVFP4 Kimi-K3, TP8, BF16
activations, FP8 KV and four-token EAGLE3 verify as M18. Python 3.12.3,
PyTorch 2.13.0/CUDA 13, driver 580.167.08, tokenspeed-triton 3.8.10,
FlashInfer 0.6.18 and the cached model/draft revisions are unchanged. Decode
graphs capture batch sizes 1/2/3/4; segmented prefill graphs and overlap are
enabled, while graph padding is disabled. Attention breaks in the prefill
graph still execute eagerly.

The bounded model sequence is new L32, new L64, then another independent
original-baseline restart. It uses the unchanged 75-request timing protocol
above. Both candidate runs and the final baseline complete without request
errors or preemption. Each candidate has one exact
output sequence at C1 and two at C4. All 30 measured batch output multisets
match the corresponding M17 capacity run, including multiplicities, and
acceptance is unchanged. This checks the tile change on this workload; it is
not token equivalence to the original implementation or a new AIME result.

| Metric | Preceding original baseline | M19 L32 | M19 L64 |
| --- | ---: | ---: | ---: |
| C1 median client latency | 1,061.186 ms | 1,080.693 ms (+1.84%) | 1,068.392 ms (+0.68%) |
| C4 median client latency | 1,604.796 ms | 1,765.626 ms (+10.02%) | 1,750.318 ms (+9.07%) |
| C4 median whole-batch latency | 1,615.668 ms | 1,869.533 ms (+15.71%) | 1,842.901 ms (+14.06%) |

Against the earlier M17 run at the same capacity, M19 L32 client medians
decrease 0.89% at C1 and 1.89% at C4. L64 decreases 0.57% at C1 and 0.06%
at C4; the latter is effectively unchanged, not evidence of a meaningful
speedup. These are observed single-restart comparisons, with three timing
rounds per restart, not confidence bounds for a general workload improvement.
The following baseline provides a separate drift check: C1 median client
latency is **1,060.345 ms**, C4 **1,599.908 ms**, and C4 whole-batch latency
**1,611.606 ms**. Its 30 measured batch output multisets and acceptance match
the preceding baseline. All seven completed cases in this GPU cohort pass
the shared input/protocol, source/native, eight-GPU identity and startup-flag
checks. All eight ranks select buffered recurrence in each candidate run.

Against the following baseline, M19 L32 client latency increases
**1.92% / 10.36%** at C1/C4 and L64 **0.76% / 9.40%**. C4 whole-batch latency
increases **16.00% / 14.35%**, respectively. The comparisons against the
preceding baseline remain in the table; neither baseline is pooled or replaced.
**Neither candidate passes the overall EAGLE3 no-regression gate.** The tile
change preserves this workload's earlier buffered outputs and acceptance,
so it also leaves their difference from the original implementation intact.

The final controller's post-stop idle probe times out after all measurements
and result validation have finished. A separate read-only check confirms both
GPU nodes are idle. The controller failure and successful cleanup recheck are
both retained; no model, timing sample or output is rerun to resolve cleanup.

The local `m19/phase-comparison.json` report keeps every baseline/candidate
comparison, all output checks, tested source hashes and cleanup evidence.
The adjacent runbook records exact environment, commands and raw artifact
locations. This phase adds no AIME score or Nsight capture. The prepared
cross-layer capacity-flush and broader-corpus experiments remain unexecuted;
work is paused after this phase. Buffered replay remains opt-in, with no
recommended default capacity.
