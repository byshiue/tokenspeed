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
No default capacity or performance benefit has been established. M1 adds
an unregistered prototype, not the complete serving feature.

| Milestone | Status | Evidence / exit condition |
| --- | --- | --- |
| A. Recurrence, history representation, conv and acceptance reference | M1 CPU/GPU numerical gates passed | 19 CPU + 8 GPU cases; serving equivalence is a separate gate |
| B. LCM ownership, retention and lifecycle contract | M2 retention/capacity, M3 request-local ownership and M4 empty-prefill allocation verified | GPU field/metadata binding, endpoint materialization and handoff still pending |
| C. Unified GPU forward and commit | Isolated recurrence prototype; not registered or integrated | LCM, conv/gate integration and serving dispatch remain pending |
| D. Graphs, overlap and lifecycle integration | Not started | Fixed addresses, padding, request reuse, prefill transitions, prefix reuse and recovery |
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
