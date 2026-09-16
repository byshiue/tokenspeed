# KDA buffered replay: implementation record

This record tracks implementation of [the buffered replay plan](kda-buffered-replay-plan.md).
It separates reference results from kernel tests and real-model measurements.
Passing a reference test does not mean buffered replay is available in serving.

## Current validation status

M70 optimizes the committed shared-arithmetic implementation (`5fc562dd`)
without changing its numerical contract. The working patch uses warp-local
verify reductions, token-interleaved recurrence and fused conv/gate producers.
Final-source validation passes 197 kernel tests, 240 runtime tests plus 117
subtests (three existing skips), and all 1656 real-input layer/cases. The CUDA-
graph KDA-only comparison reaches frozen-original latency at L8/B4/T4; see M70
below for repeated measurements and scope. This is not a full-model performance,
AR or AIME pass. Those gates remain open.

M69 integrates replay-order producers and ordered reconstruction with a
user-approved shared verify arithmetic contract into the working tree. Local
validation passes: 187 kernel tests, 240 runtime tests plus 117 subtests (three
existing skips), and all 1656 saved real-input layer/cases across eight ranks.
The new unbuffered/buffered pair matches bitwise; the frozen original remains a
separate reference and has small output differences. The compiler-specific M59
diagnostic is not adopted. This change includes the M69 implementation and
validation record on top of `2829f469`; current-version AR, AIME and the Eagle3
no-regression gate remain open. Earlier
results below apply only to their recorded source, not this new arithmetic.

M47R3 restores full-model L64/C1 tokens and acceptance in M50: both paths have
AR 0.8638 and acceptance length 3.59. Latency still increases 9.01%, so that
result is not a performance pass. M51 traces the remaining same-trajectory
cost to history reconstruction and history-gate work, with a separate late-
rank outlier retained in the profile. C1 recovery does not establish C4.

The preceding AR investigation was fixed at L8/C4. M53's first L8 run retains
C1 outputs but changes every measured C4 batch. Its later L16 run stops on a
cache-hit/workload mismatch; the six-startup comparison is incomplete and
the partial samples are not replaced. M55 isolates B4 verify differences
despite exact producers and accepted/flush state. M58/M59 attribute those
same-input differences to normalization ownership and projection/update/dot
contraction. An explicit diagnostic reproduces the original compiler's
rounding in all 1656 B4 cases; C1 carried-state, 163 kernel tests and 201
runtime tests plus 77 subtests pass (three existing skips).

M61 revision 2 completes the real NVFP4 TP8, CUDA-graph-enabled L8/C4
comparison on the same base source. Unchanged buffering reproduces the AR
gap; aligning only verify arithmetic restores all output-token and acceptance
multisets to the unbuffered control. All twelve batches have the same 1+1+2
prefill grouping and cache workload. This supports verify arithmetic as the
cause in this fixed case, following the same-input state/verify checks above.
The diagnostic's compiler-specific value-row masks are not a portable
production fix. No performance, current AIME or broader-capacity pass follows.
The first attempt produces no C4 sample because the gateway rejects batched
input IDs; the recorded recovery uses four concurrent scalar-input requests.
Details, retained failures and remaining implementation work follow below.

M62–M64 test three alternatives to the compiler-specific diagnostic. Inferred
Triton layouts worsen verify and accepted-state equality; a static verify
window and BF16 verification input storage leave all M57 per-layer metrics
unchanged. None is adopted. M69 implements a shared arithmetic contract instead;
it still requires new full-model AR and performance checks, not a relaxed gate.

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

### M20: cross-layer capacity-flush feasibility

Work resumes at `3ee2666f`, whose production kernels are unchanged from the
frozen M19 source `324c796e`. This first experiment changes no production code,
flush policy, state ownership or runtime placement. It runs on an idle GB300
from the M19 allocation with the same cached software, not a model server.

The comparison is 69 per-layer recurrences versus one existing cross-layer
endpoint materialization followed by 69 recurrences with empty history only
on the flushed rows. Both use the same pre-candidate endpoint and the existing
`h + 2*T > L` rule. The fixture has three cache groups, 12 local heads,
128-dimensional keys/values and T4; it tests L8/16/32/64, B1/B4, empty and
flush rounds, plus mixed B4 rounds with padding. Two seeds give **40 numerical
cases: all pass the existing tolerances, and 34 are bitwise equal** for output,
full state and K/U/decay history. The six non-bitwise cases are retained; no
tolerance changes or full-model accuracy claims accompany these results.

For one seed, each of the 20 shapes uses CUDA-graph timings in
per-layer/batched/batched/per-layer order: 16 calls per graph, 25 replays per
event sample and five samples per variant. The measured total includes the
materializer even in empty rounds. Relative to the bracketed per-layer timing:

| History capacity | B1 flush | B4 all flush | B4 mixed |
| --- | ---: | ---: | ---: |
| 8 | −5.61% | +19.27% | +21.57% |
| 16 | −27.81% | −2.55% | +14.61% |
| 32 | −42.03% | −12.55% | +12.73% |
| 64 | −56.30% | −22.25% | +10.39% |

Empty rounds slow down by 0.17–1.03%. All inputs are ready before timing;
the test excludes model compute, cache-load fences, acceptance, publication
and terminal-request handling. The mixed-batch regressions rule out adopting
this relocation as-is. The follow-up below separates those costs without
changing the frozen source or moving any runtime operation.

The local M20 runbook records the environment, source/script/result hashes,
commands, complete shape results and independent idle checks. No new AIME
score or Nsight capture is available. The original EAGLE3 no-regression gate
remains unmet, and buffered replay remains opt-in without a default capacity.

The follow-up isolates a launch-scheduling problem in the existing endpoint
writer. Its flattened work order repeats the request/value-tile cycle. With
B4 and four value tiles, the original 304-program stride is a multiple of that
16-item cycle. When only row zero needs a write, just 76 programs ever visit
selected work, each handling up to 44 tiles. A 303-program stride spreads the
same 3,312 selected tiles across all 303 programs, with 8–12 tiles each. These
are enumerated logical work counts, not measured GPU occupancy.

A separate grid-size sweep preserves all state/output/history bits. Increasing
the grid can recover mixed-row parallelism but increases empty-round launch
cost. Smaller value tiles also fail to provide a general improvement: all 56
configuration/shape checks pass the existing tolerance, but several lose
bitwise equality or regress despite eliminating compiler-reported spills.
Those negative results remain in the local artifact.

The candidate instead reduces a persistent grid's program count until its
stride is coprime to the request/value-tile cycle. It never exceeds the existing
cap and leaves full grids unchanged. This is static shape arithmetic: it adds
no GPU metadata, allocation, host readback or runtime branch on request flags.
The FP32 reconstruction, flush policy, endpoint/stamp order and cache fences
are unchanged. In particular, it does **not** install the experimental
pre-forward cross-layer capacity flush used by the diagnostic fixture.

A fresh 40-case sweep is bitwise equal to the original endpoint launch for
all output, state and K/U/decay fields. Bracketed CUDA-graph endpoint timings
for mixed B4 are:

| Capacity | Original endpoint | Balanced endpoint | Change |
| --- | ---: | ---: | ---: |
| 8 | 222.233 us | 124.878 us | −43.81% |
| 16 | 297.374 us | 88.764 us | −70.15% |
| 32 | 465.896 us | 133.704 us | −71.30% |
| 64 | 810.322 us | 223.505 us | −72.42% |

The diagnostic's combined endpoint-plus-69-recurrence latency falls
12.18–28.56% in those mixed cases. Empty endpoints remain about 1.3–1.5 us;
their paired changes range from −0.123 to +0.122 us. L8 still regresses versus
keeping capacity reconstruction inline, even after balancing, so this is not
a blanket endorsement of moving flushes. Nor are these serving speedups.

The actual workspace API, rather than the experimental launch override, passes
38 further bitwise comparisons against frozen M19, including B2/B3/B8 and
non-power-of-two capacity 17. The existing endpoint test now parameterizes
one-program, balanced persistent and full grids against the same mixed-row
oracle, with T1/T4, padding, rejected NaN candidates, implicit zero state and
eager/graph execution. **109 kernel/reference tests** pass in 152.58s and
**201 runtime tests plus 77 subtests** pass in 54.71s, with the same three
optional/backend-specific skips. The exact all-files hooks pass again before
the signed-off source commit **`ad28ea43`**, based on `3ee2666f`. The committed
kernel and test hashes match the GPU-validated files.

That source is frozen separately for a real NVFP4 TP8 EAGLE3 L64 run, followed
by a fresh original-baseline restart on the same eight GPUs. Both retain the
M19 graph/overlap and 75-request protocol, with no profiler attached. Both
runs complete without request errors or preemption. Both-node startup checks
confirm the graph/overlap settings; all eight candidate ranks select buffered
recurrence. Hardware, software, native objects, model configuration and input
protocol match the preceding cohort.

All 30 measured batch output multisets match M19 L64, including multiplicities,
and acceptance remains 3.75 at C1 and 3.405 at C4. The two original-baseline
restarts also match all 30 output multisets. This validates the grid edit on
this workload, not general accuracy or token equivalence to the original
implementation.

| Metric | Preceding original baseline | M19 L64 | M20 L64 | Following original baseline |
| --- | ---: | ---: | ---: | ---: |
| C1 median client latency | 1,060.345 ms | 1,068.392 ms | 1,070.813 ms | 1,059.107 ms |
| C4 median client latency | 1,599.908 ms | 1,750.318 ms | 1,773.082 ms | 1,596.933 ms |
| C4 median whole-batch latency | 1,611.606 ms | 1,842.901 ms | 1,848.242 ms | 1,605.830 ms |

Against the preceding/following original baselines, M20 client latency is
**0.99% / 1.11% higher at C1** and **10.82% / 11.03% higher at C4**.
C4 whole-batch latency is 14.68% / 15.10% higher. Against M19 L64, this run is
also slightly slower: **+0.23% at C1, +1.30% at C4**, and +0.29% for the
whole C4 batch. Each restart has three timing rounds and 15 measured C4
batches; its 60 C4 requests are not 60 independent batch trials. These are
observed comparisons, not evidence that the stride change alone caused the
model-level difference. No sample is replaced and no end-to-end benefit is
claimed. **The original EAGLE3 no-regression gate remains unmet.**

The controller validates and stops both owned server steps, then confirms both
nodes idle. The final server needs a second idle probe while CUDA workers
retire; both probes remain in the log. A separate report helper initially
rejects that valid retry history, then is corrected to verify the final probe
while retaining the earlier output. No model or timing run is repeated.
The local M20 phase summary records the complete comparisons, source hashes,
startup/output checks and cleanup evidence. There is no new AIME score,
Nsight capture or broader-corpus result. The next diagnostic target is the
remaining end-to-end cost; relocating capacity flushes still needs its cache
fence and terminal-request contract resolved.

### M21: matched C4 timeline diagnosis

This phase compares frozen original `2e4b5407` with M20's `ad28ea43`, capacity
64. It changes no production code. Both run sequentially on the same new
persistent eight-GB300 allocation, with the real full 93-layer NVFP4 model,
TP8, BF16 activations, FP8 KV and four-token EAGLE3. The software, cached
target/draft revisions, native objects, graph sizes 1/2/3/4, disabled graph
padding, overlap and segmented prefill match the preceding model protocol.

Each source has one C4 warmup and one captured C4 batch: 51,936 input tokens,
51,328 cache hits, 608 new prefill tokens and 256 outputs per request. The
same parent is primed separately before each batch, outside the capture.
Both complete all four captured replies without errors or preemption.
Their warmup/capture output-token multisets match; each capture also matches
all 15 measured C4 batch output multisets from its corresponding M20 run.
This confirms reproduction of the earlier workload, not token equivalence
between implementations or a new L64 AIME result.

The first hardware-traced original attempt stalls during prefill. Its partial
reports, failed replies and host-stack diagnostics are retained separately;
they cannot serve as a valid comparison. After scoped cleanup and independent
idle checks, both completed runs use Nsight Systems 2025.6.3 with explicit
software CUDA tracing, NVTX, graph-node tracing and profiler-API capture.
No CPU sampling or context-switch collection is enabled. The profiler backend
is the only protocol change; neither model source nor cached dependencies
are modified to recover the capture.

Every GPU has prefill batches of 1+1+2 requests. All decode forwards correlate
to complete graphs with consistent node counts: original B4/B2 graphs have
2,945/2,664 kernel nodes, buffered graphs 2,876/2,595. Stop-shutdown warnings
are retained, while neither completed run reports incomplete CUPTI events.
Both reports per source are exported and packaged with clear source/capacity
and rank-group names, alongside a summary and checksums. Independent checks
confirm both nodes idle after each run.

| Captured metric | Original | Buffered L64 |
| --- | ---: | ---: |
| B4 decode graphs per GPU | 70 | 70 |
| B2 decode graphs per GPU | 1 | 13 |
| B4 graph median, rank 0 | 15.246 ms | 16.088 ms (+5.53%) |
| B4 graph medians, all eight GPUs | 15.239–15.261 ms | 16.082–16.101 ms |
| Median of per-graph KDA verify/recurrence launch medians, rank 0 B4 | 7.968 us | 17.872 us |
| Median between-graph gap, rank 0 | 351.088 us | 335.360 us |
| Decode GPU window, rank 0 | 1,109.538 ms | 1,356.424 ms (+22.25%) |

Across ranks, the B4 graph median increases **5.44–5.61%**. The original
acceptance lengths are 3.70/3.64/3.64/3.70; buffered lengths are
3.70/3.11/3.70/3.11. Two buffered requests therefore keep the batch running
for 12 additional B2 rounds. These are actual graph counts, not estimates
from rounded acceptance. Rank 0's B4 graph spans total 1,070.091/1,131.418 ms
and its B2 spans 13.204/181.524 ms. Different generated tokens also affect
other model work, so the graph-span difference is not attributable solely
to the recurrent kernel.

The trace confirms that original accepted replay is already batched across
all 69 KDA layers: one recurrent commit per round, median **136.352 us** on
rank 0, outside the model graph. Removing it does not save 69 launches per
round. Buffered endpoint materialization has 83 launches, median **1.664 us**
and maximum **255.808 us**; its usual cost is small, while history
reconstruction remains in every layer's recurrence. The captured metadata
does not expose exact capacity-flush flags, so timing is not used to invent
a flush count.

The broad rank-0 CPU interval after result synchronization grows from
8.080 to 575.504 us. It includes validation, bookkeeping and rank skew, and
does not translate into a larger typical between-graph gap. One candidate
gap reaches 16.289 ms and remains in the analysis. Neither dropping the
cross-rank validity check nor moving its collective is justified by these
timings. Likewise, overlapping kernel duration sums are not additive
end-to-end costs.

The next work targets history reconstruction and the numerical/acceptance
difference. A reconstruction prototype must retain the state-and-output
route, unified T1/T4 protocol, cache ownership and existing flush policy;
compare it with the current implementation and independent references before
any model performance claim. Batched capacity-flush relocation remains
unimplemented and still needs its load-fence and terminal-request proofs.
This phase adds no unprofiled timing result or AIME score. **The EAGLE3
no-regression gate remains unmet; no default capacity is recommended.**

The local M21 runbook records exact commands, source/environment provenance,
all attempts, validation, graph samples, package hashes and cleanup evidence.
Capture-helper tests pass all four cases; these are diagnostic-helper checks,
not additional model-accuracy tests.

### M22: history reconstruction experiments

Production source remains `ad28ea43`, with the M21 record at `8142a2e8`.
This phase tests isolated, unregistered kernel copies; it does not change
serving dispatch, cache ownership, flush policy or the default capacity.
Experiments run sequentially on one otherwise idle GB300 in the same
persistent allocation used for M21. The cached environment is Python 3.12.3,
PyTorch 2.13.0+cu130, CUDA 13, driver 580.167.08 and
tokenspeed-triton 3.8.10.post20260906. No profiler or model server is running.

The candidate replaces the broadcast product/reduction in
`U^T @ (K * suffix(D))` with TF32x3 matrix multiplication and FP32 accumulation.
Checkpoint and K/U/D storage remain FP32, as does the division-free suffix
product. This still reconstructs state and computes outputs; it is not an
output-only route. Changing reduction order does not imply bitwise agreement.

The first L64/B4/history-32 smoke passes the independent sequential FP32
reference and current-kernel comparison, but is **5.78% slower**. A subsequent
tile sweep covers B1/B4, capacities 16/32/64, empty/mid/near-flush histories,
value tiles 16/32/64 and history tiles 16/32/64. All **198 records** pass the
numerical checks. Some long-history configurations improve 10–22%, while
short-history configurations regress. The best tile for each individual
fixture is not a deployable policy. Compiler register, spill, shared-memory
and Tensor Core instruction records are retained alongside every timing.

A GPU-side short-history scalar / long-history matrix branch then loses its
expected benefit: the same B4/history-32 smoke is **9.87% slower**, despite
passing numerical checks. This does not isolate the compiler or occupancy
cause, and the branch is not adopted.

The next candidate uses one static history tile of 16 for native-input,
four-token L64 recurrence with 12 local heads, K=V=128 and B1–4. Other shapes
retain the current scalar calculation and launch configuration. Confirmation
covers every reachable history length 0–60 at B1/2/3/4, with two seeds:
**488 cases**. Both implementations pass the independent reference, but nine
cases fail the current-to-prototype BF16 output tolerance. Those failures
remain failures; no tolerance is widened. Their timing samples are omitted
by the predeclared correctness gate, not replaced by successful reruns.
An untimed replay of all nine finds one mismatched element per case: adjacent
BF16 values on opposite sides of the independent FP32 reference. Both kernels'
outputs, state and history still pass that reference at the unchanged
tolerances. This identifies rounding-boundary crossings in these fixtures,
not a state-lifecycle failure; it does not explain the earlier full-model
acceptance difference or establish model accuracy for the prototype.

There are **239 paired timing cases**, each ordered current/candidate/candidate/
current. As before, each sample uses a 16-call CUDA graph, 25 replays and five
event-timed repetitions. Observed changes range from **−14.08% to +14.43%**.
The table summarizes mean changes over the timed histories in each interval;
it is not weighted by a model's actual history distribution.

| Batch | History 1–15 | History 16–31 | History 32–47 | History 48–60 |
| --- | ---: | ---: | ---: | ---: |
| 1 | +5.76% | +1.39% | −1.20% | −2.13% |
| 2 | −1.13% | −5.04% | −6.75% | −7.05% |
| 3 | −1.24% | −5.09% | −6.69% | −7.10% |
| 4 | −1.22% | −5.36% | −7.70% | −8.98% |

Even intervals with a negative mean contain slower short-history cases.
These are generated-history, kernel-only measurements, not a claim of
full-model speedup or an unbiased estimate over all confirmation cases.

The static prototype also passes **109 existing kernel/reference tests** in
167.91s and **201 runtime tests plus 77 subtests** in 88.27s. The three skips
are the same two optional FLA prefill cases and AMD-specific indexed decode
case. Tests cover multi-round reconstruction, weak/identity/zero decay,
rejection, flush, page reuse, padding, endpoint materialization and eager/
CUDA-graph execution. Prepared-input and unsupported static geometries keep
their original arithmetic; native L64 cases exercise the new path.

Validation replaces the registered callable's code and its JIT/helper
references only inside the test process, preserving callable identity for the
normal resolver test. It does not edit a frozen archive or installed package.
The first combined test invocation stops at collection because kernel and
runtime suites share a top-level package name; its failure is retained.
Running the suites in separate processes, as in the established runbook,
resolves that harness issue. An earlier smoke preflight also mistakenly treats
Torch's ordinary CUPTI dependency as profiler injection; that failed attempt
is retained, and corrected runs check the actual Nsight injection mappings.

**No reconstruction candidate is adopted in this phase.** The current code
is unchanged, the pairwise output gate remains unsatisfied, and short-history
performance needs work. There is no new full-model timing, NSYS capture or
AIME score. The EAGLE3 no-regression gate remains unmet. The local M22 runbook
retains experimental sources, hashes, commands, every failure, raw timings
and test results. Further work should measure actual history-length usage
and isolate same-input producer/recurrent numerical differences before another
model-level comparison; a favorable tile-sweep minimum is not enough.

### M23: real-model history and same-input numerical diagnosis

Production source remains `ad28ea43`; the preceding experiment record is
`dc36f6fd`. This phase adds local diagnostic helpers and evidence, not a
production kernel, dispatch or arithmetic change. The EAGLE3 no-regression
gate is still unmet.

The run uses a new persistent allocation under the same binding: eight GB300
GPUs across two nodes with a healthy common NVLink fabric. The cached
environment remains Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13,
driver 580.167.08, tokenspeed-triton 3.8.10.post20260906 and FlashInfer 0.6.18.
The real target is the full 93-layer Kimi-K3 NVFP4 model, TP8, BF16 activations
and FP8 KV cache. Target config SHA256 remains
`66ff1cc0486ab1a0901ce5e1f8bc00ad075ff263fc357991d6c23b6e756706b5`;
the EAGLE3 draft revision is `4c48d2bb72134094340067e3012ebe0e822fac37`.
L64, four-token verify, graphs 1/2/3/4 without padding, overlap, segmented
prefill, prefix caching and the existing attention/MoE backends stay enabled.
Both-node audits verify all eight owned workers and no Nsight injection.

A process-local wrapper copies one rank's first local KDA layer into separate,
fixed-address diagnostic buffers allocated before graph capture. It samples
checkpoint/history and conv state after the layer's load fence, then captures
the produced conv, gate, output and candidate K/U/D before shared scratch is
reused. Recording after ordinary accepted commit also retains every group's
positions, flush flags and accepted counts. These copies and synchronized
readbacks make the run **unsuitable for timing claims**. Production compute,
acceptance, validity agreement, checkpoint ownership and graph inputs are
unchanged; snapshot memory is reported separately from the production budget.

The probe first runs six existing real-cache GPU cases: T1/L8 and T4/L8/L37,
each eager and CUDA graph. Their numerical, lifecycle, rejection and rebind
assertions pass. An independent CPU check validates the resulting 192 valid
rounds, six intentionally invalid-acceptance rounds and 16 tensor fixtures.
The original helper's final count check incorrectly expected 192 total rather
than 198; that failed attempt is retained, and the original artifacts are
verified without rerunning the GPU cases or changing their tolerances.

The real-model client performs one warmup and one diagnostic batch at C1 and
C4: ten continuation requests, each with 256 outputs. Inputs are the same
frozen agentic continuation used in M20: 51,936 prompt tokens, 51,328 cached,
608 new, temperature 0, seed 1 and ignore-EOS. Each batch starts with a cache
flush and the matching frozen parent prime. All ten requests finish without
preemption. Each batch's output-token multiset matches all 15 corresponding
M20 L64 batches; this reproduces the current implementation, not the original
baseline's numerical behavior or an AIME accuracy result.

The ledger contains 313 rounds: five initial short-request rounds, four
parent-prime rounds and 304 continuation rounds. HTTP completion can precede
the overlapped commit recorder: each prime inherits the following stage
marker, and each continuation batch has one unmarked trailing round. The
client's completion snapshot reports 312 because the final trailing record
arrives afterward. The original ledger and failed summary attempts remain
intact. Reconciliation separates parents by their exact input endpoint and
accepted count, then validates trailing slot/end/checkpoint continuity. It
does not reset a failed transition or discard a continuation record.

After that reconciliation, both repeats have the same history distribution:

| Concurrency | Decode rounds | Mean history per live row | Rounds with a capacity flush | Aligned materializations, per group |
| --- | --- | ---: | ---: | ---: |
| 1 | 69 × B1 | 25.78 | 3 | 2 |
| 4 | 70 × B4 + 13 × B2 | 27.54 | 7 | 4 |

All recorded groups agree on history lengths and checkpoints. Across every
continuation, the existing `h + 2*T > L` flush rule, accepted endpoint
advancement and aligned materialization transitions validate. At B4,
**61 of 70 rounds have mixed history lengths**; the average per-round maximum
is 32.10, versus a per-row mean of 27.06. About 45.7% of B4 row observations
have history below 24. This is the distribution a follow-up kernel experiment
needs to cover; uniform histories and a best tile per fixture are insufficient.

After stopping only the owned server and independently confirming both nodes
idle, an offline GPU analysis processes all 18 saved tensor fixtures: four
parent-prime and 14 continuation snapshots, including empty, mixed and
near-flush histories. It uses the original verify and batched accepted-replay
primitives, whose source hash matches the frozen baseline. Both start from
the same independently reconstructed FP32 state. A separate CPU pass checks
all 18 snapshots against the unchanged sequential reference: output and conv
use `atol=2e-5, rtol=2e-4 + BF16_eps/2`; FP32 candidate fields use
`atol=2e-5, rtol=2e-4`. All pass. An initial fixture/ledger equality check
mistakenly included the filename added only after tensor serialization;
its failure is retained and corrected by validating that filename separately.

The same-input comparison isolates a local numerical boundary:

- Original verify versus buffered recurrence, with a common start state and
  common BF16 producers, differs in only 22 of 245,760 BF16 output elements;
  maximum absolute difference is `3.82e-6`.
- Starting buffered recurrence from the independent reconstructed state versus
  the captured output differs in 10 elements, with maximum `1.91e-6`.
- Original accepted replay versus buffered accepted history has a maximum
  state difference of `2.05e-3`, with relative L2 errors `2.01e-4`–`6.21e-4`.
  Original replay computes conv and gate in FP32; buffered history consumes
  the BF16 conv/gate used for verification.
- An independent FP32-conv/FP32-gate recurrence matches original replay's
  state within `2.39e-7` maximum absolute error. The BF16-producer recurrence
  matches buffered history within `7.16e-7`. Convolution rounding contributes
  more than gate rounding in these fixtures; both effects are retained in
  the four-way producer comparison.

This accounts for most of the **sampled layer's local state difference**. It
does not establish all-layer/rank parity, explain EAGLE3 acceptance causally,
or justify changing verification producers. The one-layer original descriptor
call is a numerical diagnostic, not its production all-layer launch geometry.
Raw fixtures include model-derived tensors and stay in local artifacts.

The local M23 runbook retains source hashes, environment and input provenance,
all helper versions/failures, request responses, history ledgers, tensor
fixtures and analysis commands. No serving implementation is promoted, and
there is no new unprofiled performance result, NSYS report or AIME score.
The next kernel experiment should use recorded mixed histories and a rotating
cross-layer working set, keeping producer precision as a separate controlled
question. Any candidate still needs unchanged reference/regression gates and
the original full-model EAGLE3 comparison before adoption.

### M24: cache policies under recorded mixed-history patterns

Production remains `ad28ea43`, with M23 recorded in `90253f7a`. This phase
tests three unregistered kernel copies: evict-first checkpoint loads,
evict-last K/U/D history loads, and their combination. A CPU source-contract
test removes only those cache hints and the registration decorator, then
compares the complete parsed source trees with production. All three match:
no arithmetic, loop, tile, dtype, store, acceptance or flush-policy change.

The experiment reuses the verified persistent M23 allocation. Both nodes are
idle before GPU work; the original server step is confirmed absent. The
eight-GB300 fabric and frozen source/dependency hashes pass the same preflight.
Measurements run on one otherwise idle GB300 with the cached Python 3.12.3,
PyTorch 2.13.0+cu130, CUDA 13, driver 580.167.08 and
tokenspeed-triton 3.8.10.post20260906 environment, without Nsight injection.

Unlike the earlier dense fixtures, the actual Kimi-K3 recipe and target
config allocate the test arena. It contains 48 LCM parents and 69 KDA layers,
with three state/history groups owning disjoint parent ranges. The arena is
1,040,449,536 bytes. FP32 history strides are `(36864, 1536, 128, 1)`;
state strides are `(221184, 16384, 128, 1)`, and raw table rows retain stride
8193. Absolute checkpoints are rebased to `120 + original_checkpoint % 8`,
preserving history-page alignment while ensuring every capacity flush writes
a separate state slot. Timed graph replays cannot evolve their input state.

Nine fixtures cover seven saved M23 continuation snapshots, a C1 empty-history
row selected from a real B4 snapshot, and the recorded maximum-spread vector
`[2, 2, 60, 60]`. The last uses replicated long-history snapshot values with
the recorded lengths; it is not a capture of that complete model round.
Each selected layer's values are copied into 69 independently owned layer
fields. Comparing one warm layer with all 69 layers in rotation is a
**recurrence-only cache-pressure proxy**, not a full model: it executes no
MLA, draft, producer or accepted-commit kernels.

All **18 fixture/working-set cases** pass independent FP32 output, candidate
K/U/D and flush-state references at the unchanged tolerances. Each of the
four implementations also matches the current kernel's complete byte arena
and BF16 output bitwise, and input checkpoints remain unchanged. Those
checks repeat after timing to detect unintended writes or evolving fixtures.

Timings use CUDA graphs and balanced ordering:
current/state/history/combined/combined/history/state/current. A warm-layer
graph contains 16 calls; a rotating graph contains one 69-layer pass. Each
has ten warmup replays, followed by five event-timed samples of 25 replays.
Reported times are per recurrence call, including graph-internal launch gaps.

The rotating working set makes long histories substantially more expensive:
the current kernel's C1/history-60 time rises from 14.86 to 22.61 microseconds;
the mixed `[2, 2, 60, 60]` case rises from 19.73 to 25.61 microseconds. The
cache hints do not remove this cost. Checkpoint streaming helps the initial
C4/history-4 rotating fixture by 7.09%, but slows C1 empty history by 3.15%.
Keeping history resident slightly regresses the tested long-history rotating
cases. Its compiled B2/B4 register count rises from 128 to 130, with no spills;
this is a compiler observation, not proof of a particular occupancy bottleneck.
Checkpoint streaming retains the current register counts.

A follow-up tests checkpoint streaming over **all 70 recorded C4 history and
within-page alignment patterns**, not just the favorable fixture. Values
remain replicated from one saved long-history snapshot, and all 69 target
layer fields rotate. Both kernels pass all 140 reference/bitwise checks.
Using current/stream/stream/current ordering, the observed distribution's
weighted mean changes from **17.2123 to 17.1376 microseconds per call**:
only **0.434% faster**. Individual changes range from −6.704% to +0.136%;
32 patterns improve, 37 regress and one is unchanged. This small isolated
gain neither resolves nor replaces the original full-model EAGLE3 gate.

**No cache-policy change is adopted.** The result rules out these load hints
as a sufficient fix and supplies a reusable mixed-history/real-stride harness
for investigating reconstruction compute and scheduling. Production source,
producer precision and defaults are unchanged. The local M24 runbook retains
commands, source and input hashes, complete timing samples, compiler records,
validation results and both-node idle evidence. There is no new full-model
performance result, NSYS report or AIME score.

### Rebase integration with upstream main

Rebased onto upstream main `eaf66b5b`. Host cache registration retains the
replay-history publication guard while supplying main's newly required API
arguments. Scheduler regression tests now pass the former default values
explicitly. DeepSeek V4.1's history groups declare zero state lag and no replay
dependency, preserving their existing cache behavior; the recipe test checks
both fields. These are integration changes, not KDA arithmetic changes.

On Python 3.12.13 and PyTorch 2.8.0+cpu, the buffered reference suite passes
all 19 tests and the cache-group specification/page-count suites pass 47 tests
with two skipped. The CPU state-commit validity contract test also passes.
A fresh Release build with GCC 13.3.0 passes all 490 scheduler C++ tests;
the rebuilt Python extension passes all 153 scheduler binding tests. The
first binding run had two import failures in subprocesses; staging the source
package beside the rebuilt extension on PYTHONPATH resolves both.
Full runtime and GPU checks were not rerun: the local CPU environment cannot
satisfy the kernel package's accelerator requirement.
Earlier full-model accuracy and performance results do not validate this
rebased revision.

### M25–M27: explicit-layout history reconstruction

M25 repeated the original-versus-buffered full-model NSYS comparison with real
93-layer Kimi-K3 NVFP4, TP8, EAGLE3, C4, CUDA graphs and overlap enabled.
The matched-prefill capture reproduced the slowdown: the rank-0 B4 graph
median increased from 15.249 to 16.207 ms, and the buffered run had 13 B2 tail
rounds instead of one. A separate capture grouped prefill differently and did
not reproduce the overall slowdown; both observations are retained. Six
clearly named reports and a checksum-verified ZIP were delivered. Profiling
does not replace M20's unprofiled performance gate.

M26 tested reconstruction alternatives on one GB300 using the M24 real-stride,
rotating 69-layer cache fixture. This is a recurrence microbenchmark, not a
full model. Cache-policy changes, candidate-loop unrolling and several layout
variants were rejected after numerical differences or regressions. Keeping
history in registers and using a scalar FP32 dot became useful only after
restoring the original full-CTA Q/K normalization layout. No TF32 operand
conversion or tensor-core computation is used.

Across all 70 recorded C4 history/alignment patterns, the BV16 candidate
reduced weighted recurrence time from 17.2557 to 15.0489 microseconds per call
(12.79%). All 210 checks across current and two candidate variants passed the
unchanged independent references and complete-arena/BF16-output bitwise checks.
Values still come from replicated single-layer snapshots. Some separate
warm-cache long-history controls regressed 5.21–8.57%; those results remain in
the record. Neither result establishes E2E performance.

The first broad kernel run passed 95 tests but failed to compile 14
minimum-history cases: the scalar dot requires at least eight reduction
elements. M27 retains the original FP32 outer-product sum for smaller static
tiles and shares the corrected reconstruction between forward and endpoint
materialization. All 14 retained failures then pass. The complete frozen-source
suite passes 109 kernel/reference tests and 201 runtime tests, with 77 subtests
and three optional-dependency/platform skips. Tests cover T1/T4, eager/graphs,
acceptance, flush, cache validity and lifecycle behavior without relaxed
tolerances. The environment is Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13,
driver 580.167.08 and tokenspeed-triton 3.8.10.post20260906 on GB300.

The candidate is now integrated as an uncommitted change on rebased commit
`58318c44`. Registration, public arguments, FP32 persistent fields, native
BF16 producers, accepted-only stores and the width-parameterized execution
path are unchanged. The offset-overflow test now invokes the Gluon helper
directly. The preceding passes used frozen `ad28ea43` runtime/test sources
with process-local kernel injection; they do not validate this rebased
integration. Direct integrated-source regression, a matching scheduler build,
endpoint timings and full-model EAGLE3/AIME validation are still pending.
No default capacity is selected. Local M25–M27 runbooks retain commands,
source hashes, failed attempts and complete measurement samples.

The independent rebased scheduler build subsequently passes 490 C++ tests and
153 Python binding tests. Its staged extension is separate from the serving
venv and all earlier binaries. A paired endpoint-only benchmark covers 20
synthetic 69-layer cases at L8/L16/L32/L64 and B1/B4, including inactive and
mixed rows. Every before/after field comparison is bitwise equal to the frozen
writer. L8/L16 active cases improve, but L64 long-history materialization
regresses 11.33–14.27%. Empty-round cost stays around 1.3–1.4 microseconds.
This regression is retained: sharing a faster forward reconstruction does not
guarantee a faster batched writer. The integrated implementation remains a
candidate for full-model measurement, not a no-regression result or a default.

The first direct rebased-source kernel run stops at collection: two tests still
import the removed `thirdparty.triton.fla_kda_recurrent` module. A repository
search identifies four such references across buffered tests and their
microbenchmark. They now import the relocated `_triton.recurrent` module;
assertions and numerical tolerances are unchanged. The failed source snapshot
and collection report are retained separately from the corrected candidate.

The corrected, frozen rebased source now passes direct GPU validation without
prototype injection: 109 kernel tests; 201 runtime tests with 77 subtests and
three optional-dependency/platform skips; and 527 shared cache/scheduler tests
with 317 subtests, including GDN checks. This uses the newly built matching
scheduler, not the earlier native binary. The source patch SHA256 is
`5e6f8e8290153a8f65e143961ec361e4135fdc956fecfbdf5f757bfc7822ba6c`.
Real-model comparison has started in the fixed order candidate, original
baseline, independent candidate restart, independent baseline restart. Each
uses the unchanged C1/C4 protocol, real NVFP4 TP8 weights, EAGLE3 and CUDA
graphs/overlap enabled. This first attempt subsequently failed before model
loading: the rebased registry imports the DeepSeek V4.1 adapter, which requires
a FlashMLA API absent from the historical serving environment. All eight
ranks exited. No smoke response or performance sample completed, and the
failed logs remain separate from subsequent runs.

A separate dependency overlay now supplies the current repository's pinned
FlashMLA and DeepSelect versions, both `1.0.0.post20260910`; the historical
venv, cached image and native objects are unchanged. Both nodes pass the full
model-registry import and required-symbol checks. Original and candidate use
the same overlay. A new cohort has started with the same immutable candidate,
matching scheduler, input and fixed C1/C4 protocol. The earlier GPU suites
used the preceding optional-package environment, not this repaired overlay.
The first candidate run subsequently completes the full unprofiled protocol:
75 measured requests across 30 batches, plus six warmup batches, with no
request errors or preemption. All eight ranks select buffered recurrence at
capacity 64 and target width four; graph and overlap settings are confirmed
from startup records. C1/C4 median client latency is 1,058.983 / 1,742.155 ms;
whole-batch latency is 1,059.094 / 1,836.072 ms. Median acceptance remains
3.75 / 3.405. All 30 measured output multisets match M20 L64, including their
multiplicities. This checks the frozen continuation, not general model accuracy.

The owned candidate server is stopped only after its complete results are
validated, and both nodes are checked idle before the original baseline
starts. That same-GPU baseline and the independent repeats are still pending.
Historical latency is not substituted for this cohort's baseline, and neither
an EAGLE3 no-regression result nor a new AIME score is claimed.

The first original-baseline startup then stops before loading weights because
two control ports are still busy. It produces no measurement; the failed
attempt and completed candidate remain separate. Subsequent process/socket
checks find no surviving model process or occupied port. The local launch
helper now checks the runtime's exact bind/listen contract as well as GPU
idleness before starting another server. Its occupied/released-port test
passes. An explicitly recorded baseline recovery uses a new case label with
unchanged source, dependencies, ports and measurement protocol. This corrects
the experiment setup, not the model, and does not relax the repeat requirement.

The recovered original baseline completes the same 75-request protocol on the
same eight GPUs, also without errors or preemption. The first completed pair
does **not** meet the EAGLE3 no-regression goal:

| Median latency | Original | M27 candidate | Change |
| --- | ---: | ---: | ---: |
| C1 request | 1,059.933 ms | 1,058.983 ms | -0.09% |
| C4 request | 1,605.961 ms | 1,742.155 ms | +8.48% |
| C4 whole batch | 1,617.570 ms | 1,836.072 ms | +13.51% |

C4 batch medians are higher in all three rounds. C1's sub-one-percent
difference is unresolved variation under the preregistered rule, not a
demonstrated speedup. Each implementation has one completed server restart;
the interrupted four-case sequence and independent-repeat requirement remain
explicit in the report. Both owned servers are stopped after validation and
the nodes independently checked idle. Source, dependencies and protocol are
unchanged during the recovery, and no measured sample is replaced.

Median C4 acceptance falls from 3.67 to 3.405. The source's rounded statistics
uniquely imply 69/70 accounted rounds for the original request groups, versus
69/82 for the candidate, with 30 requests in each group. These are not NSYS
round counts or isolated kernel timings: request windows also include overlap
with other requests' prefill. No corresponding original/candidate output batch
is identical, although every candidate batch still matches M20 L64. The next
numerical investigation should separate accepted-history precision from the
verification-output contract; M23's producer difference is a lead, not yet a
causal acceptance result. Current-source AIME and final performance acceptance
remain open. The local report retains all metrics, failures and source hashes.

## M28–M29: preserve accepted-history precision in one recurrence

The M27 full-model regression remains the latest performance result. This
phase tests the producer-precision lead; it does not establish that precision
caused the acceptance change.

M28 separates BF16 verification from FP32 accepted-history producers using
two existing recurrence calls. All 18 saved real-model single-layer fixtures
pass, with 90 accepted-prefix checks against original replay at unchanged
FP32 tolerances (atol 2e-5, rtol 2e-4). The actual endpoint writer also passes;
its maximum absolute state difference is 7.62939453125e-6. Verification output
is bitwise unchanged on these fixtures, and eager/graph storage agrees.
This intentionally duplicated computation is an isolation tool, not a
production implementation or performance measurement.

M29 implements one recurrence launch. It reconstructs committed history once,
then carries BF16-producer verification and FP32-producer history recurrences
in registers. Conv and gate retain FP32 output; verification rounds them to
BF16 in the kernel. T1 retains ordinary decode's BF16 state arithmetic through
the same forward/commit. No persistent state, eager accepted replay or
output-only route is added. Shared scratch accounting now includes the wider
producer buffers; raw candidates and verification outputs remain BF16.

All 18 fixture producer-rounding checks pass bitwise. The single-pass prototype
passes all 18 fixtures and 140 accepted-prefix checks, now including mixed
acceptance, original replay and the actual endpoint writer. Its maximum
endpoint error is again 7.62939453125e-6. Forty parameterized T1/T4 multi-round
cases pass at unchanged tolerances: softplus/bounded gates, five capacities,
eager/graphs, rejected-history poisoning, repeated flush, request reordering,
physical-page reuse and slot reuse. These use compact or synthetic arenas,
not a new full-model acceptance measurement.

The implementation and references are integrated as uncommitted changes on
`58318c44`. The prototype used the frozen M27 source and matching scheduler;
direct integrated-source suites are still required. The new persistent
eight-GB300 allocation uses the same cached CUDA 13/PyTorch 2.13.0+cu130
environment, driver 580.167.08, tokenspeed-triton 3.8.10.post20260906 and
FlashMLA/DeepSelect 1.0.0.post20260910. Both nodes pass registry, dependency
and healthy/full NVLink checks. Local M28/M29 runbooks retain exact source,
script and fixture hashes, commands, terminal outcomes and environment details.

No new E2E performance or AIME result is claimed. Direct kernel/runtime/shared
cache regressions, repeated real NVFP4 TP8 EAGLE3 timings, a matching timeline
and current-source AIME remain open. Buffered replay is still experimental;
no default capacity is selected.

The first frozen M29 integration passes all 157 kernel tests, but runtime
validation reports seven T1 failures: PyTorch rejects an explicit
`out_dtype=None` in the unchanged-BF16 gate case. The other 194 runtime tests
and 77 subtests pass; three optional/platform tests skip. This is an API
integration bug, not a numerical-tolerance failure. The gate's optional FP32
output argument is now bound once at workspace construction; BF16 uses the
ordinary GEMM call. Both widths still execute the same forward/commit.
The failed snapshot and logs remain immutable. Corrected-source regressions
must complete before any full-model measurement.

The corrected frozen source now passes direct validation without prototype
injection: 157 kernel tests; 201 runtime tests with 77 subtests and three
optional/platform skips; and 527 shared cache/scheduler tests with 317
subtests. Its patch SHA256 is
`5ac8a1d86b599ccba80d0f513bb568006c44a47c1057415f163dbf38b35e0d35`.
Both nodes are independently idle afterward. Full repository hooks pass
after the T1 repair; no commit or push is made.

The real full-model comparison is now running against the frozen original,
with real NVFP4 weights, TP8, EAGLE3, CUDA graphs and overlap enabled. It uses
the same fixed C1/C4 protocol and two independent restarts per implementation.
The first candidate has completed all 75 timed requests across 30 batches,
without errors or preemption. C1/C4 client-latency medians are
1461.59/1791.41 ms; median acceptance is 2.58/3.23. The completed-source audit
confirms all eight ranks use L64/T4 and the expected 11,408,460-byte workspace
per rank. Only shared producer scratch grows: 196,608 bytes per rank.

This precision change has not restored the acceptance observed in earlier
runs. Its output multisets differ from both the historical original and the
BF16-history implementation in all 30 measured batches. Those historical
cohorts are used only to examine output and acceptance changes, not latency.
The same-GPU original and independent restart measurements are still running;
there is no new no-regression pass or AIME score. The controller preserves
all samples and only stops each owned server after validating its full run.

A separate short NSYS pair is prepared for the same frozen sources and GPU
cohort. It retains real NVFP4 TP8, EAGLE3, C4, CUDA graphs and the fixed
continuation. A completion guard prevents it from overlapping the unprofiled
comparison. The CPU preparation test passes; no new capture exists yet.

The first same-GPU original has now completed all 75 timed requests without
errors or preemption. The paired result is a performance failure, not noise:

| Median metric | Original | M29 L64 | Change |
| --- | ---: | ---: | ---: |
| C1 request latency | 1059.12 ms | 1461.59 ms | +38.00% |
| C4 request latency | 1606.49 ms | 1791.41 ms | +11.51% |
| C1 whole-batch latency | 1059.23 ms | 1461.71 ms | +38.00% |
| C4 whole-batch latency | 1618.70 ms | 1796.48 ms | +10.98% |
| C1 acceptance | 3.59 | 2.58 | lower |
| C4 acceptance | 3.67 | 3.23 | lower |

All three rounds are slower at both concurrencies. The rounded request
statistics imply 71 versus 99 accounted verify rounds at C1, and 69/70 versus
79 at C4. These are not measured graph replay counts, and they do not isolate
the cause of the regression. Each implementation has one completed restart
in this pair; the preregistered second pair is running. No samples are dropped
or replaced. Six CPU checks pass for the separate NSYS capture/analysis flow;
GPU capture, AIME and the overall no-regression gate remain open.

The second candidate restart also completes all 75 measured requests without
errors or preemption. C1/C4 medians are 1459.62/1786.74 ms; median acceptance
remains 2.58/3.23. Output multisets match the first candidate in 29 of 30
batches. The remaining C4 batch takes 2972.08 ms, with prefill times of
1683.88–1831.21 ms and acceptance of 3.64/3.70. Its outputs differ, and all
four requests remain in the timing statistics. Logs alone do not establish
why this batch differs. The final original restart is now running; no new
GPU capture or AIME result is claimed.

Both original restarts are now complete as well. All four same-GPU pairings
fail the no-regression gate: C1 client latency increases 37.81–38.37%, C4
11.22–12.30%; whole-batch latency increases 37.81–38.36% and 10.73–11.80%.
All 300 measured requests across 120 batches finish without errors or
preemption. Original output multisets match across both restarts in all 30
batches; candidate output multisets match 29 of 30. Candidate r2's unusual
C4 batch remains included, producing a 2952.51 ms request p95.

The controller exits successfully after validating and stopping each owned
server and independently checking both nodes idle. A separate matched NSYS
pair has started with the same frozen sources and GPU cohort. No capture or
AIME result is available yet. L8/L16 follow-up preparation passes its CPU
protocol check, but no smaller-capacity model has started. Those tests retain
the same arithmetic and sampling rules and cannot begin before the requested
profile package is complete or without enough allocation time.

The first diagnostic capture completes all four requests, but profiler
shutdown triggers the job launcher's peer-termination behavior while the
second node is still converting its report. One report survives; the other
is incomplete and has no recoverable raw trace. This is a capture-infrastructure
failure, not a model-request or unprofiled-timing failure. Both nodes are idle
before a separately labeled recovery starts. The recovery disables automatic
peer termination without changing model arguments, frozen source or validation
gates. The failed attempt and surviving report remain available; no complete
paired timeline or new accuracy result is claimed yet.

The recovered C4 pair is now complete on all eight GPUs, packaged and checked.
Each target decode forward maps to a graph with consistent node counts per
batch; stop-shutdown event-completeness warnings remain documented. The first
attempt's surviving single-node report is preserved separately in the ZIP.

This capture does not reproduce the usual lower acceptance and longer decode
window. Original prefill groups are 1+1+2; the candidate uses one B4 prefill.
Candidate acceptance is 3.64 for all captured requests, versus 3.23 in its
uncaptured warmup in the same server. Warmup/capture outputs differ; this
association does not prove that grouping caused the output difference.

Original executes 70 B4 target graphs and one B2; the candidate executes 71
B4 graphs. Across eight ranks, the candidate B4 median is 2.07–2.18% lower,
and its decode window is 0.21–1.34% shorter. At rank 0, B4 medians are
15.245/14.929 ms and decode windows 1099.619/1097.358 ms. KDA itself is
slower: the median of per-graph recurrent-launch medians is 8.064/17.856 us,
and the 69-launch cumulative median is 555.712/1232.575 us. Other kernels
also change; different outputs, grouping and the recorded upstream rebase
prevent kernel-only attribution. The broad post-sync CPU interval grows but
the median graph gap does not. Device flush flags are unobserved; endpoint
launch counts do not establish active state materialization counts.

The repeated unprofiled E2E gate still fails. The favorable decode capture is
retained, not substituted for those measurements. One fixed C1 original/new
pair is now running on the same persistent allocation: C1 is the largest
repeatable E2E regression and avoids multi-request prefill grouping. Serving
arguments, graph ladder, sources and inputs are unchanged; only client
concurrency is one. Three CPU protocol/accounting tests pass. All ten
non-document worktree changes still match frozen R2; no production edit,
commit or push is made in this diagnostic phase. Smaller-capacity tests
remain unsubmitted; current-source AIME and the final performance gate remain open.

The fixed C1 pair is complete and reproduces the regression. Each capture has
one prefill, matches its own warmup and all 30 corresponding unprofiled C1
outputs from two restarts. Original/candidate acceptance is 3.59/2.58;
their output sequences first diverge at zero-based token index 11. This is
not necessarily the first numerical difference or evidence of its layer.

All eight ranks show 72 original versus 100 candidate target graphs. These
are directly observed CUDA-graph counts, not the earlier inversion of rounded
request statistics. Candidate graph medians increase 3.67–3.92%; decode
windows increase 44.16–46.16%. Rank 0's medians are 12.457/12.920 ms, with
decode windows 939.422/1355.339 ms. The 415.917 ms difference reconciles as
358.125 ms for additional graphs at the original mean duration, 42.239 ms
for changed mean graph duration at the candidate count, and 15.553 ms for
all inter-graph gaps. This convention is exact accounting, not causal attribution.

KDA recurrent-launch medians increase 5.728/15.344 us, and 69-launch
cumulative medians increase 394.784/1060.512 us. The original batched replay
kernel's median is 42.480 us; the candidate does not launch it. The candidate
endpoint writer launches 100 times, but device flags are not observed and
these are not 100 proven active materializations. Graph-gap medians increase
261.664/342.208 us, much less than the broad post-sync CPU interval change.

Both-node request/source/ownership checks, graph correlation and cleanup pass;
the controller exits successfully. Four clearly named C1 reports, analysis and
output-reproduction evidence are packaged, readback-checked and delivered.
The C4 pair remains separate and unchanged. The performance gate still fails;
additional rounds dominate this C1 accounting, so acceptance must be tracked
alongside kernel cost. No new AIME score or production-source change is claimed.

### M30: isolated short-history reconstruction validation

An unregistered frozen-source copy tests token-order FP32 K/U/D reconstruction
only when the existing compile-time history tile is smaller than eight. Long
tiles retain the current implementation; there is no device-side algorithm
branch, new state store, producer change, acceptance change or output-only
route. Earlier long-history serial regressions are not discarded. At L8/T4,
the unchanged capacity rule flushes positive history on the next forward;
shorter reconstruction trades against more state writes rather than guaranteeing
a benefit. Small T1 capacities also need coverage before any adoption.

The CPU AST check passes and verifies unchanged code outside the declared
helper/registration change. After the C1 package is delivered and both nodes
are independently idle, the isolated frozen recurrence suite passes all 100
cases in 246.46 seconds on a GB300 with CUDA 13 and PyTorch 2.13.0+cu130.
It covers T1/T4, five capacities, prepared/native/dual-producer inputs, both
gate forms and eager/graph execution across multiple rounds. Existing FP32
and BF16 tolerances are unchanged. The tested prototype remains separate
from the frozen M29R2 serving source; no new commit or integration pass is claimed.

A same-GPU recurrence and actual-endpoint-writer experiment follows. Its 18
fixed geometries cover L8 short histories, B1/B4 and mixed rows, plus longer
tile controls. It uses actual cache recipe strides and FP32 producers, with
one recorded layer's values replicated across 69 independent layer fields.
Shortened histories and rebased positions are synthetic, not recorded L8
model trajectories. Timing uses balanced original/prototype/prototype/original
order for single-layer and rotating-layer working sets; reference and writer
checks must pass first. All outputs, source hashes and compiler resources are
recorded. This does not establish model acceptance, an E2E speedup or AIME
accuracy. The M29 acceptance regression remains unresolved; renewed full-model
performance and the remaining plan gates are still required.

The paired experiment completes on the same GB300. Both variants pass all
18 geometries across 69 layer fields and 280 combined actual-writer checks;
maximum endpoint error is 1.90735e-6 and eager/graph storage is bitwise equal.
Timing confirms a tradeoff, not a generally faster replacement. In the
rotating-layer L8 cases, history length one improves 6.66% at B1 and 4.82%
at B4, but length three regresses 4.65%/4.34% and length four regresses
10.81%/9.13%. The two mixed B4 cases regress 8.39–8.81%. Long-tile controls
are effectively unchanged in this measurement. Registers decrease from 168
to 154 for the short tile, with no spills in either variant; that does not
establish the cause of the timing changes.

Do not adopt the prototype based on its passing numerical gate or its best
short-history cases. The failed performance hypothesis and all measurements
are retained. Production remains frozen M29R2; there is no full-model result
for this prototype and no evidence that its acceptance recovers. The remaining
small T1-capacity extension is not claimed complete. Next, separate capacity
effects from arithmetic changes with the prepared, unchanged-source L8/L16
full-model comparison and matched original restarts on a fresh persistent
GPU cohort. Keep the existing E2E, AIME and lifecycle requirements intact.

### M31: fixed-capacity acceptance investigation

The user reprioritized acceptance after the obvious kernel issues, asking to
focus on one capacity and concurrency first. The prepared capacity sweep is
therefore not submitted. A fresh persistent TP8 allocation is reserved under
the same binding; the completed, independently idle previous allocation is
released without deleting any artifacts. The serving source stays frozen M29R2.

L64/C1 is the initial case because its lower acceptance and additional target
rounds reproduced in both independent unprofiled restarts and NSYS. Three
full-model controls distinguish the frozen pre-plan original, current source
without buffered replay, and the same current source with capacity64. Each
uses identical real weights, graph/overlap settings, frozen input and sampling,
with one warmup and three repeated continuations. This is a causal-isolation
control, not a replacement for the repeated performance gate or AIME.

The follow-up compares intermediate results only while token prefixes match,
starting from prefill state and early verify outputs before following target
and draft logits. It must identify the earliest numerical difference before
attributing an acceptance change to rounding or history reconstruction. The
first differing generated token alone is not enough. No forced acceptance,
sampling modification or new production arithmetic is part of the initial
control. Bootstrap checks the cached environment, all eight GPUs and common
healthy fabric before any model starts. Those checks pass on the new eight
GB300 GPUs, and the original control starts after an additional case-specific
idle/preflight check. The fixed three-arm controller is running; results are
not available yet. The first CPU protocol check caught a stale copied comment;
the comment was corrected and the check passes before model submission. No
production code or numerical threshold changed in that preparation.

The original and current-unbuffered controls have now completed. Each produces
four identical 256-token continuations, and the two sources match token for
token. Both report acceptance length 3.59 and acceptance rate 0.8638. For this
case, the source/rebase control does not reproduce the regression; the buffered
arm is still running. This narrows the next tensor comparison to identical
current source with and without buffered replay, without establishing which
operation causes the difference.

A separate read-only observer is prepared for that comparison. Its CPU tests
cover independent snapshot storage, generation isolation and the common target
forward seam, including segmented prefill that bypasses the model-runner seam.
GPU graph validation and instrumented serving have not run yet. Diagnostic
readbacks invalidate timings, and their output must reproduce the uninstrumented
control before it can explain that control's acceptance behavior.

All three controls complete and pass source/eight-worker audits on the fresh
cohort. Buffered L64 reports acceptance length 2.58 and rate 0.5253 in all four
responses. Its outputs are repeatable within the arm and first differ from both
unbuffered controls at generated-token index 11 in every cross-request pairing.
The original/current-unbuffered arms remain identical at 3.59 and 0.8638.
Thus the source control does not explain this case's gap; enabling buffered
replay does reproduce it. This is not yet a diagnosis of the first numerical
difference, an accuracy pass, or a no-regression performance result.

All completed model workers are stopped by their exact owned steps and the
nodes are idle. The separate observer's GPU graph/storage smoke test starts
afterward. Four CPU checks pass, including same-prefix exclusion and FP64
logical-state interpretation; raw test reports are retained. Static inspection
also finds differing Q-scale placement and FP-contraction settings between
verify kernels. These are candidates for the matched-input diagnosis, not
proven explanations of the full-model acceptance change.

The observer's GPU smoke passes on both nodes, including actual runtime hook
imports, captured replays with changing history lengths and invalid rows,
stable snapshot pointers, and eager/graph generation isolation. Nodes are idle
afterward. The instrumented current-source on/off pair is now running with
unchanged real-model, graph, overlap and sampling settings; all helper hashes
are recorded before launch. No full-model tensor comparison is available yet.

The instrumented current-unbuffered arm completes after intermittent weight-
loading I/O waits. Both continuations reproduce the uninstrumented output IDs
and acceptance length/rate (3.59/0.8638). All eight ranks save 30 forward
snapshots. A rank-0 self-repeat audit finds bitwise-identical observed fields
for continuation prefill and the first three verify rounds, including all
69 KDA layers. Other ranks' phase/count checks pass; their tensor equality has
not yet been checked by this self-audit. The buffered arm is now running.

Actual snapshots include an in-flight decode after each one-token parent
prime. A separately versioned post-processor retains and labels those rows
but excludes them from the continuation's AR interpretation; accepted-token
snapshots are also checked against HTTP output positions. The running observer
and controller remain unchanged. Cross-arm tensor diagnosis is still pending.

### M32: first same-input divergence precedes history replay

The instrumented on/off pair now completes on the same frozen M29R2 source,
eight GB300 GPUs, real full-model NVFP4 TP8, EAGLE3 T4, L64/C1, CUDA graphs
and overlap enabled. The input, draft revision and sampling remain the M31
controls described above. Each arm's two continuations reproduce its own
uninstrumented output IDs and acceptance: 3.59/0.8638 without buffering,
2.58/0.5253 with buffering. All source and eight-worker audits pass. Snapshot
readbacks invalidate timing; this is neither a performance nor an AIME result.

All eight ranks are compared by absolute input position and accepted prefix,
not forward ordinal. Continuation prefill KDA outputs match across all ranks.
The first verify has identical candidate IDs and empty history, with checkpoint
equal to the accepted endpoint. Ranks 1 and 4 already differ at KDA layer 0:
one BF16 output element each, maximum absolute errors 4.65661e-10 and
3.81470e-6 respectively. Initial recurrent/conv states, raw projections and
BF16-rounded conv/gate producers are identical at both sites. The first local
difference on rank 0 is layer 2, also with matching inputs and empty history.
Thus the earliest observed discrepancy is in verify arithmetic, before any
history reconstruction can contribute; it is not evidence of an accumulated
history error. Later layers can receive already-divergent activations.

The first four common verify windows retain identical candidate IDs and
accepted counts (4,3,2,1), despite differing logits. In the fifth common window,
the first target choice differs and generates the observed output-token
divergence at index 11. This ordering does not by itself prove that fixing the
first tiny kernel difference restores the complete trajectory. A bounded
same-input scale-placement/FMA ablation is running on the saved first-window
fixtures. Production source, accepted-state arithmetic and tolerances remain
unchanged; long-history parity and a fresh uninstrumented AR test are still
required before claiming a fix.

The first-window arithmetic ablations complete on 207 layer fixtures from
three ranks, including both ranks with the earliest layer-0 difference.
Moving Q scaling before the output dot removes that layer-0 discrepancy.
Explicit state-update and projection-reduction FMA reduce further differences,
but do not eliminate them: 16 BF16 output elements still differ across the
207 fixtures. Changing compiler contraction, value-tile width or the equivalent
two-dimensional layout does not establish complete parity. The original
compiled projection contracts selected local multiply-adds that the buffered
kernel does not. One intermediate Gluon layout experiment fails compilation;
the corrected run and the failure are both retained. None is adopted.

### M33: separate verify arithmetic from accepted-history arithmetic

A private diagnostic keeps M29R2's candidate history, checkpoint and metadata
handling, copies the reconstructed pre-verify state to scratch, and replaces
only attention output with the existing ordinary split-producer verify kernel.
No accepted length, token, state from another run, or sampling setting is
substituted. Redundant compute and scratch make its timings unsuitable for
performance claims; this is not a proposed runtime implementation.

For all 207 first-window fixtures, its output is bitwise equal to ordinary
verify and its written history is bitwise equal to native M29R2. Captured
replay, invalid rows and subsequent valid reuse pass the same exact checks.
An additional 828 fixtures cover recorded histories of length 4,7,9,10;
native-history bitwise parity, native observed-output reproduction, graph
replay and invalid-row preservation all pass. The fixed L64/C1 full-model
comparison uses a fresh current-unbuffered restart followed by the diagnostic;
its completed results are recorded below. Production code remains unchanged
by these diagnostics, and the overall Eagle3 no-regression gate remains open.

A CPU-only logit audit also locates the first choice flip. The two leading
unbuffered target logits are tied at 21.0 in the first differing output
position; buffered logits become 21.25 and 20.75 and reverse their order.
Earlier matched windows choose the same targets and accept the same counts.
This explains where the generated trajectories separate, not whether verify
alignment alone fixes the later acceptance-rate gap.

The full-model pair completes on the same eight GB300 GPUs, full real NVFP4
TP8 model, EAGLE3 T4, CUDA graphs and overlap, fixed L64/C1 SWE-smith
continuation and unchanged sampling used by M31. Both arms run one warmup
plus three repeated 256-token continuations. The frozen M29R2 unbuffered
restart reproduces all prior output IDs and 3.59/0.8638 acceptance. The private
M33 diagnostic produces four identical outputs at 3.70/0.8986. Its source is
M29R2 plus an uncommitted, retained diagnostic patch; no production commit is
made. Source, environment, input and eight-worker audits pass, and the exact
owned model steps are stopped before any subsequent GPU test.

The recovered AR does not mean identical generation: diagnostic output first
differs from unbuffered at index 11 and from native buffered at index 19.
Changing verify arithmetic therefore changes the later trajectory and AR in
this case, but does not restore the original trajectory or establish that
verify accounts for the whole gap. Continue to isolate accepted-state
arithmetic on matching inputs; do not infer correctness from AR being higher.

The diagnostic calls ordinary verify with kernel-local PDL disabled. A separate
post-serving control checks both PDL settings against recorded ordinary outputs
at the first verify and first differing-output window, on all eight ranks:
1,104 layer fixtures are bitwise equal in all three comparisons. This removes
that numerical confound for the tested windows, not a general PDL equivalence
claim. The nodes are idle after the check.

The model runs themselves complete, but the controller exits with an error in
final aggregation because a copied assertion still expects three rather than
the preregistered two arms. A separate post-processor retains the failure and
rebuilds the aggregate without rerunning inference or replacing any request.
The original cross-case environment, input and distinct-step checks are also
verified for the two-arm result. All raw artifacts and helper/source hashes
are retained. This is an acceptance-isolation result, not a production fix,
performance pass or current-source AIME validation.

### M34: separate accepted-history generation from reconstruction

This diagnostic uses the saved first T4 verify and its observed next state,
with all four inputs accepted, from M32's real NVFP4 TP8 model. Capacity stays
L64 and concurrency stays one. All eight ranks and 69 KDA layers per rank
are included. The original batched replay processes all 69 local layers
together, preserving its layer-dependent gate launch rather than treating
each layer as an isolated serving call.

Source is the same uncommitted M29R2 snapshot on parent
`58318c4430b88db9159d4a2d8af38e8c3c768daa`; production code is not changed.
The diagnostic runs on the existing GB300 environment with PyTorch 2.13.0,
CUDA 13.0 and the recorded kernel dependencies. The fixtures come from the
same graph-enabled, overlap-enabled EAGLE3 T4 SWE-smith continuation as M31.
This is a GPU tensor replay, not a fresh full-model, graph or performance test.
Local artifacts retain allocation, software, source and fixture hashes.

All 552 layer fixtures pass two bitwise reference checks: unmodified original
replay matches the saved next recurrent/conv state, and the private probe
preserves native output and K/U/decay writes. The probe only publishes the
final register-local history state; an AST check also excludes arithmetic edits.

| Same-input comparison | Maximum absolute state difference |
| --- | ---: |
| Original replay vs observed next state | 0 |
| Native history-generation state vs original replay | 9.54e-7 |
| Sequential unfused native K/U/decay vs history-generation state | 0 |
| Native endpoint reconstruction vs history-generation state | 1.91e-6 |
| Native endpoint reconstruction vs original replay | 1.91e-6 |

Every fixed-tolerance check passes with FP32 `atol=2e-5, rtol=2e-4`; no
nonfinite values occur. The exact sequential reconstruction rules out loss
when these particular K/U/decay values are written and read back. Differences
already exist when the values are generated, and the closed-form endpoint
writer adds a separate floating-point ordering difference. The decay values
also differ before reconstruction, so changing only state-update FMA cannot
make the original and native paths identical.

This does not establish the contribution of either error to the full AR gap.
Next capture original replay coefficients with a bitwise endpoint-fidelity
check, then test whether reconstruction still differs with identical original
coefficients. No diagnostic is adopted as a production fix.

### M35: original coefficients still expose reconstruction-order differences

The same L64/C1, real NVFP4 TP8 first-window fixtures and frozen M29R2 source
are used. This remains a GPU tensor diagnostic in M34's recorded environment,
not a fresh full-model or performance measurement.

The first original-replay coefficient probe fails its bitwise endpoint
check. A 69-layer ablation finds that writing K or decay changes the compiler's
normalization layout from one key per thread to four. No-store and U-only
variants remain exact. Those failed observations are retained, not treated
as original coefficients. A separate strided observation buffer preserves
the original layout and passes endpoint bitwise equality on all eight ranks
and 69 local layers. Its source AST check passes too; compiled artifacts and
GPU fidelity, rather than source similarity alone, establish the reference.

The resulting same-input checks separate coefficient generation from how
the coefficients are applied:

| Comparison across 552 layer fixtures | Maximum absolute difference |
| --- | ---: |
| New vs original K | 1.79e-7 |
| New vs original U | 7.43e-7 |
| New vs original decay | 8.94e-7 |
| Original coefficients, sequential FMA vs original state | 0 |
| Original coefficients, current endpoint writer vs original state | 2.86e-6 |

All fixed FP32 tolerances pass, with no nonfinite values. Applying the original
coefficients in the original sequential FMA order reproduces the observed
original state bitwise. Applying those exact same coefficients with the
current closed-form writer does not. Reconstruction ordering is therefore a
separate source of floating-point differences even after coefficient parity
is solved; changing only the producers cannot restore exact state here.

Together with M32/M33, the evidence now distinguishes verify arithmetic,
accepted-coefficient generation and accepted-state reconstruction. It does
not assign a fraction of the full AR gap to each: later acceptance is measured
on different generated trajectories. No new AR, AIME or performance result
is claimed, and no diagnostic code is adopted. The next candidate must test
these arithmetic changes on matching inputs before a fresh graph-enabled
full-model AR comparison. Numerical parity does not waive the original
Eagle3 no-regression gate or justify a separate serving/cache path.

### M36: align verify without the diagnostic's extra launch and scratch

The private T4 candidate shares one history reconstruction between two
register-only recurrence loops in the same kernel: original-order BF16
verification first, FP32 accepted-history generation second. It does not
repeat verification, allocate a full-state scratch or add a kernel launch.
The value tile and register layout match ordinary small-batch verify, with
compiler FMA enabled. Candidate-history arithmetic also changes slightly;
this is not an output-only substitution with bitwise-unchanged history.

On all eight ranks and 69 local layers, the BV8/FMA candidate matches original
verify bitwise on 552 empty-history fixtures. The controls remain informative:
disabling FMA leaves 172 differing BF16 output elements; BV16 with FMA leaves
22. Candidate K/U/decay comparisons stay within the unchanged FP32 tolerances.

A separate check covers 2,208 fixtures with recorded history lengths 4,7,9,10.
Every candidate output matches ordinary verify on the native reconstructed
state bitwise. The native control reproduces observed M32 outputs and history
before comparison. Candidate eager/graph output and history are bitwise equal;
width-zero and invalid rows preserve their buffers, and valid reuse passes.
The copied candidate differs only by a trailing blank line between the two
artifact directories; both file hashes are retained.

Rank-zero, first-layer hot-cache measurements use 32 launches per CUDA graph,
nine event samples and a native-before/candidate/native-after sandwich:

| History length | M29R2 before / after, us | Candidate, us |
| --- | ---: | ---: |
| 4 | 8.274 / 8.274 | 7.948 |
| 7 | 8.302 / 8.346 | 8.013 |
| 9 | 9.150 / 9.184 | 9.068 |
| 10 | 9.202 / 9.183 | 9.162 |

These recurrence-only timings range from roughly unchanged to 4% faster.
They exclude producers, accepted commit and the full-model working set;
they do not satisfy the Eagle3 performance gate. Source remains a private
uncommitted candidate on the frozen M29R2 environment, GB300, PyTorch 2.13.0
and CUDA 13.0, using the saved real NVFP4 TP8 SWE-smith continuation. No new
AR score or AIME result is claimed.

### M37: generalize the candidate before a serving comparison

A separate frozen source copy extends M36's structure to the existing T1/T4,
native BF16/FP32 and prepared-input contracts. A compile-time phase loop shares
the recurrence body; only dual-precision multi-token input needs the second
phase. T1 and prepared inputs retain non-contracted arithmetic. Cache ownership,
metadata, launch count, persistent state and scratch budgets do not change.

Four new GPU cases compare native BF16/FP32 output with ordinary verify at
B1/B4 and check that no-flush verification leaves the checkpoint unchanged.
They run alongside the existing numerical, graph, flush, multi-round and
runtime regression suites. The generalized source must pass its own checks:
M36's results do not automatically validate a different compiled kernel.
No candidate is adopted or marked as a full-model correctness/performance pass.

The first generalized snapshot finishes with 23 failed and 138 passed kernel
tests; runtime tests do not start. Twenty-two failures are compile errors in
the small-history reconstruction branch: its sliced reduction is added to
the new plain 2D state layout without conversion. R2 explicitly converts the
reduction to the caller's layout before addition; the shared endpoint helper
uses the same operation, with no conversion needed when layouts already match.

The remaining failure is in a newly added, overly broad bitwise assertion for
BF16-only B1 input: one of 6,144 output elements differs by 4.77e-7. Both
FP32-producer B1/B4 exact comparisons pass. R2 makes that distinction explicit:
the actual dual-FP32 Eagle3 path still requires exact verify output; BF16-only
input is compared at its established `atol=2e-5, rtol=2e-4` numerical contract.
This revises the new BF16 assertion, not any pre-existing reference tolerance,
and does not claim BF16-only bitwise parity. The failed result is retained.
R2 is frozen separately and reruns the full suites before any serving test.

R2 completes with 161 kernel tests, 201 runtime tests and 77 subtests passed.
Three existing environment-specific cases are skipped: two optional FLA
prefill comparisons and one AMD contract. Source identity and idle checks
pass. This validates the generalized kernel's existing contracts, not full
model acceptance. A fresh L64/C1 comparison is prepared on another eight-GB300
cohort in the same dependency environment, with real NVFP4 TP8 weights,
Eagle3 T4, CUDA graphs and overlap enabled. The final wrapper first rechecks
saved real tensors against M36; the serving run then compares current
unbuffered code with this candidate. No production adoption or new accuracy
score is claimed.

M38's final-wrapper check then passes all 2,760 saved real-tensor cases:
eight ranks, 69 local layers and five windows with history lengths 0,4,7,9,10.
Output matches the validated M36 candidate bitwise, history remains within
fixed tolerances, and eager/graph/invalid-row checks pass. The native control
still reproduces observed outputs; empty-history output also matches ordinary
verify directly. A fresh full-model two-arm AR comparison is now running.

### M38: verify-aligned candidate improves AR, but does not restore it

The fresh two-arm comparison completes on the same eight GB300 GPUs: full
real 93-layer NVFP4 TP8 Kimi-K3, real Eagle3 T4, BF16 activations, FP8 KV,
CUDA graph sizes 1/2/3/4, padding disabled and overlap enabled. Source is
parent `58318c4430b88db9159d4a2d8af38e8c3c768daa` plus frozen M29R2 and
the uncommitted M37R2 candidate; complete patch hashes, dependency versions,
GPU identities and commands are retained with the private run artifacts.

The same SWE-smith continuation uses 51,936 input tokens, 51,328 prefix-hit
tokens, 608 new prefill tokens and 256 output tokens. C1 only, temperature
zero, seed one, ignore EOS; each arm has one warmup and three repeats, with
cache flush/reprime before each. There is no observer, profiler, forced
acceptance, state import or sampling change.

| Implementation | Acceptance length | AR | Repeat median request latency |
| --- | ---: | ---: | ---: |
| Current unbuffered control | 3.59 | 86.38% | 1045.57 ms |
| M37R2 one-launch candidate, L64 | 3.11 | 70.33% | 1229.42 ms |

All four responses within each arm are identical. Control output matches
the earlier original/current controls exactly. Candidate output first differs
from control at generated index 11 in all 16 cross-arm comparisons. It first
differs from the earlier native M29 output at index 34 and from the diagnostic
M33 output at index 19. Both source/eight-worker audits pass, no preemption
occurs, both model steps are stopped and the nodes are confirmed idle.

The candidate's AR is higher than historical M29's 52.53%, but lower than
control and M33's diagnostic 89.86%. These runs follow different generated
trajectories: this does not assign a fraction of the AR gap to verify versus
history arithmetic. The fresh pair still shows 17.58% higher median request
latency. This is a narrow AR diagnostic, not independent-restart/C4 performance
validation, and clearly not a no-regression pass. It has no AIME score.

The candidate remains isolated; root production stays at M29R2. Next compare
the actual candidate's accepted K/U/decay and endpoint with original replay
on identical saved inputs, retaining original-probe bitwise fidelity and all
fixed numerical tolerances. Aligning same-input verify output alone has not
resolved acceptance, so further capacity sweeps are still deferred.

### M39: accepted-state differences remain in the current candidate

Using the same recorded original first T4 window, the actual M37R2 wrapper
passes bitwise original verify output on all 552 layer/rank cases. Original
replay reproduces the recorded next recurrent and conv states exactly, and
the coefficient observer again preserves the original endpoint bitwise.
The source-level observer check also passes. Rounded conv/gate values match
the original BF16 producers in every fixture.

The FP32 accepted path still differs. Candidate versus original K/U/decay
maximum errors are 1.79e-7 / 9.61e-7 / 8.94e-7. Applying candidate coefficients
in the original serial FMA order leaves a maximum state difference of 9.54e-7;
the actual writer differs from that serial result by up to 1.91e-6. Original
coefficients plus serial FMA match original state exactly, while the same
original coefficients through the actual writer differ by up to 2.86e-6.
All unchanged FP32 tolerances pass; no nonfinite values occur.

This is a fixed-input numerical isolation, not another AR run. It confirms
that coefficient generation and reconstruction ordering remain distinct
differences in the candidate, without assigning their contributions to later
acceptance. Next isolate FP32 conv accumulation, gate reduction and key
normalization while preserving the validated verify arithmetic. Same-input
GPU evidence must confirm source-level hypotheses. The current candidate
is not adopted, and the no-regression and current-version accuracy gates
remain open.

### M40–M41: pin down normalization and history-update arithmetic

The same 552 real first-window fixtures are used throughout, on GB300 with
the frozen M37R2 source and unchanged original-replay oracle. These are
private kernel ablations, not serving changes or new acceptance scores.
Original replay still reproduces recorded state exactly; its coefficient
observer preserves the endpoint bitwise. Every ablation leaves ordinary
verify output unchanged.

M40 varies history-only convolution order, original log-decay and one-warp
key normalization. The latter reduces differing K elements from 673,476 to
181,283; adding replay-order convolution reduces that to 158,440. Using the
original gate output removes decay differences, but is diagnostic input,
not yet a replacement gate implementation. The alternate convolution rounds
three BF16 elements differently across all fixtures. One comparison exceeds
the existing FP32 tolerance; that failed attempt and observation are retained.
It cannot replace the shared verify producer unchanged. Reference tolerances
are not relaxed.

Compiler inspection then finds that original normalization rounds squared
keys before summing, while the candidate contracts some squares and additions
into FMA. M41 forces separately rounded squares only in history normalization.
With replay-order convolution, K now matches original replay bitwise across
all 3,391,488 elements. A separate history state-layout change reduces U
differences further, but does not eliminate them: the best combination still
differs in 1,742,231 U elements and 37,445,738 serial endpoint elements, with
maximum errors 5.27e-7 and 9.54e-7 respectively. All existing state/coefficient
tolerances pass; these are not bitwise matches.

The remaining PTX shows a concrete update-order difference: the candidate
can fuse old-state times decay into an already rounded correction product,
whereas original replay fuses correction times key into the rounded decayed
state. Next isolate that FMA choice without changing verify arithmetic.
Source/helper hashes, five compiled M41 variants and complete rank results
are retained with the run artifacts. M38 remains the latest full-model result:
AR is not recovered and the performance gate is still failed. No new AIME
result or production adoption is claimed.

### M42–M43: exact accepted coefficients on identical real inputs

M42 explicitly chooses `fma(U, K, rounded(S * D))` for the history update.
This reduces differing U elements from 1,742,231 to 1,506,635, but leaves
first-token U differences unchanged. Inspection of original replay reveals
another fusion boundary: `fma(conv_acc, sigmoid(conv_acc), -projection)`.
Storing the post-SiLU value first, even in FP32, loses the rounding behavior
of that fused multiply/subtract.

M43 retains the pre-SiLU value in private diagnostic scratch and uses the
original fusion boundary when computing U. Combined with the established
normalization, convolution, gate and state-update controls, all 552 fixtures
now match original replay bitwise: 3,391,488 elements each for K/U/decay and
108,527,616 elements for the serial-FMA endpoint. Every ordinary verify
output remains bitwise unchanged. Original replay versus recorded state,
observer fidelity, finite checks and unchanged numerical tolerances all pass.
The GPU runs complete with clean source and pre/post-idle checks; original
probe source tests also pass.

This establishes an exact arithmetic construction for the saved first T4
window, not full-model correctness or acceptance recovery. The gate is still
supplied by the original replay oracle, and state comparison uses a serial
diagnostic reconstruction instead of the current closed-form writer. M44
next computes the same gate directly from ordinary model inputs. Actual
producer/reconstruction integration, graph and multi-round regression, and a
fresh L64/C1 full-model AR comparison remain required. Diagnostic extra
launches/scratch are not a production performance result. Root code remains
M29R2, with M38 still the latest failed AR/performance gate and no new AIME.

### M44–M45: remove oracle inputs and validate paged ordered reconstruction

The existing standalone gate function fails the first exact comparison even
at the original BT4/BK32/one-warp geometry: 100 of 6,144 values differ, with
maximum error 3.58e-7. That attempt is retained. Reusing the actual descriptor
gate kernel for one layer, with only ordinary input pointers and a separate
output buffer, passes all 552 fixtures. This independently generated gate
still produces exact K/U/decay and serial endpoint. No original gate or state
result is supplied to the candidate calculation.

M45 then replaces the diagnostic flat-history reference with a reusable
paged reconstruction helper. It loads eight history entries together but
applies them in original token order, using a rounded decay multiply and an
explicit correction FMA. All 552 real coefficient sets reproduce the original
endpoint bitwise in eager and CUDA-graph execution. A further 260 synthetic
cases cover every history length from 0 through 64, four checkpoint offsets,
permuted pages, interleaved field strides and zero decay. Graph refreshes with
changed lengths/offsets, including empty history, also match the independent
one-token-at-a-time reference exactly.

There is a cost to preserving this order. C1 hot-cache reconstruction-only
measurements use 32 launches per graph and nine event samples:

| History length | Existing before / after, us | Ordered, us |
| --- | ---: | ---: |
| 4 | 2.698 / 2.669 | 2.805 |
| 7 | 2.816 / 2.816 | 2.915 |
| 9 | 4.016 / 4.043 | 4.003 |
| 10 | 4.046 / 4.034 | 4.138 |
| 32 | 6.847 / 6.836 | 7.343 |
| 63 | 12.008 / 12.037 | 13.151 |
| 64 | 12.131 / 12.149 | 13.283 |

These are neither recurrence nor full-model timings. Some histories regress;
that result is retained rather than presented as a speedup. The helper has
not yet replaced production recurrence/endpoint reconstruction. Next remove
the candidate gate's CPU-built descriptor, integrate the proven producer
and reconstruction arithmetic with preallocated workspace, and validate
multi-round/graph behavior before a fresh L64/C1 AR pair. The current source
and latest full-model AR/performance status remain unchanged.

M46 removes the private CPU-built gate descriptor: a direct-pointer kernel
retains the original descriptor kernel's reduction body, static geometry and
unknown-alignment assumptions. Its independently computed gate, K/U/decay
and serial endpoint are bitwise equal to original replay on all 552 fixtures;
verify output is unchanged. Source/pre/post-idle checks and the original-probe
source test pass. This proves a directly callable arithmetic building block,
not that pointer alignment alone explained the earlier standalone failure.
It has not been integrated into the serving workspace or performance-tested.

The next integration should keep these boundaries explicit:

1. Preserve BF16 verify producers and their validated arithmetic. History
   needs replay-order K and the value **before** SiLU, not a stored post-SiLU
   value. Build them with the existing conv preparation rather than borrowing
   reference output or replaying accepted state after sampling.
2. Compute the independent history gate from normal model inputs. Any shared
   scratch belongs to the existing workspace, must be preallocated and
   included in recipe accounting, and needs stable eager/graph views.
3. Use the same ordered reconstruction in forward and endpoint materialization.
   Keep scheduler/cache ownership, position validation, flush and commit
   sequencing unchanged; width one remains the same parameterized path.
4. Freeze the resulting source, validate real multi-round inputs and existing
   graph/lifecycle tests, then repeat the same real NVFP4 TP8 L64/C1 serving
   comparison. Only that run can establish AR recovery. The original C1/C4
   performance and current-version AIME gates remain separate and open.

### M47: integrate the arithmetic before repeating the AR test

The private candidate now computes replay-order K and pre-SiLU V in the
existing conv launch, writes a separate history gate from model inputs, and
uses the ordered reconstruction in both forward and endpoint materialization.
Verification keeps its original producers and arithmetic. History scratch is
shared across layers, preallocated, and included in the memory budget: at
batch 4, width 4 and 12 local 128-dimensional heads, it adds 288 KiB per rank.
The separate gate launch remains a performance cost to measure.

The first integration regression passed 140 kernel tests and failed 23. These
failures exposed two integration mistakes: missing kernel registration and
an unguarded gate-bias reference in the prepared-input specialization. Neither
failure was a numerical assertion. Revision 2 restores the registration and
guards that reference, without changing arithmetic or tolerance, and reruns
the complete kernel/runtime suites. Failed source and results are retained.

Revision 2 passes all 163 kernel tests. The runtime suite passes 189 tests and
77 subtests, with three existing skips, but 12 workspace cases stop because
the test's initial pointer snapshot includes the two new scratch buffers
while its post-round list omits them. Revision 3 adds those buffers to the
post-round list; kernel and runtime source are unchanged. It reruns both
suites so that the full 32-round lifecycle checks execute past that assertion.

Revision 3 passes 163 kernel tests and 201 runtime tests, plus 77 subtests,
with three existing skips. The 32-round workspace cases now complete, including
graph execution, reordering, flush, endpoint publication and memory accounting.
This establishes integration regression coverage, not exact real-model state
or acceptance parity. The next diagnostic uses the frozen revision 3 APIs on
the saved real inputs before any new serving comparison.

This remains a private, uncommitted candidate on parent `58318c44`; the working
source and latest full-model AR results are unchanged. Real-input comparisons
must exercise the integrated producer and recurrence code, including graph
replay and multiple accepted windows, before another L64/C1 serving pair.

### M48: integrated first-window arithmetic is exact

All eight ranks and 69 KDA layers per rank pass the actual revision 3 API
comparison on the saved first-window inputs. Independently computed history
gate, K/U/decay, serial endpoint and the production endpoint writer match
original replay exactly. Verify output matches the recorded original output.
Three graph replays per layer, with poisoned output/history buffers, reproduce
the eager result and leave the checkpoint unchanged. The original coefficient
observer still matches the unobserved original endpoint exactly.

These 552 checks use the integrated producer and recurrence, not the private
ablation kernel. They do not run the complete model. M49 carries the next five
real accepted windows from a single initialized checkpoint, refreshing only
model inputs and checking state/conv against the original at every endpoint.

### M49: carried state remains exact across the first five windows

The five-window trace passes on all eight ranks: 2,760 layer/window checks.
History lengths are 0, 4, 7, 9 and 10, with accepted lengths 4, 3, 2, 1 and 1.
SSM state is initialized once; subsequent windows refresh raw model inputs,
carry the candidate's history and commit its own convolution state. Verify
outputs, accepted SSM endpoints and conv state match the original exactly.
Captured forward matches eager execution, and the captured endpoint writer
remains exact as positions and acceptance change. Endpoint inspection uses
separate scratch and cannot reseed the next forward's checkpoint or history.

This closes the initial real-input integration check. It still does not prove
free-running AR parity: the other model layers are not executed in this
diagnostic. M50 repeats the uninstrumented full-model L64/C1 control/candidate
pair with the same input, sampler and CUDA-graph settings as M38.

M50's first full-model attempt fails before measurement: model loading and
graph capture finish, but the HTTP gateway exits with an address-in-use error.
No protocol request or new AR result is produced. The exact conflicting
listener is not recorded. A read-only check finds the host's ephemeral range
covers the old service ports; checking availability before a long model load
does not reserve them. The retry assigns the same explicitly checked ports
outside that range to both arms, without changing model code, inference
settings or OS networking. The failed run is retained separately and both
arms will restart from scratch.

### M50: L64/C1 acceptance and generated tokens recover; latency still regresses

The fresh, uninstrumented pair completes successfully after the network-only
retry. Both arms use eight GB300 GPUs, full 93-layer real NVFP4 weights, TP8,
the same real EAGLE3 draft (width 4), CUDA graphs and overlap enabled. The
software stack remains Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13,
driver 580.167.08, FlashInfer 0.6.18 and Triton 3.8.10.post20260906.
No model, sampler or precision setting changes between the arms.

Source is parent `58318c4430b88db9159d4a2d8af38e8c3c768daa`, rebased on
`eaf66b5b`, plus the previously recorded M29R2 patch. The private candidate
adds M37R2 and frozen M47 revision 3 patch
`7f9b5f192d7de2fb2c387fefb3ec0a511270a744f656bf8ae39095b89a941517`.
There is no new implementation commit or production adoption. Source,
dependencies and all eight workers pass the run audits; neither arm has a
preemption or profiler injection.

The workload is the same SWE-smith continuation, revision
`08e109b4a59eaeebf80e4675cd125d42e7ac99a4`, instance
`pandas-dev__pandas.95280573.pr_59144`: 51,936 input tokens, 51,328 cached,
608 new, then 256 generated tokens at temperature 0 and seed 1, ignoring EOS.
One warmup and three repeats per arm each flush and reprime the cache.

| Fixed C1 result | Unbuffered control | Exact-history candidate |
| --- | ---: | ---: |
| Acceptance rate, all four requests | 0.8638 | 0.8638 |
| Average acceptance length | 3.59 | 3.59 |
| Median total latency, three repeats | 1045.84 ms | 1140.08 ms |
| Median prefill latency | 156.50 ms | 157.66 ms |
| Median decode window, total minus TTFT | 885.30 ms | 976.18 ms |

All four candidate outputs equal all four control outputs token-for-token:
all 16 cross-arm comparisons have no difference. Their common output SHA is
`23765e5cce413fa199056442cc666933ea8e983b079c5b25cc26277b3d3dcd2c`, also matching
the earlier controls. The prior M37R2 result was AR 0.7033 / length 3.11 and
diverged at token 11. Together with the same-input ablations and exact carried
state checks, this supports floating-point producer/reconstruction differences
as the cause of the recovered AR gap in this fixed case. It does not assign a
fraction of the gap to each individual arithmetic change.

Performance remains a failure: total latency increases **9.01%**, and the
decode window is about **10.27%** longer. These are three repeats in one fresh
process per arm, not the broader independent-restart C1/C4 performance gate.
No current-candidate AIME evaluation has run. AR recovery does not justify
making buffering the default or claiming the overall goal complete.

Reproduction artifacts are the frozen M47 revision 3 manifest, M48 real-input
records, M49 carried-window records, the M50 revision 2 runbook and comparison,
and its separately validated summary. They preserve the failed first attempt
and the port-only retry. Next profile this exact candidate against the same
unbuffered control, now without different token trajectories confounding
latency. Keep exact producer/state/graph checks and this AR protocol as gates
while reducing the remaining kernel cost; capacity/concurrency expansion and
current-version accuracy validation remain later steps.

### M51: matched-token profile started after acceptance recovery

A new persistent eight-GB300 cohort has passed source, real-weight, dependency,
NVLink-fabric and idle checks. The software, full-model NVFP4 TP8 settings,
EAGLE3 width, CUDA graphs, overlap, input and sampler remain those of M50.
The control is frozen M29R2 with buffering disabled; the candidate is the
same frozen M47 revision 3 exact-history implementation. No new production
commit or source adoption is implied.

Four CPU preparation checks pass, including unchanged generation/model
arguments, both actual M50 startup logs, all prior correctness/AR prerequisites
and exact decode-window accounting. The sequential Nsight pair has started:
one C1 warmup and one captured 256-token continuation per arm, with identical
flush/reprime setup outside capture. NSYS uses software CUDA/NVTX tracing and
graph-node attribution on all eight GPUs; no other GPU workload may overlap.

The analysis will check whether both captures reproduce M50's output tokens
and acceptance, then separate actual verify-round counts, graph spans and
inter-graph gaps. The target is the remaining 9.01% unprofiled latency increase,
not the earlier slowdown caused partly by different generated trajectories.
Reports and any failures are retained, with no automatic replacement requests.
There is no completed M51 profile, new performance result or AIME score yet.

M51's unbuffered control has since completed its capture and export. All eight
GPUs contain one prefill forward and 72 target decode graphs. Warmup/capture
outputs match M50's common token sequence and AR 0.8638 / length 3.59. Both
nodes are idle after cleanup; the exact-history candidate capture is now
starting on the same cohort. Its result is still pending, so there is no paired
M51 performance conclusion yet.

### M52/M53 preparation: exact flush coverage before capacity timing

Two follow-ups are prepared, not submitted. They retain frozen M47 revision 3
and do not adopt or modify production source. Each passes two CPU protocol
checks; those checks do not establish GPU correctness or model performance.

M52 extends the saved real-input check to 12 consecutive verify windows, all
eight rank fixtures and all 69 KDA layers, at capacities 8/16/32/64. Actual GPU
prepare/validate/commit kernels manage positions; a shifted logical origin
also crosses a state-page boundary. It compares graph/eager execution from
candidate-owned state, checks exact original verify output and flush state,
and poisons rejected history. Original state is loaded only at initialization.
Endpoint inspection uses a separate pool that cannot repair the next forward.
Eleven windows have an original following-state reference; the twelfth does
not, and that missing endpoint check is recorded explicitly. This diagnostic
uses independent saved rank inputs, not a full TP8 model or free-running AR.

M53 preserves the full unprofiled real-model protocol, with two independent
restarts each for the pre-plan original, exact-history L8 and exact-history L16.
Each startup covers C1/C4 with three rounds, one warmup and five measured
batches per concurrency/round. The fixed order is original/L8/L16/L16/L8/original.
It compares every candidate restart against both originals and retains all
timings, acceptance and output differences; C4 is counted by batch, not as
independent requests. Neither a shorter capacity nor a speedup is assumed.
The controller requires M51's completed, matching-token profile/package and
M52's exact GPU checks, idle nodes and enough lease for the complete sequence.
If the candidate source changes, these prepared runs must be revised before
submission. Current-source AIME and the broader original goal remain open.

### M51 completed: equal acceptance, but reconstruction still costs more

Both profiles finish with identical output tokens, AR 0.8638 and acceptance
length 3.59, matching M50. Every GPU records one prefill and 72 target decode
graphs. Source, dependencies and all eight workers pass audits; both model
processes stop and the cohort is idle. No implementation source is adopted.

At fixed L64/C1, median target graph time increases 9.78–10.09% across ranks.
Rank 0 changes from 12.422 to 13.673 ms. Its recurrent kernel median increases
from 5.568 to 18.368 us per launch; median sums across 69 layers are 385.184
and 1273.567 us per graph. The candidate's separate history gate adds a
425.760 us median kernel sum. These operations can overlap: their sums are
not an additive decomposition of wall time. Removing the old batched accepted
replay has not offset the exact per-layer reconstruction/producer cost.

The complete profiled rank-0 decode window increases 20.15%, from 948.191 to
1139.259 ms. This includes a 93.506 ms graph outlier: seven ranks spend about
79.1 ms in an allreduce while rank 4 has a 79.429 ms gap before submitting that
graph. The trace supports a late-rank wait, but without CPU sampling or context
switches it does not establish why that submission was late. All outliers
remain in the reports and accounting. The profiler result does not replace
M50's unprofiled +9.01% total-latency regression or establish AIME accuracy.

The paired-report package contains four clearly named reports, validation,
analysis and the report's warnings/limitations. M52's real-input flush checks
are the next gate before M53's prepared capacity comparison. Exactness must
survive the additional checkpoint writes; shorter capacity is not assumed to
be faster. Current-source AIME and the complete performance goal remain open.

### M52: exact state survives capacity flushes and a state-page boundary

The unchanged M47 revision 3 candidate passes all 32 saved-input cases:
capacities 8/16/32/64, eight rank fixtures and 12 continuous verify windows
across 69 KDA layers. All 26,496 layer/window verify comparisons and 24,288
accepted endpoint comparisons match the original exactly. Graph and eager
execution agree from the same candidate-owned state snapshot. Conv state,
flush checkpoints, non-flush preservation and poisoned rejected entries pass.

The fixture starts at logical position 121 to cross a 128-token state page.
Each rank performs 11/2/1/0 capacity flushes at L8/16/32/64. Original state is
loaded only at initialization; accepted endpoint inspection writes a separate
pool and cannot repair the following forward. Only eleven windows have an
original following-state reference, so no twelfth endpoint comparison is claimed.
Actual GPU metadata kernels manage positions, validated against the schedule.

Environment/source are unchanged from M51; source audits and final GPU/port
idle checks pass. This is an isolated real-input diagnostic, not a full TP8
model or free-running acceptance measurement. No source is adopted or committed.
It permits the prepared M53 original/L8/L16 independent-restart comparison;
it does not establish performance, full lifecycle correctness or AIME accuracy.

M53 has now started after these gates and fresh cohort-idle/source checks.
The fixed sequence is original/L8/L16/L16/L8/original, preserving two separate
startups per implementation/capacity and the full C1/C4 protocol. Source and
executable helpers are frozen before submission. No M53 performance or AR
result exists at startup; the unchanged candidate remains private and disabled
by default. Do not overlap GPU work or modify the running experiment.

### M54 preparation: isolate history-gate token ownership

While M53 runs, a CPU-only preparation targets the cost identified in M51.
The candidate's extra history-gate launch replaces the old payload-capture
launch, leaving rank-0 graph node count unchanged at 2,667. Its duration,
rather than simply the number of launches, deserves investigation. At C1,
the exact gate retains an all-layer tile choice but launches only one layer:
48 programs, each serially processing four token rows.

The private experiment reuses the frozen JIT body and varies only token rows
per program (1/2/4/8), retaining channel tile 32, one warp, pointer-alignment
specialization and all expressions. It leaves the rows>=16 tensor-core route
unchanged. More programs reread more weights, so no benefit is presumed;
compiler changes can still alter arithmetic, so bitwise equality is required.

The prepared protocol uses all twelve saved windows and 69 layers across
eight rank fixtures, testing four-row windows and concatenated gate-only
B2/B3/B4 shapes. Original descriptor-gate output is the oracle; eager and
poisoned graph replay must agree exactly. Timing follows all rank-local
correctness checks, traverses 69 distinct layer weights and retains nine
samples with alternating-order production brackets. It excludes other model
work and is not an E2E or acceptance test. Two CPU protocol checks and syntax
checks pass; no GPU run or source adoption has occurred. The controller refuses
submission before M53 completes and fresh idle/source/lease checks pass.

### M53 first original startup complete; candidate results still pending

The first frozen-original startup completes the full 75 measured requests
and 30 batches, plus all warmups. Source/worker ownership, native libraries,
software and launch settings pass validation; there are no preemptions or
profiler injections. Its measured C1 engine total-latency median is 1045.39 ms,
AR 0.8638 and acceptance length 3.59. All fifteen C1 outputs match the M50
control token SHA. C4 engine total-latency median is 1580.175 ms, median
AR 0.8898 and acceptance length 3.67; two output sequences occur thirty times
each. C4 has fifteen measured batches, not sixty independent trials.

Whole-batch client medians are 1060.665 ms for C1 and 1620.588 ms for C4.
The three C4 round medians are 1626.221 / 1616.287 / 1618.972 ms. These values
are retained with every sample and prefill grouping. The owned original
server is stopped only after complete validation; the controller will proceed
to the L8 candidate after both-node idle and port cleanup. There is no paired
M53 speedup or AR conclusion yet, and the second original startup is still
required. Experiment executable hashes remain frozen and verified.

### M53 L8 first startup: C1 stays exact; C4 acceptance still differs

L8 revision 1 completes all 75 measured requests without preemption. All
fifteen C1 batches retain the original tokens, AR 0.8638 and acceptance length
3.59. Engine total-latency median is 1108.08 ms versus 1045.39 ms for the first
original startup; client whole-batch median increases 5.81%.

C4 does not retain that equality. Every one of its fifteen measured batch
output multisets differs from the original. Thirty requests have AR 0.716 /
length 3.15 and thirty have 0.881 / length 3.64; the original has thirty each
at 0.881 / 3.64 and 0.8986 / 3.70. Matching acceptance statistics do not imply
matching tokens: neither candidate C4 sequence matches either original one.
The C4 median AR is 0.7985 versus 0.8898. Engine total-latency median is
1732.22 versus 1580.175 ms; client whole-batch median is 1841.186 versus
1620.588 ms (+13.61%). These are one-startup comparisons, not the completed
six-startup result or an isolated kernel speed measurement.

Both startup logs explicitly show `enable_mixed_batch=False`, so the mixed
prefill/decode path is not a supported explanation for this run. The next AR
diagnostic is fixed at L8/C4: compare same-source buffering-off control, then
the same inputs/state through verify, history producers and accepted replay.
The pre-plan original differs in upstream source as well as buffering; a
same-source control is needed before assigning the C4 regression to replay.
Saved C1 checks do not cover real B4 arithmetic or request-state association.
No implementation is changed or adopted on the strength of the C1 result.

The fixed M53 sequence continues unchanged; L16 revision 1 starts after L8
validation, owned-server shutdown and both-node idle checks. No diagnostic GPU
work overlaps it. M54 remains prepared only. A separate read-only check of
M51/current worker startup logs finds all eight workers pinned to 72 NUMA-local
CPUs; this rules out the simple one-CPU-per-worker explanation for M51's late
rank, but does not identify the host-delay cause. No CPU sampling was captured.

### M53 interrupted by prefix sharing; M55 isolates B4 arithmetic

The L16 first startup stops at its second C4 warmup: three requests reuse
51840 cached tokens instead of the protocol's 51328, leaving only 96 new
tokens each rather than 608. The server logs show a one-request 608-token
prefill followed by a three-request 288-token prefill. The cache-hit assertion
correctly rejects this changed workload; the model has not crashed. All
partial results remain, including the changed-prefix requests. They are not
replacement performance samples or a completed six-startup comparison.
The controller is terminal and the audited remaining server is stopped;
both nodes and strict ports pass idle checks. The other startups are not run.

M55 now fixes capacity 8 and batch 4 to isolate arithmetic. It constructs
three independent B4 batches from twelve saved real C1 windows per rank and
uses all 69 layers/eight rank fixtures. Original B4 verify and accepted replay
are recomputed from those same inputs/state; C1 outputs are not a B4 oracle.
It compares BF16 conv/gate producers, verify output, history gate, accepted
conv/state and the following capacity flush. Candidate-owned history feeds
the next forward; inspection state cannot repair it. Rejected entries are
poisoned, and two graph replays check the first window. The second window is
eager, not a claim of full runtime or graph-lifecycle coverage.

A diagnostic substitution of original gate, then original gate and conv,
separates producer from recurrent arithmetic. In particular, native verify
uses a BF16 GEMM result while buffered replay uses an FP32 result converted
to BF16 inside the kernel; C1 equality alone does not establish B4 equality.
This is a hypothesis, not a confirmed cause. CPU packing/report tests pass;
the fixed GPU diagnostic is submitted after idle/source gates, without source
changes. It records every numerical difference instead of stopping at the
first layer. No full-model AR or performance conclusion follows from it.
M54 gate timing is deferred while this AR investigation takes priority.

### M55 completed: state replay is exact in these B4 cases, verify is not

All 1656 layer/group cases complete with finite results on the unchanged
candidate. Conv and verify gate, including the BF16 conversion, match the
original exactly. The history gate, accepted conv/state for both [1,2,3,4]
and [4,3,2,1], and the following flush state also match exactly. The endpoint
inspection pool never feeds the next forward, and poisoned rejected entries
do not affect reconstruction. Two first-window graph replays match eager.

Nevertheless, 2448 of 40,697,856 first-window verify output elements differ
across 1273 of the 1656 cases; maximum absolute difference is 3.05e-5.
Substituting original gate, then both original gate and conv, leaves exactly
the same difference counts. Next-window verify also differs despite an exact
reconstructed starting state. This isolates a B4 verify-recurrence numerical
discrepancy in these inputs, rather than a producer or accepted-state mismatch.
It does not yet prove that this discrepancy explains the full-model AR gap.
The prior BF16-versus-FP32 gate-GEMM hypothesis is not supported by this set.

Source/helper checks and final both-node idle checks pass. The controller
records successful diagnostic execution but numerical inequality, not a
correctness pass. No model output, AR, performance or AIME result is inferred.
M56 now observes the compiled original/candidate B4 kernels without changing
launch arguments or arithmetic, requiring the same first-group results as
M55. The next candidate must be justified by this compiler evidence and then
checked in the same full-model source-off/on setting.

### M56/M57: one-warp normalization ownership is a concrete candidate

The compiler observer's first attempt saves both kernels but fails while
recording a launch argument: the original passes BV by keyword, the candidate
positionally. The failed run remains. A separately recorded revision fixes
only argument recording, passes its CPU test for both call forms, and completes
the observer's numerical-preservation and final idle checks.

Both B4 kernels use BV8, one warp and three stages. Original verify loads
normalization vectors with four contiguous elements per thread and allows
reordering for reduction; buffered verify fixes the vector ownership to one
element per thread. That is a floating-point reduction-order difference, even
with identical BF16 inputs and starting state. It supplies a specific candidate
explanation, not yet proof of the full-model AR cause.

M57 changes only that vector layout expression for single-warp verify.
Four-warp verify, T1 and accepted-history normalization are unchanged. An AST
check proves the kernel body equals the frozen candidate after reverting the
one expression. The process-local private candidate now runs the unchanged
M55 L8/B4 protocol across all ranks/layers, with source/environment records.
No production source is adopted, no additional state or launch is introduced,
and no acceptance rule or tolerance changes. C1/shared regressions and a
same-source full-model off/on comparison remain required after this gate.

M57 subsequently completes all 1656 cases. The one-expression layout change
reduces first-window differing elements from 2448 to 877, with maximum absolute
difference falling from 3.05e-5 to 7.63e-6. The two next-window checks still have
870 and 834 differing elements (maximum 7.63e-6). Producers, accepted/flush
states and first-window graph/eager comparisons remain exact and finite.
The candidate therefore does not pass the bitwise verify gate and is not
adopted. Both nodes are idle after the run. The next investigation stays at
L8/B4 and compares the remaining normalization/reduction and FMA instruction
order; reducing the error count is not proof that model AR has recovered.

### M58/M59: separate compiler contraction from state replay

M58 observes the unchanged M57 candidate and original on the same first B4
group. Results exactly match the prior run. Both compiled kernels retain
BV8, one warp and three stages, and the normalization now uses the same
local-pair/warp-butterfly sequence. The remaining recurrence differs in
floating-point contraction. In the frozen original, projection contracts
local products for value rows 0,1,2,7 within each eight-value tile, but not
3–6. Its update contracts the decay product for row 7 and the correction
product for the others. Only row 7 contracts the output dot. The candidate
does not reproduce those compiler-dependent distinctions.

M59 explicitly reproduces these three operations in a private diagnostic,
retaining the same history, producer, launch and metadata paths. These row
masks are evidence about this compiler result, not a portable production
contract. Even a bitwise pass would still require a full-model same-source
off/on comparison to explain AR, and a maintainable implementation before
adoption. No performance benefit is inferred from this diagnostic.

The first attempt fails during compilation because this Gluon version lacks
`broadcast_to`; no numerical result is produced. Its artifacts remain.
After both nodes pass idle checks, a separately registered revision uses the
inline-assembly operation's documented implicit broadcasting. The CPU AST
check confirms that only the declared verify substitutions differ from M57.
Revision 2 completes the unchanged M55 L8/B4 gate: all 1656 cases, including
40,697,856 first-window verify elements, are bitwise exact. Original-producer
substitutions, both accepted prefixes, next-window verify/flush, initial-state
preservation and graph/eager checks all have zero differing/nonfinite elements.
Source/helper checks and final both-node idle checks pass. This establishes
the arithmetic cause of the isolated verify discrepancy, not full-model AR.

The same private override now runs M60: the L8 subset of the existing carried
C1 trace across all eight rank fixtures, then the complete kernel/runtime
regressions. The C1 checks and 163 kernel tests pass. Runtime completes with
199 passed, 77 passed subtests, three existing skips and two argument-parsing
failures. Those tests inherit the two-node fixture launch and automatically
select a rendezvous port inside the host's ephemeral range; the safety check
rejects it before SSM executes. The first run remains failed. A separate
revision reproduces both failures on unmodified source, then restores M47's
one-node launch for the full runtime suite. No assertions, port checks or
host settings are changed, and no test is skipped to obtain a pass.
The unmodified control reproduces precisely those two launcher failures.
The separately recorded single-node runtime rerun then passes 201 tests and
77 subtests, with the same three skips. No source/test patch was needed, and
both nodes are idle afterward. Combined with the retained kernel/C1 results,
the diagnostic's regression prerequisite is satisfied; the first failed run
remains part of the record.

M61 registers a comparison of full-model processes
using identical M47R3 base source with buffering off, unchanged L8 on and the
arithmetic diagnostic L8 on. Fixed C4, real NVFP4 TP8 and CUDA graph remain.
One warmup plus three repeated batches per arm record all token and AR
multisets. Batched API delivery replaces competing HTTP calls only in this
new AR diagnostic; actual prefill grouping must still be reviewed. It is not
a replacement for M53 samples or a performance/AIME gate.

Model initialization succeeds, but the first C4 request receives HTTP 400:
the gRPC gateway does not support batched input IDs. No C4 sample is produced;
the failed attempt and its frozen helpers are retained. Revision 2 restores
the prior runbook's four concurrent scalar-input requests with a start barrier.
It changes neither the gateway nor the model, workload or sampling settings.
After fresh source and process-ownership checks, it reuses the loaded control
from cache flush and parent prime; none of the failed request is continued.
The new control's four batches have identical output/AR multisets: two requests
at AR 0.881 and length 3.64, two at AR 0.8986 and length 3.70. Its exact owned
server is stopped and both nodes pass idle checks before unchanged L8 starts.
The retained control has one extra parent-only request; this recovery is an AR
diagnostic, not a fresh-process performance comparison. No production code is
adopted, and no commit or push is made in this stage.

### M61 completed: L8/C4 AR recovery isolates verify arithmetic

Revision 2 completes all three arms, including source/environment and eight-
worker audits, unchanged helper hashes, and final both-node idle checks.
Each arm runs one warmup and three repetitions, four concurrent requests per
batch, 256 output tokens per request. The same-source unbuffered control,
unchanged buffered L8 and arithmetic-aligned buffered L8 use the same real
93-layer NVFP4 model, TP8, EAGLE3 T4, CUDA graphs and overlap.

All twelve batches have identical prefill grouping (1+1+2), 51936 input,
51328 cached and 608 new tokens per request. Every batch in an arm has the
same joint output-token/AR/acceptance-length multiset. Each table entry below
describes two requests per batch, not a pooled or weighted acceptance rate.

| Same-source arm | AR values | Acceptance lengths | Output tokens versus control |
| --- | --- | --- | --- |
| Buffering off | 0.8810 / 0.8986 | 3.64 / 3.70 | Reference |
| Unchanged L8 | 0.7160 / 0.8810 | 3.15 / 3.64 | Both output sequences differ |
| Arithmetic-aligned L8 diagnostic | 0.8810 / 0.8986 | 3.64 / 3.70 | Exact multiset match, all four batches |

All sixteen control/corrected cross-batch comparisons match tokens and
acceptance jointly. None of the sixteen control/unchanged comparisons match;
all have the same prefill grouping. The override is confirmed on every GPU
worker at the actual B4/one-warp launch geometry. No sampler, acceptance rule,
history representation, flush policy, persistent state or launch geometry is
changed between the two buffered arms.

Together with M55/M59, this isolates the observed gap to verify's normalization
reduction and projection/update/output contraction, not an accepted-state
mismatch in the checked inputs. Mathematical equivalence alone did not preserve
floating-point execution. Matching prefill groups is not a record of every
decode scheduling decision; the conclusion remains scoped to this workload,
one process per arm and the recorded environment.

Next, replace the compiler-specific diagnostic with maintainable arithmetic
that preserves the checked contract, then repeat same-input, C1/C4 and shared
regressions before measuring performance. The root implementation is unchanged;
this is an AR-cause milestone, not an adopted fix, a no-regression performance
pass or current-version AIME validation.

### M62–M64: three bounded alternatives do not fix verify arithmetic

These experiments retain the M47R3 source snapshot: parent commit
`58318c4430b88db9159d4a2d8af38e8c3c768daa`, uncommitted patch SHA256
`7f9b5f192d7de2fb2c387fefb3ec0a511270a744f656bf8ae39095b89a941517`.
They use the same GB300 environment, Python 3.12.3, PyTorch 2.13.0+cu130 and
Triton 3.8.10 as the preceding diagnostics. Each completes the unchanged M55
L8/B4/T4 protocol: eight saved rank fixtures, three groups per rank and 69
layers, totaling 1656 cases. Original B4 verify and accepted replay are
recomputed from identical inputs; these are fixture tests, not new TP8 model
inference, AR measurements or performance samples.

| Candidate | Change from M57 | First verify differing elements | Accepted/flush state |
| --- | --- | ---: | --- |
| M62 | Let Triton infer layouts instead of explicit Gluon layouts | 4066 | Also differs; rejected |
| M63 | Use configured T for register-only verify; mask stores to live width | 877 | Exact |
| M64 | Load BF16 verification buffers; keep independent FP32 history inputs | 877 | Exact |

M62 mechanically preserves the arithmetic and control flow while removing
explicit layouts. It changes history ownership too, so the accepted-state
regression rules it out. M63 and M64 have exactly the same per-layer metric
dictionaries as M57 across all eight ranks, including the remaining 870/834
next-verify differences. Their maximum verify error is 7.63e-6. Producers,
history gates, accepted/flush state, conv state, initial-state preservation
and first-window graph/eager comparisons remain exact. All three runs are
finite, complete source/helper checks and finish with both nodes idle.

Each candidate has a CPU scope test. M62 checks the complete mechanical
transformation; M63 checks the two declared kernel changes and live-store
coverage; M64 checks its unchanged JIT and four-pointer/stride adapter. M64
allocates BF16 copies below the frozen wrapper's dtype check solely to isolate
storage precision. This is neither a supported API nor a performance design.
No candidate reaches real-model AR or timing, and the working implementation
remains unchanged. Frozen helpers, source hashes and failed numerical reports
are retained in each private runbook; no failed sample is replaced.

Read-only inspection narrows the next question. Gluon delegates sum/reduce to
the same Triton definitions, and both kernels enable floating-point fusion.
The observed allow_reorder attribute is on a normalization reshape, not a
different reduction default. The saved LLVM programs pack verify state into
different vector groups, consistent with the subsequent value-row-dependent
contraction. This identifies another compiler difference, not yet its causal
pass. A noinline helper taking register-state tensors is unsupported by this
compiler. Neither blanket frontend replacement nor a loop-bound or input-
dtype change provides a maintainable exact replacement for M59's diagnostic.
The original/control math and the numerical/AR gates remain unchanged.

### M65: offline compilation isolates the role of SLP vectorization

Using the saved M58 TTGIR and the same compiler, SM103 target and launch
options, M65 recompiles both kernels without loading a model or launching a
GPU. The first attempt stops on a candidate LLVM-file hash mismatch. Inspection
finds only debug metadata renumbering: its PTX is byte-identical to M58, as
are the original kernel's LLVM and PTX. A separately recorded revision requires
exact LLVM instruction-body equality without debug attachments and exact PTX
bytes. Both controls pass; the first failure remains recorded.

The revision then separately disables three LLVM optimization options, for
eight compilations including controls. Disabling extracted-add vectorization
does not change either PTX. Disabling packed-fop scalarization leaves the
original PTX unchanged and changes candidate register unpacking without
changing arithmetic opcode counts. Disabling SLP changes both kernels: the
original no longer has value-row-dependent projection/output FMA contraction,
and its state correction consistently uses scalar FMA. Thus the mixed
contraction observed in M58 depends on SLP in this compilation.

This is not a numerical or performance pass. In particular, turning SLP off
also changes the original arithmetic; applying it to both paths cannot establish
equality to the frozen reference. No installed compiler, runtime code, state
protocol or launch setting is changed. The offline script's recorded per-line
counters cover scalar FP32 instructions only; a separate count includes packed
FP32x2 instructions, and neither count is a timing measurement. CPU checks
cover option isolation/restoration, control reproduction and rejection of a
changed arithmetic instruction.

The next bounded hypothesis is an identity register boundary between history
reconstruction and verify. It would add no state buffer or launch, but must
first show the desired emitted arithmetic, then pass the existing numerical
and model-AR checks. No such candidate or GPU result is claimed here.

### M66: register boundary does not recover the frozen verify arithmetic

M66 tests that hypothesis by adding a pure four-register bit-copy operation
before verify's zero-initialized accumulator in the frozen M58 TTGIR. Flush
and history still consume the original reconstructed state. The CPU scope
test proves that removing the boundary exactly restores the entire input IR;
an initial materialization error is caught and corrected before registration.
No floating-point operation, memory access, control flow or launch is added.

Offline compilation first reproduces both unchanged controls' PTX exactly
and their LLVM instruction bodies exactly. The boundary then compiles with
the same shared-memory and scratch requirements. It does not recover the
desired arithmetic: the output-dot reduction still has eight packed adds and
no packed FMA, versus seven packed adds and one packed FMA in the original.
Inspection confirms that its separate product/add sequence remains. These
static counts include FP32x2; they are not numerical or timing measurements.
The candidate is rejected before GPU testing, and no runtime source is adopted.

This closes the bounded register-boundary experiment. The next proposal needs
a user decision: define one explicit verify arithmetic contract shared by
unbuffered and buffered execution, instead of reproducing the frozen compiler's
value-row-specific choices. If approved, keep three distinct comparison arms:
the untouched frozen original, the shared-contract unbuffered path, and the
shared-contract buffered path. The latter pair isolates replay; the frozen
original remains the quality, AR and performance reference. Changing the
contract could change generated tokens and must not be presented as recovery
of the frozen original's bitwise output. The proposal is not implemented and
the existing gates are not silently relaxed. Performance and current AIME
validation remain open.

### M67: isolate the arithmetic-aligned candidate's KDA cost

The next user-directed experiment measures the existing explicit-arithmetic
candidate, without changing the original numerical contract or running an E2E
model. Compare unchanged original, unaligned buffered, and aligned buffered
using the same saved real-model inputs, fixed acceptance, and CUDA graphs.

The fixed scope is L8/B4/T4, 69 KDA layers, 12 heads per TP8 rank and head
dimension 128. A two-window cycle includes producers, verify/history and
accepted commit, with a capacity flush in the second window. Separate graphs
measure recurrence with and without accepted history. Immutable seed pages
make repeated graphs self-contained, without timed state resets. Producers
are serialized and fixture pools are dense, so these measurements will not
represent runtime stream overlap, full-model cache pressure or E2E latency.

Before timing, repeat the existing eight-rank exactness gate. Each benchmark
arm also checks eager/graph consistency and repeated-cycle state correctness.
The aligned arm remains bitwise equal to the unchanged original: all 1,656
layer/cases in the existing eight-rank gate pass, as do the benchmark's output,
accepted-state, conv-state, graph/eager and repeated-cycle checks. Two CPU
protocol checks pass. The frozen base is commit
`58318c4430b88db9159d4a2d8af38e8c3c768daa` plus the recorded M47R3 patch;
the aligned JIT is unchanged M59R2. The environment is GB300, Torch 2.13.0,
CUDA 13.0 and tokenspeed-triton 3.8.10.post20260906.

On two separate GPUs, nine repeated candidate samples per scope show aligned
recurrence taking 10.85–10.88% less time without history and 9.83–9.99% less
with history/flush than unaligned buffered. KDA-core time, including producers
and accepted commit, is 1.438–1.448 ms per 69-layer window for aligned versus
1.528–1.537 ms unaligned and 1.052–1.056 ms original. These are the two GPUs'
medians, averaging one no-history and one flush window, not a steady-state
L8 distribution. Original brackets remain stable within 0.04% median drift.

Alignment therefore reduces this local buffered cost by 5.82–5.92%, but the
aligned path is still 36.67–37.10% slower than original in this workload. This
does not pass the plan's no-regression goal, nor establish E2E latency, AR or
AIME accuracy. All raw samples and reproduction details are retained in the
private M67 report. No production kernel change or candidate adoption is
included; no benchmark or model service remains running.

### M68: direct kernel timelines for original and aligned KDA

The user requests a unit-level NSYS comparison, without an E2E run. M68 reuses
the exact M67 workload and frozen M47R3/M59R2 sources, rather than the newer
branch checkpoint. L8/B4/T4, 69 layers, TP8 per-rank H12/D128 geometry and CUDA
graphs remain fixed. Each of two GB300 GPUs runs original and aligned pytest
cases separately. Loading, compilation, warmup, reference arithmetic and
inspection copies stay outside the capture range.

All four GPU pytest cases and two CPU scope/accounting checks pass. Aligned
outputs and accepted recurrent/conv states remain bitwise equal to original;
graph outputs match eager before and after profiling. Each report contains
three graph replays for each of no-history recurrence, history/flush recurrence
and the complete two-window KDA cycle. All nine intended NVTX ranges and their
graph-launch correlations pass coverage checks, with 1,674 original or 2,118
aligned GPU kernels per report. Generic CUDA/NVTX collection warnings are
retained and disclosed. An additional audit verifies exactly nine launches,
nine synchronizations and every intended kernel family's call count; no
intended range or kernel is missing from the accounting.

Individual recurrence-kernel medians in the isolated graphs are 10.016 µs
original versus 11.968–12.000 µs aligned without history, and 9.888–9.920 µs
original versus 13.184 µs aligned with history/flush. These are direct kernel
durations, not graph time divided by layer count. In producer-interleaved
cycles, recurrence medians are lower for both implementations; cross-scope
numbers must not be mixed. The original recurrence-only scope excludes its
later accepted-state replay.

The complete cycle identifies two main costs: longer buffered recurrence and
69 per-layer history-gate launches. The latter take about 238–242 µs per
69-layer window, versus 12.2–12.4 µs for original's batched replay-gate
preparation. Removing original's approximately 113 µs accepted-state replay
does not offset those costs. Kernel count rises from 210 to 284 per window.
The recurrence uses 128 rather than 80 registers per thread, with the same
768 one-warp CTAs; this is a diagnostic clue, not proof of an occupancy or
spill bottleneck. Complete two-window graph idle gaps are smaller for aligned
(about 21–22 µs) than original (28–29 µs), so larger bubbles do not explain
this local slowdown.

Four clearly named reports, raw kernel analysis, correctness records and
checksums are packaged in the private M68 delivery. Nsight Systems 2025.6.3
uses software CUDA tracing with graph-node detail. Profiler timings do not
replace M67's unprofiled performance numbers. No E2E/AR/AIME result, production
kernel change or candidate adoption is claimed; the no-regression goal remains
open. Final both-node idle checks pass.

### M69: shared verify arithmetic, implementation and local validation

The user approved a shared verify arithmetic contract for unbuffered and
buffered execution, retaining the untouched original as a separate reference.
This permits different rounding from the original compiler output; it does not
permit replacing the old quality/performance baseline or claiming unchanged AR.
Work starts from `2829f469` on `kda-buffered-replay`; this change records the
implementation, tests and validation summary together.

The working implementation integrates the previously isolated M47R3 replay-order
producers and ordered history reconstruction. Blackwell 128-key multi-token
target verify and buffered verify now call the same source helpers: adjacent-key
balanced reductions, explicit product/add rounding, normalized/scaled Q and K,
rounded state decay, then correction FMA. The helpers adapt register ownership
for Triton/Gluon without value-row masks or global compiler-flag changes.
The acceptance rules, frozen accepted-replay kernel, T=1 decode arithmetic,
LCM ownership and state commit/flush protocol are unchanged. Extra producer
scratch is preallocated, shared across layers and included in the recipe budget.
At B4/T4/H12/D128, the added scratch is 288 KiB per rank, not per layer. The new
arithmetic has no user-selectable switch; other architectures/head dimensions
and the non-shared verify branch retain their existing operation order.

Local validation uses a new persistent allocation and the existing cached
GB300/CUDA 13/PyTorch 2.13/Triton 3.8.10.post20260906/FlashInfer 0.6.18
environment. Tests load working-tree code, not the older private candidate.
Each attempt records the base commit, working patch, source hashes, added source
files, environment, exact command, logs and exit status; source changes during a
run invalidate it. The first two GPU smoke attempts failed at Gluon layout
compilation. They are retained; the third passed all six initial numerical cases.

Final-source results:

- **187 kernel tests pass.** Coverage includes conv, recurrence, metadata,
  endpoint, capacity/padding/invalid backing and eager/CUDA graph execution.
  An independent FP32 reference checks state, correction and output exactly
  across four tile/warp settings, Triton/Gluon and compiler FP fusion on/off.
  Native verify covers B1/B4/B16, BF16/FP32/separate history producers and both
  bounded and softplus gates. Existing tolerances were not widened.
- **240 runtime tests and 117 subtests pass**, with three existing skips.
  These include workspace budget/stable storage, cache lifecycle, accepted
  commit, mixed execution, graph metadata and shared GDN regression.
- **All 1656 saved real-input layer/cases pass**, across eight TP8 rank
  snapshots at L8/B4/T4, H12/D128. Each rank checks three groups of 69 layers.
  Producers, verify output, two graph replays versus eager, accepted conv/state
  and next-window verify/flush match exactly. Accepted lengths are [1,2,3,4]
  and [4,3,2,1]; rejected history is poisoned, and the implementation carries
  state between windows without oracle substitution. B4 is assembled from
  independent C1 snapshots, not a new C4 model rollout.

The original reference remains a separately imported, frozen module. Shared
verify differs from it in **15167 / 122093568 BF16 output elements** (about
0.0124%), with maximum absolute difference **0.00048828125**. Producers match.
This is a same-input comparison over first/subsequent verify windows, not a
generation or AR result. Equality of the new pair does not recover the original
token trajectory by definition.

Failed attempts remain part of the record. The initial runtime run had 21
missing-helper import failures; correcting the harness import path fixed them.
The first real-input gate matched verify but found accepted-state differences
up to 1.9073486328125e-6. A focused diagnostic isolated history correction U:
K and decay were exact. Moving beta before projection had changed the legacy
compiler contraction. Restoring history's original source order restored exact
K/U/decay and all eight ranks. Review also restored the non-shared unbuffered
branch's beta position, followed by a full final-source rerun. No compiler-row
mask or tolerance relaxation was adopted.

The three final suites share one Python source manifest, SHA256
`6886860d22368761e293c1d8dcc7c6d0563cc6b2a0a43c114337b4c9b84c19b7`,
and each confirms unchanged source during execution. Their patch, added files,
commands, environment, fixture hashes and failures are retained in the local
M69 report/runbook. Repository-wide pre-commit hooks and added-file hooks pass.

No full model, AR, AIME, new NSYS or performance measurement is claimed here.
Earlier M61/M67/M68 numbers do not apply to the new arithmetic contract. Next
measure the three arms in the same KDA benchmark, then validate fixed L8/C4
real-model token/acceptance behavior and current-source AIME.

### M70: recover shared buffered KDA performance

This stage starts from `5fc562dd51a495b5dfdf5f9636fd99bf0632dfb3` on
`kda-buffered-replay`. It is a working-tree optimization, not a new committed
revision. The local M70 runbook retains each command, source patch, added file,
fixture hash, environment and result; failed attempts are kept separately.
All final runs use one Python manifest, SHA256
`7accc1efab0a825b19f4e52406acd18149ab7b187e5258edff7756931ff7213b`, and
confirm unchanged source during execution.

The changes target work introduced by exact buffered replay:

- Gluon verify reductions keep the same adjacent-key balanced tree. Four
  adjacent keys per lane allow two local pair levels followed by five explicit
  rounded warp-shuffle additions, avoiding repeated layout conversions.
- Independent verify/history recurrences are interleaved per token. Static
  window unrolling exposes instruction overlap; live width still guards reads
  and stores. Each chain retains its arithmetic and only history writes K/U/D.
  Native B4 uses the measured BV8/four-warp geometry.
- Independent conv/raw-capture and gate CTAs share a launch, with heavier gate
  CTAs first. Verification receives the BF16-rounded raw dot. Replay keeps its
  own bias-initialized MMA accumulator; adding bias to the raw result is not
  equivalent. Below 16 packed rows, history retains its separate scalar kernel
  because fusing that reduction changes FP32 rounding.
- The workspace calls the fused producer directly and no longer needs its
  producer stream or per-layer events. Scratch allocation and LCM accounting
  do not grow. T1 and T4 retain one workspace entry.

No scheduler, ownership, acceptance, sampling, commit/flush policy, weights or
GDN implementation changes are included. The frozen old verify is still an
independent reference, not replaced by the new shared contract.

Measurements use a persistent two-node GB300 allocation, the cached CUDA 13 /
Torch 2.13 / Triton 3.8.10.post20260906 / FlashInfer 0.6.18 environment and driver
580.167.08. Each run uses one GPU. Inputs are saved real-NVFP4 TP8 rank 0/4
activations, L8/B4/T4, 69 layers, H12/D128. A cycle has one empty-history window
and one reconstructed-history/flush window, accepting [1,2,3,4] then [4,3,2,1].
It includes producers and accepted commit, with no timed reset or profiler.
CUDA graph is enabled: five warmups, nine alternating candidate samples,
16 replays per sample and 18 original bracket samples. Old producer work is
serialized; the benchmark does not model its runtime StreamFork overlap.

Before optimization, the committed buffered implementation takes 3650.816 us
on the first node and 3664.768 us on the second, versus frozen original
2107.892 / 2095.169 us. Final repeated timings are:

| Saved rank / node | Frozen original (us) | Optimized buffered (us) | Reduction from committed buffered |
| --- | ---: | ---: | ---: |
| 0 / first | 2107.904–2108.096 | 2101.758–2101.888 | 42.43% |
| 4 / second | 2094.786–2095.168 | 2094.464–2094.720 | 42.84–42.85% |

Each range covers two independent benchmark runs. The optimized medians are
0.003–0.294% below their original brackets: effectively parity, not a material
speedup over original. The shared-unbuffered control remains 2443–2447 us.
These are two-window graph times, not individual-kernel or E2E latency.
Recurrence alone remains slower: 769–770 us without history and 864–866 us
with history/flush per 69-layer window, versus about 702 us for original
verify. Producer fusion and saved accepted replay also contribute to parity.

Correctness on final source:

- **197 kernel tests pass**, including the existing independent explicit FP32
  arithmetic reference and ten parameterized fused-producer cases. The latter
  cover B1/B4/B16, T1/T4, both gate forms, strided input, partial widths,
  padding/invalid state and repeated CUDA graph execution in one test function.
- **240 runtime tests and 117 subtests pass**, with three existing skips.
  Workspace sizing/pointers, cache lifecycle, graph metadata, accepted commit,
  mixed execution and shared GDN regressions remain covered.
- **All 1656 saved real-input layer/cases pass** across eight ranks. Producers,
  new-pair verify output, graph/eager, accepted conv/state, next verify and flush
  are exact. Rejected history is poisoned and state is carried between windows.
  B4 rows are assembled from C1 snapshots, not a fresh model rollout. Relative
  to frozen verify, the existing M69 difference remains 15167 / 122093568 BF16
  elements, maximum absolute difference 0.00048828125; it is not hidden or
  relabeled as exact original-output recovery.

Rejected experiments include raw-dot-plus-bias reuse and small-row scalar
fusion (accepted-history differences), tile/warp choices that fail equality,
slower normalization layouts and producer tiles, and ineffective stage changes.
The first static-loop and interleaving attempts fail compilation and remain in
the record. No tolerance, acceptance rule or compiler-specific value-row mask
is used to obtain the final result.

This closes the measured local KDA latency gap, not the full-model gate.
Current-source AR/token trajectory, AIME, broader capacity/concurrency timing,
and E2E performance with runtime stream overlap remain separate next steps.
Repository-wide pre-commit, explicit new-file hooks and diff checks pass.
Both reserved nodes pass the final GPU-process and service-port idle check.
This stage leaves the validated changes uncommitted for review.
