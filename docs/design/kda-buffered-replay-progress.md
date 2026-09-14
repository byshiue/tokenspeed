# KDA buffered replay: implementation record

This record tracks implementation of [the buffered replay plan](kda-buffered-replay-plan.md).
It separates reference results from kernel tests and real-model measurements.
Passing a reference test does not mean buffered replay is available in serving.

## Source and milestones

Baseline: `2e4b540743878959eb51793ce01d74b5898b0e38` (upstream main).
Implementation branch: `kda-buffered-replay`. Implementation changes are
uncommitted until a milestone explicitly records its commit below. No default
capacity or performance benefit has been established.

| Milestone | Status | Evidence / exit condition |
| --- | --- | --- |
| A. Recurrence, history representation, conv and acceptance reference | CPU gates passed; GPU numerical prototype in progress | 19 CPU cases passed; GPU and serving equivalence are separate gates |
| B. LCM ownership, retention and lifecycle contract | In progress: audit | Protect the lagging checkpoint and history; accurate capacity accounting; safe publication and handoff |
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

Source: uncommitted milestone on the baseline above. Python 3.12.13,
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

Source: uncommitted milestone on the baseline above. One NVIDIA GB300,
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

Initial measurements (microseconds; serial prototype, before formatting):

| Batch / T | Existing recurrence | L=16 buffered | L=64 buffered |
| --- | ---: | ---: | ---: |
| 1 / 1 | 2.24 | 5.86 | 12.16 |
| 1 / 4 | 4.42 | 8.63 | 14.63 |
| 8 / 1 | 3.59 | 7.08 | 14.05 |
| 8 / 4 | 6.25 | 10.66 | 17.80 |

This prototype does not establish a useful default capacity. Serial history
reconstruction and commit overhead outweigh the avoided state stores in this
microbenchmark. A batched/fused commit and faster reconstruction must be
measured before serving activation; no full-model speedup is claimed.

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

Required before runtime dispatch:

1. Represent history as LCM-owned token rows with a checkpoint dependency.
   A materialized prefix hit initializes an empty history; history must neither
   demand nonexistent prefill entries nor be published as ordinary reusable KV.
2. Own request-position metadata through the cache contract, with stable GPU
   views and reset/rebind rules. Runtime batch position is not request identity.
3. Separate accepted progress from materialized checkpoint progress. Resolve
   current physical blocks from the live tables, including canonicalization;
   do not retain stale physical addresses after prefix deduplication.
4. Order endpoint materialization before prefix publication/transfer and before
   retraction sources can be reused. Published checkpoints remain immutable.
5. Generalize target commit to ordinary decode while preserving GDN/PLE/QSA;
   integrate conv commit and prepared gate inputs into that same operation.

**Serving remains unchanged.** Capacity CLI, LCM integration, lifecycle tests,
real-weight TP8 performance, AIME 2026 and full-model NSYS are still pending.
