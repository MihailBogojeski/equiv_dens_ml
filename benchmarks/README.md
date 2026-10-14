# DenSNet computation and performance contracts

These instruments compare identical saved **initialized** tensors and generated
inputs against the paper source. No trained checkpoint or labeled dataset is
required. Predictive accuracy and replacement-backbone quality are separate work.

The clean neighboring reference is `/home/johannes/equiv_dens_ml-paper-reference`
at `b15a0ff7c1e6070265c36d80a75706749778e9da`. Do not edit it. Each scientific
source arm imports in a fresh process; full-model provenance checks every import
path and saves source/input/state hashes. Local component adapters explicitly
state which neighboring code is shared to isolate one worker.

For a fresh checkout, create that source reference before running these tests:

```bash
GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach ../equiv_dens_ml-paper-reference \
  b15a0ff7c1e6070265c36d80a75706749778e9da
```

The reference source and generated test inputs suffice for numerical tests.
Fixtures, reference arrays and timing records are generated locally. Historical
campaign scripts can require previously generated fixture files.

| Worker | Regression tests | Component/workflow instrument | Applicable downstream work |
| --- | --- | --- | --- |
| A preparation | `test_geometry.py` | `geometry.py`, `calculator.py` | Actual System/calculator E/F; coefficient preparation; repeated frames and batches |
| D packing | `test_packing.py` | `packing.py`, `paired_pipeline.py` | Coefficients, grid density, E/F, ordinary/mixed training gradients; state compatibility |
| B kernel experiment | `test_kernels.py` | `kernels.py`, `experimental_kernels.py` | Primitive forward/backward/HVP/mixed; rejected experiment retained outside production |
| C grid evaluator | `test_density_evaluation.py` | `density_evaluation.py`, `paired_pipeline.py` | Supplied-grid rho/spatial derivative; density training/inference; exact derivative and memory contracts |
| E energy accumulator | `test_energy_orchestration.py` | `energy_orchestration.py`, `paired_pipeline.py` | Density-mediated E/F and frozen-density energy-stage force-loss training |
| Integration/instrument | `test_pipeline.py`, `test_paired_pipeline.py` | `fixtures.py`, `pipeline.py`, `precision.py`, `timing.py` | Requested branches, FD checks, matched optimizer updates, source/precision isolation and failures |

On an established compute allocation, use the qualified runtime wrapper:

```bash
DENSNET_NUM_THREADS=1 tools/runtime/run.sh torch python -m pytest tests/performance_contracts -q -ra
tools/runtime/run.sh torch python -m benchmarks.pipeline \
  --source-root /home/johannes/equiv_dens_ml-paper-reference \
  generate --dtype float32 --output /absolute/output/fixture.pt
```

Controlled timing requires a preregistered contrast, the shared measurement lock,
an idle allocated device, five alternating pairs, warmup and synchronized timing.
Use the benchmark scripts in `tools/performance/`; they stage the image,
record hardware/source/runtime/terminal job state and preserve raw artifacts.
Ordinary tests contain numerical gates, never fragile speed assertions.

`paired_pipeline` supports coefficient/density inference and Adam training, E/F
inference and energy-stage force training. Each round resets model/optimizer
outside timing, validates requested outputs/derivatives before timing, replays
matched warmup/updates, and compares final states. Inactive CUDA allocation pools
are released before the next arm starts. Failures are saved, including partial
timing rows. Reset/model construction is distinct from process-cold startup.
`calculator` includes actual SchNetPack units, packing, model E/F and System update.

P1 uses explicit highest FP32 matrix precision/TF32-off. C2 additionally tests
and measures matching TF32-on policies without adding reduced precision to the
new GTO contractions. Effective precision is recorded; unrelated test imports
cannot silently change the P1 gate. Source defaults/old-object fallbacks and
reference capability exclusions are documented in each worker report.

The matched native Torch/JAX instruments cover current-model E/F inference,
density/spatial-gradient SGD, frozen-density E/F SGD and a supplied-grid
changed-geometry calculator boundary. They save timings, numerical gates,
source/runtime/hardware provenance and cold-start or failure records in the
selected output directory. Speedups apply to the measured workload and
precision policy.

The strict compiled Torch comparison uses `application_fixtures.py` and
`torch_compile_speed.py`, orchestrated by
`tools/performance/three_backend_speed.py`. It uses real ethanol conformations,
the local full H/C/O auxiliary basis and identical initialized arrays across
eager Torch, fullgraph Inductor and native JAX. Requested application boundaries
include changed-coordinate E/F, host calculator transfers, density/spatial
gradient inference, and each staged SGD update. These modes do not include
QC preparation or the paper's AMSGrad/EMA.
Numerical acceptance is independent per case/backend/round/mode. The default
`isolated` cache policy measures independent cold compilation; explicitly
selected `shared-after-first` and seeded node-local continuations record cache
reuse and distinguish first-call startup from warm performance. Each run records
successful contrasts, bounded failures, startup costs and memory scope.
