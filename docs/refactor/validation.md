# Consolidation validation and adoption status

Status: implementation is ready for review on `refactor/repository-layout`.
Production remains on `main`; nothing is pushed or adopted into the JACS campaign.
Full production-model reproduction remains an adoption prerequisite.

## Isolation and baseline

Source baseline: `2bd1cfb8` (the local consolidation commit).
The development and detached reference worktrees have independent copies of the
production dependency environment, installed against their own source directories.
Package imports were checked to resolve inside the corresponding worktree.
No writable data/output/checkpoint/shard directory is shared with production.

Validation used CPU only, one OpenMP thread, disabled W&B, temporary fixtures and
scheduler stubs. No real jobs were submitted, cancelled or restarted. Manuscript
submodule paths and commit pins are unchanged. Protected production code hashes
were checked against `baseline.json` and matched.

## Results

| Check | Outcome |
| --- | --- |
| Reference repository suite | 503 passed, 14 failed, 19 skipped |
| Existing refactor tests after focused corrections | Same outcomes as reference; no new failures |
| Additional compatibility checks | 18 passed |
| Combined verified outcomes | 521 passed, 14 known failures, 19 skips |
| All relocated Slurm scripts | 30 files byte-identical to baseline |
| Seeded water geometry generator | 16 files match; only temporary output prefixes normalized |
| Campaign Python bodies | 136 unchanged after import/file-location normalization |
| Historical scripts and notebooks | 98 canonical files byte-identical to baseline; old executable modes retained |
| Python and shell syntax | Checked for migrated implementations |
| Wheel outside source checkout | Package loads from wheel; CLI and six command help routes pass |
| Wheel numerical resources | Clebsch–Gordan coefficient asset included |
| Experiment command without checkout | Clear error and exit code 2, as designed |
| Production code checksum audit | No changes |
| Completed paper ethanol model | Both versions stop at the same missing free-atom coefficient error; predictions unverified |

The combined result uses the original broad run plus focused reruns overriding
the earlier result for the same test. It does not claim an entirely green suite.
Machine-readable cases and counts are in `validation_results.json`.

Five new failures in the first broad refactor run were source-inspection tests
reading forwarding wrappers instead of canonical implementations. Their original
assertions now inspect the canonical source via a test-only fixture, and all five
pass. An initial AST comparison also compared indentation inside documentation
strings; the corrected check excludes free-standing documentation strings and
continues comparing the actual workflow statements and their order.

The 14 remaining failures match the isolated reference:

- One pre-existing IR-overlay test expects a PNG that the existing figure helper
  does not create. Plotting behavior was not changed.
- Thirteen checks require ignored dataset files not copied into these worktrees:
  atomic-energy arrays, SAD/prior data, and the external Rackers geometry file.

The 19 skips include CUDA/checkpoint-dependent inference checks and unavailable
local data. No large data migration or access to actively written checkpoints was
used to make those checks pass.

## What the numerical comparisons establish

The four training workflow bodies retain their statement order, apart from
explicit argv forwarding, a successful return, and placement inside `main`.
All four synthetic density/energy workload cases exercise the real shared Trainer,
including predictions, loss, optimizer/scheduler state, EMA, validation and the
next step after loading identical baseline checkpoint bytes in the new code.
A five-step CPU ASE trajectory through the DenSNet calculator also matches exactly.

A separate real-model probe copied the completed published ethanol checkpoint
`2024-03-22_96w7KyGG`, its args, full small training array and three basis/prior
assets into each isolated worktree. All six inputs were verified by SHA-256 before
and after copying; there are no links to production. Both versions load the model
and reconstruct the same force scaling, then fail before producing predictions
with `ValueError: No free atom MO or DF coefficients found!` in the dipole head.
The copied spline prior does not provide those coefficients. This is a shared
baseline limitation, recorded separately from the pytest counts. No model head,
prior or scientific default was altered to bypass it. Input hashes and the outcome
are recorded in `validation_results.json`; the reproducible probe is
`tests/fixtures/refactor/paper_model_parity.py`.

The passing synthetic fixtures validate equivalence for small controlled workloads. They
are not complete production DenSNet training runs, do not test a full multiphase
campaign transition, and do not replace comparisons against completed JACS model
checkpoints and full datasets. Those comparisons remain unverified and block
production adoption.

## Compatibility and packaging details

Existing run.py, direct Python/shell, config and sbatch paths remain usable.
The root and themed watchdog retain their different Rackers classification/rank
and SKIP_TRAINING behavior through explicit internal profiles. Source-location
lookup works for watchdog wrappers copied into logs/watchdog. Slurm directives,
claim logic, output locations, completion markers and successor chaining are kept.

Python helper adapters preserve canonical module identity and monkeypatches,
including callers using explicit importlib loaders that register their module.
The experiment registry is explicit and lists 180 executable tasks. It is local
to a checkout; the scientific library wheel does not require campaign code.

The dipole entry point previously imported GPUProfiler/profile_context from a
missing profile_utils file. Neither name was used. Removing that unused import
makes the existing command portable without changing numerical operations.
RNG initialization now occurs when the dipole command runs, not when it is imported.

## Reproducing checks

Run from the development worktree with its environment; never run these commands
against the active production checkout. `EQUIV_DENS_REFERENCE` must point to an
isolated baseline checkout with its own environment and temporary outputs.

```bash
EQUIV_DENS_REFERENCE=/scratch/mh7373/projects/equiv_dens_ml_baseline \
PATH="$PWD/scratch/refactor_validation/fakebin:$PATH" \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 WANDB_MODE=disabled \
  .venv/bin/python -m pytest tests/compatibility -q

PATH="$PWD/scratch/refactor_validation/fakebin:$PATH" \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 WANDB_MODE=disabled \
  .venv/bin/python -m pytest tests -q

.venv/bin/python -m pip wheel --no-deps --no-build-isolation . -w scratch/wheels
```

For integration tests, keep scheduler stubs first on PATH. The validation run used
`scratch/refactor_validation/fakebin/` with sbatch/scancel/srun blocked and scheduler
queries stubbed. Raw JUnit reports, logs and wheel evidence remain local under
`scratch/refactor_validation/` in the two isolated worktrees. Checkpoint/MD fixture
outputs are temporary; fixture source is versioned under `tests/fixtures/refactor/`.

The completed-model probe is deliberately separate from automated pytest discovery.
After copying the six recorded inputs under `scratch/refactor_validation/paper_fixture/`
in each workspace, run its interpreter with that workspace's `src` on PYTHONPATH:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 WANDB_MODE=disabled \
PYTHONPATH="$PWD/src" .venv/bin/python \
  /scratch/mh7373/projects/equiv_dens_ml_refactor/tests/fixtures/refactor/paper_model_parity.py \
  --root scratch/refactor_validation/paper_fixture \
  --output scratch/refactor_validation/paper_model.json
```

It currently reproduces the shared error above; a passing result must contain
three energy/force predictions and a five-step CPU trajectory matching the baseline.

## Adoption gates — deliberately not performed

- JACS resubmission confirmed complete.
- Final revision fixes merged from main into this branch and revalidated.
- Completed real-model/dataset comparisons verified, including phase transitions.
- Known baseline failures assessed separately from this structural change.
- No affected running/pending jobs, successor chains or background writers.
- Retain the original checkout/environment for the JACS reproduction record.

Only new work should first adopt the reorganized checkout. Rollback routes future
work to the preserved checkout; it never resets code underneath active jobs.
