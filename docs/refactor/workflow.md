# Repository consolidation workflow

## Goal and boundaries

One reusable `equiv_dens` library, one CLI, and contained JACS experiments.
Every existing invocation keeps its arguments, working-directory conventions,
configuration precedence, scientific defaults, and output paths.

Development branch: `refactor/repository-layout`. Production stays on `main`.
No pushing, production deployment, dependency upgrades, cancellation/submission
of real Slurm jobs, manuscript edits, data migration, or numerical redesign.
Adoption waits until after JACS resubmission and the adoption gates below.

## Workspaces

```text
/scratch/mh7373/projects/
  equiv_dens_ml/             production; do not edit
    .venv/                  production environment; do not repoint
    datasets/ results/      ongoing jobs may read/write here
    logs/ docs/             production logs and manuscript repositories
  equiv_dens_ml_refactor/    development branch and independent environment
    scratch/refactor_validation/   temporary validation only
  equiv_dens_ml_baseline/    detached source at the recorded baseline commit
    .venv/                  independent reference environment
    scratch/refactor_validation/   reference validation only
```

Git metadata/history is shared. Writable environments, outputs, checkpoint
directories, shard queues, claims and completion markers are not shared.

## Target structure

```text
src/equiv_dens/
  cli.py  __main__.py  paths.py
  commands/{train,infer,evaluate,md,gxtb_md,dipole,experiment}.py
  training/workflows/{density,joint,multiphase,energy}.py
  training/{trainer,model_loader,parse_command_line_arguments,errors}.py
  nn/ data/ inference/ md/ integral/ density_functionals/ utils/ compat/
  train_all.py                       old entry point, forwarding only
experiments/
  jacs_revision/
    README.md tasks.json config/ lib/ water/ qm7x/ csh/ hbonds/
    thiophene/ analysis/ campaign/ slurm/
  legacy/{experiments_scripts,notebooks}/
scripts/{training,revision,md}/        existing paths, compatible adapters
scripts/{analysis,data}/               existing general utilities
config/ args/                         general configs and old-path aliases
tests/{compatibility,fixtures/refactor,revision}/
docs/refactor/{workflow.md,migration_map.json,validation.md}
docs/repository_layout.md
docs/{equiv-ml-paper,response-to-jacs-review}/  retain gitlink paths/pins
datasets/ results/ logs/ scratch/      keep established paths
```

Model, dataset, loss and other checkpoint-serialized classes keep their module
locations. Experiments are repository-local code, not another installable library.

## Stage 1: isolation and baseline

```python
base = git(production, "rev-parse", "main")
git_worktree_add(development, branch="refactor/repository-layout", start=base)
git_worktree_add(reference, detached=True, start=base)
for workspace in (development, reference):
    create_independent_environment(workspace, production_dependency_versions)
    install_local_package(workspace)
    assert imported_package_path(workspace).is_relative_to(workspace)
record_baseline(base, dependencies, submodule_pins, production_code_hashes)
```

Gate: own environments import their own source; production branch and code stay
unchanged. Record a read-only job inventory without altering scheduler state.

## Stage 2: map compatibility and characterize baseline

Inventory every affected executable, helper import, config and notebook.

| Material | Treatment |
| --- | --- |
| Core entry points | Importable package implementation plus old-path wrapper |
| JACS programs | Canonical experiment module plus wrapper/import shim |
| Slurm files | Canonical file with full directives plus old-path symlink |
| Revision-only configs | Byte-preserved canonical file plus old-path symlink |
| Shared/general configs | Keep existing location |
| Legacy scripts/notebooks | Group under legacy; preserve old paths |
| Serialized classes | Keep existing import path |
| Data, results, checkpoints, manuscripts | No relocation |

```python
for entry in affected_entries:
    record_arguments_cwd_environment_outputs(entry)
    record_canonical_destination_and_adapter(entry)
for duplicates in suspected_duplicates:
    if executable_behavior_differs(duplicates):
        preserve_explicit_compatibility_variants(duplicates)
```

Run only reviewed tests, from the isolated reference checkout. Check absolute
paths and subprocess effects before executing a test or even a help command.
Gate: each moved old path has a destination and preservation strategy.

## Stage 3: core commands

All existing `run.py`, `equiv-dens`, direct-script and config invocations remain.
Move script bodies into guarded `main(argv=None)` functions without reordering
scientific operations. Four explicit workflows remain four workflows.

```python
WORKFLOWS = {mode: import_path for mode in
             ("density", "joint", "multiphase", "energy")}
def main(argv=None):
    mode, remaining = parse_training_mode(argv)
    return import_module(WORKFLOWS[mode]).main(remaining)

def density_main(argv=None):
    args, hyperparams = parse_command_line_arguments(argv=argv)
    state = existing_initialization(args, hyperparams)
    datasets = existing_dataset_preparation(args, state)
    loaders = existing_loaders(args, datasets)
    model = existing_model_loading(args, datasets)
    trainer = existing_trainer(args, state, loaders, model)
    trainer.run(...)
    existing_final_evaluation(...)
```

Extend the parser with keyword-only `argv`; preserve the existing `arg_file`
whitespace parsing, `@file` semantics, defaults, conversions and overlay order.
Old direct entry points call the same function with the same cwd. The main CLI
keeps its subprocess boundary and executes package modules instead of files in
`scripts/`. A checkout installation retains its checkout-root cwd; a standalone
wheel uses caller cwd. Modules do not initialize W&B, datasets or runs on import.

Preserve RNG call order, splits, batching, phase order/freezing, units, basis,
SAD, normalization, losses, optimizer/scheduler/EMA, resume and output naming.
Worker helpers remain module-level and picklable. Extract only proven-equivalent
common code. Gate: invocation, worker and checkpoint comparisons pass.

## Stage 4: campaign and Slurm relocation

Move JACS code into its themed experiment directories and use qualified imports.
Keep shard claims, recovery, completion/stop markers, thresholds and output paths.
Slurm adapters must preserve effective directives; never resubmit a second job
from an adapter. Canonical Slurm files still start with their original directives
and `cd "${SLURM_SUBMIT_DIR:-$PWD}"` conventions.

Two watchdog variants differ and must retain these profiles:

```python
profiles = {
  "root_compat": {"rackers_large": False, "rackers_rank": 99,
                  "honor_skip_training": False},
  "water_compat": {"rackers_large": True, "rackers_rank": 10,
                   "honor_skip_training": True},
}
```

Old wrappers select their profile internally; public arguments stay unchanged.
Canonical/default watchdog behavior follows the production root entry. Consume
internal settings without exporting them to child jobs. Find the project by
walking ancestors, not counting directory levels. Test both original script
locations and snapshots copied into `logs/watchdog/`.

Gate: scheduler-stub tests preserve directives, submission arguments, dependency
strings, snapshot startup, claim handling and both watchdog profiles.

## Stage 5: experiment CLI, legacy layout and packaging

```text
equiv-dens experiment jacs list
equiv-dens experiment jacs run water/campaign_status --root datasets/revision/shards
equiv-dens experiment jacs submit submit_water_train --partition=a100_chemistry
```

Use a static `tasks.json` registry for executable Python, shell and Slurm tasks.
Names are canonical relative paths without extensions; Slurm job IDs use unique
stems. Helpers are not exposed as commands. Forward task arguments verbatim.

```python
task = registry[name]
if action == "submit":
    require(task.kind == "slurm")
    argv = ["sbatch", *remaining, task.canonical_path]
elif task.kind == "python":
    argv = [sys.executable, "-m", task.module, *remaining]
elif task.kind == "shell":
    argv = ["bash", task.canonical_path, *remaining]
else:
    error("Use submit for a Slurm task")
return subprocess.call(argv, cwd=checkout, env=os.environ.copy())
```

Keep legacy execution/import context through adapters. Make pyproject authoritative
with a minimal setup.py shim and package numerical assets, including CG tables.
Core commands must work from a wheel outside the checkout. No dependency upgrades.

## Stage 6: validation

Every run gets a temporary fixture/output directory. No shared writable data,
live checkpoints, production symlinks, online W&B or real scheduler submissions.
Copy only completed stable checkpoints; use synthetic data where appropriate.

```python
for mode in ("density", "joint", "multiphase", "energy"):
    before = run(reference, mode, fixed_fixture, fixed_seeds, temporary_output())
    after = run(development, mode, fixed_fixture, fixed_seeds, temporary_output())
    compare_arguments_splits_initial_parameters(before, after)
    compare_predictions_losses_and_one_step(before, after)
    compare_optimizer_scheduler_ema(before, after)
    compare_next_step(resume(reference, before.checkpoint),
                      resume(development, before.checkpoint))
```

Require equality for deterministic same-environment CPU runs. Preserve established
test tolerances; never widen them to hide unexplained changes. Normalize only
identified timestamps, random IDs and temporary path prefixes.

Test old/new invocation behavior, overlay order, all four workflows, worker
startup, checkpoint resume, fixed predictions, short deterministic MD, JACS
metrics, Slurm directives/dependencies, watchdog snapshots, failure/stop markers,
wheel assets and output isolation.

The prior targeted result was 158 passed / one missing IR-overlay PNG failure;
refresh it in the isolated baseline and record other baseline failures separately.
No silent scientific fixes. Missing reference data/dependencies are explicit
unverified gates, not passing checks.

## Review and adoption

Local commit sequence: baseline/contracts; core; campaign; scheduler compatibility;
experiment interface/legacy; validation/docs. Update the migration map throughout.
Merge subsequent relevant main fixes into this branch and rerun affected tests.
Never merge the refactor back during resubmission work and never push.

```python
ready = all([
  resubmission_confirmed_complete,
  final_revision_fixes_included,
  required_validation_verified,
  no_unresolved_behavior_changes,
  no_affected_running_or_pending_jobs,
  no_affected_successor_chains,
  no_affected_background_writers,
])
```

Retain production as the JACS reference. Adopt for new work first. Rollback routes
future work to that reference; never reset code underneath running jobs.

## Completion checklist

- [x] Production code/environment/campaign remain untouched.
- [x] Refactor is local on the dedicated branch.
- [x] One maintained implementation per equivalent behavior.
- [x] Core library and contained experiment tree are documented.
- [x] Existing commands, helper imports and config paths remain compatible.
- [x] Behavioral variants are explicit and tested.
- [ ] Required comparisons verified; gaps separately reported.
- [x] Workflow, migration map, baseline and validation report saved.
- [x] Adoption deferred until all post-resubmission gates pass.
- [x] Nothing pushed.

Implementation and controlled-fixture comparisons are complete. Full real-model
comparisons remain unverified; see validation.md for the evidence and adoption gates.
