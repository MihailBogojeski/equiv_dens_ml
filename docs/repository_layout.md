# Repository layout

`src/equiv_dens/` contains the reusable model, training, inference and MD code.
`equiv-dens`, `python -m equiv_dens` and `python run.py` use the same CLI.
Existing commands, configuration arguments and output paths remain supported.

```text
run.py / equiv-dens / python -m equiv_dens
  -> equiv_dens.cli
       -> package command -> training workflow / inference / MD
       -> experiment command -> registered JACS task
old direct Python/shell paths -> the same canonical implementation
old .sbatch paths -> canonical Slurm file, including original directives
```

The four training modes remain density, joint, multiphase and energy. Their
canonical implementations live in `training/workflows/`; old training scripts
are forwarding entry points. Importing a workflow does not start a run.

JACS-specific implementations live in `experiments/jacs_revision/`, grouped by
water, QM7-X, CSH, hydrogen bonds, thiophene, analysis, shared helpers and Slurm.
Experiment code is repository-local and is not part of the installed scientific
library. Legacy scripts and notebooks live in `experiments/legacy/`.

Compatibility paths under `scripts/`, `config/`, `args/`, `experiments_scripts/`
and `notebooks/` are retained. Maintain the canonical implementation rather than
adding logic to a wrapper. The complete old-to-new mapping is in
[refactor/migration_map.json](refactor/migration_map.json).

```bash
python run.py train --mode density @config/training/h2o_small_all_001.txt
python scripts/training/train_dens.py @config/training/h2o_small_all_001.txt
python run.py experiment jacs list
python run.py experiment jacs run water/campaign_status --root datasets/revision/shards
# Submit a registered Slurm file (campaign launcher tasks may also submit jobs):
# THEORY=pbe_d4_avdz RUN=water equiv-dens experiment jacs submit submit_water_train --partition=a100_chemistry
```

Configs retain their contents, overlay order and original relative-path semantics.
Revision-only configs have a canonical home in the campaign tree; their old paths
are relative symlinks. Shared/general configurations remain in `config/`.

Dataset, checkpoint, result, log, claim and completion-marker locations have not
been migrated. Manuscript submodule paths and pins are preserved. A standalone
wheel runs core commands using caller cwd; experiment commands require a checkout.

The two old water watchdog entry points retain their different behavior: the
root entry keeps its original Rackers classification and ignores SKIP_TRAINING;
the water entry retains its Rackers classification/rank and honors SKIP_TRAINING.
One canonical script implements those internal profiles. Its default is the root
entry profile. Snapshot wrappers discover the checkout by walking to pyproject.toml.

See the [workflow](refactor/workflow.md) and [validation report](refactor/validation.md).
This branch is not adopted into the production JACS campaign before resubmission.
