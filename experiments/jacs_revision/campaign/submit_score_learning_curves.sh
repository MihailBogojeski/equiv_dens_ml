#!/bin/bash
# Submit scoring jobs for the Wave 1 learning-curve matrix.
# Jobs whose save_dir has no checkpoint yet exit 0; re-run this later.
#
# Usage:  bash scripts/revision/submit_score_learning_curves.sh
#   DRY_RUN=1  print sbatch commands without submitting

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DRY_RUN=${DRY_RUN:-0}
THEORY=${THEORY:-pbe_d4_avdz}

submit() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: $*"
  else
    "$@"
  fi
}

splits=(water_id_test water_ood_size water_ood_order water_ood_density)
steps=(5000 10000 25000 50000 75000 100000)

for arm in sad direct; do
  for split in "${splits[@]}"; do
    for n in 50 125 250 500 1250; do
      submit env THEORY="$THEORY" MODE=ntrain ARM="$arm" N_TRAIN="$n" SPLIT="$split" sbatch \
        --job-name="sc-n${n}-${arm}-${split}" \
        scripts/revision/submit_score_curve_ckpts.sbatch
    done
    for step in "${steps[@]}"; do
      submit env THEORY="$THEORY" MODE=step ARM="$arm" STEP="$step" SPLIT="$split" sbatch \
        --job-name="sc-s${step}-${arm}-${split}" \
        scripts/revision/submit_score_curve_ckpts.sbatch
    done
    submit env THEORY="$THEORY" MODE=step ARM="$arm" STEP="" SPLIT="$split" sbatch \
      --job-name="sc-sfinal-${arm}-${split}" \
      scripts/revision/submit_score_curve_ckpts.sbatch
  done
done

echo "$(date -Is) submitted learning-curve scoring jobs"
