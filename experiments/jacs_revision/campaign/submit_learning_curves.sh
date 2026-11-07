#!/bin/bash
# Wave 1 water learning curves: PBE-D4 SAD/direct subset trains + two
# checkpoint-keeping full-N runs. Does not touch the main campaign save_dirs.
#
# Usage:  bash scripts/revision/submit_learning_curves.sh
#   DRY_RUN=1  print sbatch commands without submitting
#   SCORE=1    also submit scoring jobs for any finished checkpoints

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DRY_RUN=${DRY_RUN:-0}
SCORE=${SCORE:-0}
THEORY=${THEORY:-pbe_d4_avdz}
export DRY_RUN THEORY

submit() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: $*"
  else
    "$@"
  fi
}

# N=1250 SAD/direct reuse the campaign runs (water_${THEORY} /
# water_direct_${THEORY}). Do not duplicate those 300k-step jobs.
for n in 50 125 250 500; do
  submit env THEORY="$THEORY" RUN="water_n${n}" sbatch \
    --job-name="tr-water-n${n}-${THEORY}" \
    scripts/revision/submit_water_train.sbatch
  submit env THEORY="$THEORY" RUN="water_direct_n${n}" sbatch \
    --job-name="tr-water-direct-n${n}-${THEORY}" \
    scripts/revision/submit_water_train.sbatch
done

submit env THEORY="$THEORY" RUN=water_curve sbatch \
  --job-name="tr-water-curve-${THEORY}" \
  scripts/revision/submit_water_train.sbatch
submit env THEORY="$THEORY" RUN=water_direct_curve sbatch \
  --job-name="tr-water-direct-curve-${THEORY}" \
  scripts/revision/submit_water_train.sbatch

echo "$(date -Is) submitted PBE learning-curve trainings (N=50/125/250/500 + keep-ckpt)"

if [[ "$SCORE" == "1" ]]; then
  bash scripts/revision/submit_score_learning_curves.sh
fi
