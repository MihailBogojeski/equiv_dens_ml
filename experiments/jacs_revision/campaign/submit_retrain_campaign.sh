#!/bin/bash
# Submit the full multiphase retraining campaign (density then energy stages).
#
# Usage:  bash scripts/revision/submit_retrain_campaign.sh
#   DRY_RUN=1  print sbatch commands without submitting

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DRY_RUN=${DRY_RUN:-0}
submit() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: $*"
  else
    "$@"
  fi
}

for theory in wb97mv_def2tzvpd pbe_d4_avdz; do
  for run in water malonaldehyde water_direct; do
    submit env THEORY="$theory" RUN="$run" sbatch \
      --job-name="tr-${run}-${theory}" \
      scripts/revision/submit_water_train.sbatch
  done
  if [[ "$theory" == "pbe_d4_avdz" ]]; then
    for cut in 4 5 6 8; do
      submit env THEORY="$theory" RUN="cutoff_${cut}" sbatch \
        --job-name="tr-cutoff_${cut}-${theory}" \
        scripts/revision/submit_water_train.sbatch
    done
  fi
done

for seed in 7 123 20260821; do
  submit env THEORY=wb97mv_def2tzvpd RUN=water INIT_SEED="$seed" sbatch \
    --job-name="tr-water-wb97mv_def2tzvpd-s${seed}" \
    scripts/revision/submit_water_train.sbatch
done

echo "$(date -Is) submitted density multiphase jobs (energy stages follow via watchdog)"
