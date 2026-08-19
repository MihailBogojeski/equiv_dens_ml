#!/bin/bash
# Assemble one level of theory's labels and train DenSNet on them.
#
# Not submitted with --dependency on the labelling arrays. The watchdog
# re-submits arrays as tasks are killed at the walltime, so the array ids that
# exist when the campaign starts are not the ones that finish it, and a
# dependency on them would fire while most shards were still unlabelled. The
# watchdog launches this instead, once campaign_status.py says the shards are
# actually done.
#SBATCH --job-name=water-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=l40s_public
#SBATCH --output=logs/water_train_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

THEORY=${THEORY:-wb97mv_def2tzvpd}
MIN_COMPLETE=${MIN_COMPLETE:-0.6}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p logs

scripts/revision/assemble_water_labels.sh "$THEORY" "$MIN_COMPLETE"

case "$THEORY" in
  wb97mv_def2tzvpd)
    water_cfg=(@config/training/water_wb97mv_001.txt)
    malon_cfg=(@config/training/malonaldehyde_pt_001.txt)
    ;;
  pbe_d4_avdz)
    # Overlays on the wB97M-V configs so the two levels differ only in their
    # labels: the PBE-vs-hybrid comparison is only about the reference data if
    # every architecture and optimisation setting is identical.
    water_cfg=(@config/training/water_wb97mv_001.txt @config/training/water_pbe_orca_overlay.txt)
    malon_cfg=(@config/training/malonaldehyde_pt_001.txt @config/training/malonaldehyde_pbe_orca_overlay.txt)
    ;;
  *)
    echo "no training config registered for theory ${THEORY}"
    exit 1
    ;;
esac

./.venv/bin/python run.py train "${water_cfg[@]}" 2>&1 | tee "logs/train_water_${THEORY}.log"
./.venv/bin/python run.py train "${malon_cfg[@]}" 2>&1 | tee "logs/train_malonaldehyde_${THEORY}.log"

echo "$(date -Is) ${THEORY} training finished"
