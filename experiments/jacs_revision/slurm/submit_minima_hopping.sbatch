#!/bin/bash
# DenSNet minima hopping on terthiophene (Frank et al. 2024 protocol).
#
# Usage:
#   sbatch scripts/revision/thiophene/submit_minima_hopping.sbatch
#
# Resubmit with the same command to continue from minima.json (--resume).
#SBATCH --job-name=thio-mh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/thiophene_mh_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE=offline

N_ESCAPES=${N_ESCAPES:-500}
ESCAPE_STEPS=${ESCAPE_STEPS:-200}
RELAX_STEPS=${RELAX_STEPS:-200}
OUT=${OUT:-results/revision/thiophene_mh/minima.json}
mkdir -p logs results/revision/thiophene_mh

echo "=== terthiophene MH ${N_ESCAPES} escapes on $(hostname) at $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

RESUME=()
if [[ -f "$OUT" ]]; then
  RESUME=(--resume)
  echo "resuming from ${OUT}"
fi

./.venv/bin/python scripts/revision/thiophene/minima_hopping_densnet.py \
  --structure datasets/thiophene3mer_init.npy \
  --model paper/models/polythiophene/2024-03-23_1XDL67zp_ext \
  --args-file config/md/nn/polythiophene_2mer_500ps.txt \
  --np-dataset datasets/thiophene_all_train_d4.npy \
  --use-gpu \
  --n-escapes "$N_ESCAPES" \
  --escape-steps "$ESCAPE_STEPS" \
  --relax-steps "$RELAX_STEPS" \
  --checkpoint-every 10 \
  --out "$OUT" \
  "${RESUME[@]}"

echo "finished $(date -Is) exit $?"
