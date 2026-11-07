#!/bin/bash
# Score one water-curve checkpoint (or one N_train run) on one reviewer split.
#
# Usage:
#   MODE=step ARM=sad STEP=5000 SPLIT=water_id_test \
#     sbatch scripts/revision/submit_score_curve_ckpts.sbatch
#   MODE=ntrain ARM=direct N_TRAIN=50 SPLIT=water_ood_size \
#     sbatch scripts/revision/submit_score_curve_ckpts.sbatch
#SBATCH --job-name=score-curve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/score_curve_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MODE=${MODE:-step}
ARM=${ARM:-sad}
SPLIT=${SPLIT:-water_id_test}
THEORY=${THEORY:-pbe_d4_avdz}
STEP=${STEP:-}
N_TRAIN=${N_TRAIN:-}

unset LD_PRELOAD
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p logs results/revision/eval/learning_curves

if [[ "$ARM" == sad ]]; then
  prefix="water"
else
  prefix="water_direct"
fi

if [[ "$MODE" == step ]]; then
  save_dir="results/revision/${prefix}_${THEORY}_curve"
  extra=()
  if [[ -z "${STEP}" ]]; then
    stem="step_final_${ARM}_${SPLIT}"
  else
    stem="step_${STEP}_${ARM}_${SPLIT}"
    extra=(--step="$STEP")
  fi
elif [[ "$MODE" == ntrain ]]; then
  if [[ -z "${N_TRAIN}" ]]; then
    echo "N_TRAIN is required for MODE=ntrain" >&2
    exit 1
  fi
  if [[ "$N_TRAIN" == 1250 ]]; then
    save_dir="results/revision/${prefix}_${THEORY}"
  else
    save_dir="results/revision/${prefix}_n${N_TRAIN}_${THEORY}"
  fi
  extra=(--n-train="$N_TRAIN")
  stem="ntrain_${N_TRAIN}_${ARM}_${SPLIT}"
else
  echo "unknown MODE ${MODE}" >&2
  exit 1
fi

run_dir=$(scripts/revision/lib/latest_run_dir.sh "$save_dir")
if [[ -z "$run_dir" ]]; then
  echo "no checkpoint yet under ${save_dir}; skip"
  exit 0
fi

out="results/revision/eval/learning_curves/${stem}.json"
echo "[$(date -Is)] ${MODE} ${ARM} ${SPLIT} ${save_dir} -> ${out}"
./.venv/bin/python scripts/revision/analysis/score_learning_curve_ckpts.py score \
  --run-dir "$run_dir" \
  --split "$SPLIT" \
  --arm "$ARM" \
  --theory "$THEORY" \
  --out "$out" \
  "${extra[@]}"
echo "[$(date -Is)] wrote ${out}"
