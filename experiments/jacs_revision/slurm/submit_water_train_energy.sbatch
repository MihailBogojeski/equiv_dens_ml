#!/bin/bash
# Stage-D energy training for one or more water-campaign models.
#
# Solo:  THEORY=... RUN=... sbatch scripts/revision/submit_water_train_energy.sbatch
# Pack:  ENERGY_PACK=run:theory,run:theory SLOTS=2 sbatch ...
#
# Two energy heads share one GPU so host-bound collate on one fills kernels
# on the other. Sequential A-then-B would not raise SM util.
#SBATCH --job-name=water-train-en
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --comment="gpu_mps=yes"
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/water_train_energy_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export OMP_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p logs

SLOTS=${SLOTS:-2}
ENERGY_PACK=${ENERGY_PACK:-}

if [[ -n "$ENERGY_PACK" ]]; then
  # Comma-separated RUN:THEORY[:SEED] list from the watchdog.
  IFS=',' read -r -a specs <<< "$ENERGY_PACK"
  ./.venv/bin/python scripts/revision/run_energy_pack.py \
    --slots "$SLOTS" --specs "${specs[@]}"
  exit $?
fi

THEORY=${THEORY:-wb97mv_def2tzvpd}
RUN=${RUN:-water}
spec="${RUN}:${THEORY}"
if [[ -n "${INIT_SEED:-}" ]]; then
  spec="${spec}:${INIT_SEED}"
fi
FORCE_NEW_FLAG=()
if [[ "${FORCE_NEW:-0}" == "1" ]]; then
  FORCE_NEW_FLAG=(--force-new)
fi
./.venv/bin/python scripts/revision/run_energy_pack.py \
  --slots 1 --specs "$spec" "${FORCE_NEW_FLAG[@]}"
