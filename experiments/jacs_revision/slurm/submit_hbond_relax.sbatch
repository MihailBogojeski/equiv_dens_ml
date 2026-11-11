#!/bin/bash
# R1.1: relax size-OOD water clusters under the published ωB97M-V water model.
#SBATCH --job-name=hbond-relax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --output=logs/hbond_relax_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=offline
mkdir -p logs results/revision/hbond/wb97mv_def2tzvpd

# Density-only itJSxLYq has no energy/forces. Stage-D seed7 energy head does.
RUN_DIR=${RUN_DIR:-results/revision/water_wb97mv_def2tzvpd_seed7_energy/2026-09-11_UkU20xmG}
NP=${NP:-datasets/revision/water_labels/water_train_wb97mv_def2tzvpd_base.npy}
LIMIT=${LIMIT:-50}

if [[ ! -f "${RUN_DIR}/checkpoints/latest_checkpoint.pth" && ! -f "${RUN_DIR}/best_itJSxLYq.pth" ]]; then
  echo "no water checkpoint in ${RUN_DIR}"; exit 1
fi

./.venv/bin/python scripts/revision/water/relax_water_clusters.py \
  --xyz datasets/revision/water_clusters/ood_size.xyz \
  --run-dir "$RUN_DIR" \
  --np-dataset "$NP" \
  --limit "$LIMIT" \
  --out-xyz results/revision/hbond/wb97mv_def2tzvpd/ood_size_densnet_relaxed.xyz \
  --out-json results/revision/hbond/wb97mv_def2tzvpd/ood_size_relaxation.json

./.venv/bin/python scripts/revision/water/evaluate_hbond_metrics.py \
  --ref datasets/revision/water_clusters/ood_size.xyz \
  --pred-xyz results/revision/hbond/wb97mv_def2tzvpd/ood_size_densnet_relaxed.xyz \
  --out results/revision/water_hbond_densnet_wb97mv.json
