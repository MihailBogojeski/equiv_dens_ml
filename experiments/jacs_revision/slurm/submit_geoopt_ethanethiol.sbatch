#!/bin/bash
# R2.11: DenSNet vs PBE equilibrium on ethanethiol (CPU).
#SBATCH --job-name=etsh-geoopt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --output=logs/geoopt_ethanethiol_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=offline
mkdir -p logs results/revision

./.venv/bin/python scripts/revision/analysis/optimize_densnet.py \
  --structure results/revision/ethanethiol_train_frame0.xyz \
  --model paper/models/ethanethiol/2024-02-29_NUmID4hT_ext4 \
  --args-file config/md/nn/ethanethiol.txt \
  --np-dataset datasets/ethanethiol_combo_Aidx-1000.npy \
  --dft-xc pbe \
  --out results/revision/geoopt_ethanethiol.json
