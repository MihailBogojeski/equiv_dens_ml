#!/bin/bash
# R2.2: 25 ps MACE-OFF ethanol IR on CPU vs archived DenSNet 25 ps.
#SBATCH --job-name=mace-etoh-ir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --output=logs/mace_ethanol_ir_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""
mkdir -p logs results/revision/ir_mace_ethanol

./.venv/bin/python scripts/revision/analysis/run_mace_ethanol_ir.py \
  --structure datasets/ethanol_train_10.xyz \
  --steps 25000 \
  --timestep-fs 1.0 \
  --sample-every 10 \
  --device cpu \
  --out-dir results/revision/ir_mace_ethanol
