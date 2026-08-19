#!/bin/bash
#SBATCH --job-name=csh8_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=l40s_public
#SBATCH --output=logs/csh8_train_%j.out

# 8-element CSH DenSNet density training on the L40S public queue.
# Do not start this interactively on gl056 (tps-cofolding / gpu_burn).
# --exclude=gl056 is rejected by the scheduler, so the queue assigns the node.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/mh7373/projects/equiv_dens_ml}"
mkdir -p logs results/revision/omol_csh_neutral_8el_001

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

./.venv/bin/python run.py train --mode density \
  @config/training/omol_csh_neutral_8el_001.txt
