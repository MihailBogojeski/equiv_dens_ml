#!/bin/bash
# Merge md40x geometry trees and submit the PBE-D4 labelling wave.
#SBATCH --job-name=md40x-pbe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --output=logs/md40x_pbe_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
unset LD_PRELOAD
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p logs/revision

WORKERS="${WORKERS:-16}" scripts/revision/water/finish_md40x_and_launch_pbe.sh
