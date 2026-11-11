#!/bin/bash
# v3 water DIRECT energy heads (fresh head, normalize_en=0; recipe in train_energy_head.sh). Both theories share one GPU.
# Save roots results/revision/water_direct_<theory>_energy_v3; resumable: re-submit (or chain) to continue.
#SBATCH --job-name=energy-v3-direct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/energy_v3_direct_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=4
for th in pbe_d4_avdz wb97mv_def2tzvpd; do
  LOG_TAG=direct_${SLURM_JOB_ID} experiments/jacs_revision/slurm/train_energy_head.sh direct $th &
done
wait
