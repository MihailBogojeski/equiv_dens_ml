#!/bin/bash
# Independent cluster copy of the v3 water SAD energy heads: own save roots (*_energy_v3_slurm), own logs and scoring output,
# next to the local-GPU runs in *_energy_v3. Both theories share one GPU. Resumable: re-submit (or chain) to continue.
#SBATCH --job-name=energy-v3-slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/energy_v3_slurm_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=4
for th in pbe_d4_avdz wb97mv_def2tzvpd; do
  SAVE_ROOT=results/revision/water_${th}_energy_v3_slurm LOG_TAG=slurm_${SLURM_JOB_ID} \
  SCORE_OUT=results/revision/energy_forces_v2/water_${th}_v3_slurm \
    experiments/jacs_revision/slurm/train_energy_head.sh sad $th &
done
wait
