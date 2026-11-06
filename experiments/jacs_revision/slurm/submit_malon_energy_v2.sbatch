#!/bin/bash
# Replaces job 18681393 (archived recovery code, with the energy-loss and layer-norm
# bugs): finish both malonaldehyde density runs, fit their energy heads with the
# fixed working tree, then score energies along the proton-transfer coordinate.
#SBATCH --job-name=malon-energy-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/malon_energy_v2_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 WANDB_MODE=offline
# Two small molecular trainings share the GPU, as in the original job.
./.venv/bin/python -B experiments/jacs_revision/campaign/run_malon_energy_v2.py --theory pbe_d4_avdz &
pbe_pid=$!
./.venv/bin/python -B experiments/jacs_revision/campaign/run_malon_energy_v2.py --theory wb97mv_def2tzvpd &
hybrid_pid=$!
status=0
wait "$pbe_pid" || status=1
wait "$hybrid_pid" || status=1
exit "$status"
