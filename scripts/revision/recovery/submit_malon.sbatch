#!/bin/bash
#SBATCH --job-name=jacs-malon-recover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/jacs_malon_recovery_%j.out
set -euo pipefail
: "${CODE_ROOT:?set CODE_ROOT to the recovery checkout}"
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="$CODE_ROOT/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 WANDB_MODE=offline
# Two small molecular trainings share the GPU, as in the water energy pack.
./.venv/bin/python -B "$CODE_ROOT/scripts/revision/recovery/run_malon_recovery.py" --repo "$PWD" --theory pbe_d4_avdz &
pbe_pid=$!
./.venv/bin/python -B "$CODE_ROOT/scripts/revision/recovery/run_malon_recovery.py" --repo "$PWD" --theory wb97mv_def2tzvpd &
hybrid_pid=$!
status=0
wait "$pbe_pid" || status=1
wait "$hybrid_pid" || status=1
exit "$status"
