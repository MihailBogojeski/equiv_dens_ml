#!/bin/bash
# Stage-D energy training for QM7x (frozen density head from bjOUNzrR).
#
# Strict two-stage recipe: density was trained separately (Mihail, 2024);
# this job is energy/forces only, matching submit_water_train_energy.sbatch.
#
# Prerequisites:
#   results/revision/qm7x/.density_training_done
#   datasets/qm7x_{train,valid}_dft_augccpvdz.npy (+ orca_base npy)
#
# Usage: sbatch scripts/revision/submit_qm7x_energy.sbatch
#SBATCH --job-name=tre-qm7x-energy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/qm7x_energy_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# Long energy runs fragment the allocator; 16466935 OOM'd a 5.7 GB block
# after 27.8k steps with 3.8 GB free on an 80 GB A100.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs results/revision/qm7x_energy

density_parent="results/revision/qm7x"
density_marker="${density_parent}/.density_training_done"
if [[ ! -e "$density_marker" ]]; then
  echo "missing ${density_marker}; density stage not recorded" >&2
  exit 1
fi

density_run=$(cat "${density_parent}/.density_run_dir")
if [[ ! -d "$density_run" ]]; then
  echo "density run directory missing: ${density_run}" >&2
  exit 1
fi

for f in \
  datasets/qm7x_train_dft_augccpvdz.npy \
  datasets/qm7x_train_dft_augccpvdz_orca_base.npy \
  datasets/qm7x_valid_dft_augccpvdz.npy \
  datasets/qm7x_valid_dft_augccpvdz_orca_base.npy; do
  if [[ ! -f "$f" ]]; then
    echo "missing assembled label file ${f}" >&2
    exit 1
  fi
done

save_dir="results/revision/qm7x_energy"
extra=(
  --args_file_name=qm7x_energy_001
  --save_dir="$save_dir"
  --load_from="$density_run"
)

has_atomic_energies() {
  local args_file="$1/args.txt"
  [[ -f "$args_file" ]] || return 1
  grep -qE '^--atomic_energies=' "$args_file" || return 1
  grep -qE '^--atomic_energies=(None)?$' "$args_file" && return 1
  return 0
}

FORCE_NEW=${FORCE_NEW:-0}
energy_resume=$(scripts/revision/lib/latest_run_dir.sh "$save_dir")
if [[ "$FORCE_NEW" == "1" ]]; then
  echo "FORCE_NEW=1: not resuming ${energy_resume:-<none>}"
  energy_resume=""
elif [[ -n "$energy_resume" ]] && ! has_atomic_energies "$energy_resume"; then
  echo "not resuming ${energy_resume}: no --atomic_energies (global-mean labels)"
  energy_resume=""
fi
if [[ -n "$energy_resume" ]]; then
  echo "resuming qm7x energy from ${energy_resume}"
  extra+=(--restart="$energy_resume")
fi

./.venv/bin/python run.py train --mode energy @experiments/jacs_revision/config/training/qm7x_energy_001.txt \
  "${extra[@]}" 2>&1 | tee -a "logs/train_qm7x_energy.log"

energy_run=$(scripts/revision/lib/latest_run_dir.sh "$save_dir")
if [[ -z "$energy_run" ]]; then
  echo "energy training finished but no checkpoint under ${save_dir}" >&2
  exit 1
fi

./.venv/bin/python scripts/revision/analysis/validate_training_recipe.py "$energy_run"

touch "${density_parent}/.training_done"
touch "${save_dir}/.training_done"
echo "$(date -Is) qm7x energy stage finished (${energy_run})"
