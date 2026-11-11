#!/bin/bash
#SBATCH --job-name=jacs-water-energy-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --array=0-1
#SBATCH --output=logs/jacs_water_energy_eval_%A_%a.out
set -euo pipefail
: "${CODE_ROOT:?set CODE_ROOT to the recovery checkout}"
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="$CODE_ROOT/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
theories=(pbe_d4_avdz wb97mv_def2tzvpd)
runs=(2026-09-14_Dw6lXobF 2026-09-14_w837jxPx)
theory=${theories[$SLURM_ARRAY_TASK_ID]}
./.venv/bin/python -u -B "$CODE_ROOT/scripts/revision/analysis/energy_generalization.py" \
  --repo "$PWD" --run "results/revision/water_${theory}_energy/${runs[$SLURM_ARRAY_TASK_ID]}" \
  --system water --theory "$theory" --out "results/revision/jacs_recovery_20260927/water_energy_${theory}"
