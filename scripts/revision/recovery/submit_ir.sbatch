#!/bin/bash
#SBATCH --job-name=jacs-ir-recover
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --array=0-3
#SBATCH --output=logs/jacs_ir_recovery_%A_%a.out
set -euo pipefail
: "${CODE_ROOT:?set CODE_ROOT to the recovery checkout}"
cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="$CODE_ROOT/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
methods=(densnet maceoff macepolar so3lr)
method=${methods[$SLURM_ARRAY_TASK_ID]}
out="results/revision/jacs_recovery_20260927/ethanethiol_ir/$method"
./.venv/bin/python -u -B "$CODE_ROOT/scripts/revision/analysis/run_mlip_ir_md.py" \
  --method "$method" --device cpu --out-dir "$out" --production-ps 50 --equilibration-ps 2 --windows-ps 25 50
