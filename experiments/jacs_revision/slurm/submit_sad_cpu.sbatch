#!/bin/bash
# SAD-prior and DF baselines (ε_ρ, LDA exchange) against the full SCF density on
# the CPU, one label set per array task (score_density_functionals.py --sad-only
# --cpu-reference). Task list: results/revision/rescore_fullref/sad_cpu_tasks.txt
#   sbatch --array=0-$(( $(wc -l < results/revision/rescore_fullref/sad_cpu_tasks.txt) - 1 )) \
#     experiments/jacs_revision/slurm/submit_sad_cpu.sbatch
#SBATCH --job-name=sad-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --output=logs/sad_cpu_%A_%a.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
read -r manifest label model < <(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" results/revision/rescore_fullref/sad_cpu_tasks.txt)
echo "task ${SLURM_ARRAY_TASK_ID}: ${label} (prior of ${model})"
./.venv/bin/python -B experiments/jacs_revision/analysis/score_density_functionals.py --manifest "$manifest" \
  --only "$label" --models "$model" --sad-only --cpu-reference --device cpu \
  --out-dir results/revision/density_functionals_sad_cpu
