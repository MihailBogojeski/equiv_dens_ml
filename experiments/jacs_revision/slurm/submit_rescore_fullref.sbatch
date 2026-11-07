#!/bin/bash
# Full-density re-score of the response-letter density errors (one manifest).
#
# Usage:
#   MANIFEST=results/revision/rescore_fullref/manifest_pbe_water.json \
#     sbatch experiments/jacs_revision/slurm/submit_rescore_fullref.sbatch
#
# The walltime is above 48 h only so the job is routed to the gpu168 QOS (the
# gpu48 per-user GPU cap is held by other runs). The job exits when the
# manifest is done and resumes from its partial outputs if resubmitted.
# Reference densities are evaluated on the GPU (gpu4pyscf): Torch cancels GPU
# jobs that sit at low utilisation for two hours, which the CPU-side PySCF
# evaluation caused on the first attempt (jobs 18704265/6).
#SBATCH --job-name=rescore-fullref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=2-01:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --output=logs/rescore_fullref_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${MANIFEST:?set MANIFEST to a manifest JSON}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export WANDB_MODE=offline
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1

echo "host $(hostname)  gpu ${CUDA_VISIBLE_DEVICES:-none}  manifest ${MANIFEST}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
only_args=()
for name in ${ONLY:-}; do only_args+=(--only "$name"); done   # space-separated label-set names
./.venv/bin/python -B experiments/jacs_revision/analysis/rescore_full_density.py \
  --manifest "$MANIFEST" "${only_args[@]}"
