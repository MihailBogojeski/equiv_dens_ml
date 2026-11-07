#!/bin/bash
# Several full-density re-score manifests at once on one GPU.
#
# Usage:
#   MANIFESTS="m1.json m2.json m3.json" sbatch experiments/jacs_revision/slurm/submit_rescore_parallel.sbatch
#
# Torch cancels a GPU job whose average utilisation stays below 50% for two
# hours. One process alternates between CPU work (grid, free-atom density) and
# GPU work (reference densities, model), so the GPU idles half the time;
# several processes, each owning different models (no shared output files),
# keep it busy and finish sooner. Each process resumes from its partial outputs.
#SBATCH --job-name=rescore-par
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=2-01:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --output=logs/rescore_par_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${MANIFESTS:?set MANIFESTS to space-separated manifest JSONs}"
read -r -a list <<< "$MANIFESTS"
n=${#list[@]}
threads=$(( ${SLURM_CPUS_PER_TASK:-16} / n )); (( threads < 2 )) && threads=2
export MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads OMP_NUM_THREADS=$threads
export WANDB_MODE=offline TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1
echo "host $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-none}: $n processes x $threads threads"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

pids=()
for m in "${list[@]}"; do
  log="logs/rescore_par_${SLURM_JOB_ID}_$(basename "$m" .json).log"
  ./.venv/bin/python -B experiments/jacs_revision/analysis/rescore_full_density.py --manifest "$m" ${EXTRA_ARGS:-} > "$log" 2>&1 &
  pids+=($!)
  echo "started $m (pid $!) -> $log"
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "all done, status $status"
exit $status
