#!/bin/bash
# Per-step timing scan (benchmark_step_timing.py drive), R1.6 / R2.2 / R3.2 / R3.4.
#
# Whole node (the site rejects --exclusive; all four GPUs and all CPUs of a
# 64-core A100 node give the same isolation), gpu48 QOS:
#   OUTDIR=results/revision/timing_v2/whole_node sbatch --gres=gpu:4 --cpus-per-task=64 --mem=400G \
#     experiments/jacs_revision/slurm/submit_step_timing.sbatch
# One dedicated GPU and 16 dedicated cores on a shared node, gpu168 QOS:
#   OUTDIR=results/revision/timing_v2/dedicated_gpu sbatch --time=2-01:00:00 \
#     experiments/jacs_revision/slurm/submit_step_timing.sbatch
# Only the first GPU and the first THREADS allocated cores are used.
#SBATCH --job-name=step-timing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --output=logs/step_timing_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${OUTDIR:?set OUTDIR}"
THREADS=${THREADS:-16}
mkdir -p "$OUTDIR"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}   # first allocated GPU only
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 WANDB_MODE=offline
export CPATH="$PWD/tools/python312-headers${CPATH:+:$CPATH}"   # Triton headers for AIMNet2

{
  echo "host $(hostname)  job ${SLURM_JOB_ID}  cpus ${SLURM_CPUS_PER_TASK}  gpu ${CUDA_VISIBLE_DEVICES}"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
  echo "other jobs on this node:"; squeue -w "$(hostname -s)" -h -o "%i %u %C %b %j" || true
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"
} > "$OUTDIR/node_${SLURM_JOB_ID}_${DEVICES:-cuda,cpu}.txt" 2>&1
cat "$OUTDIR/node_${SLURM_JOB_ID}_${DEVICES:-cuda,cpu}.txt"

./.venv/bin/python -B experiments/jacs_revision/analysis/benchmark_step_timing.py drive \
  --devices "${DEVICES:-cuda,cpu}" --threads "$THREADS" --n-warmup 10 --n-timed 100 ${AIMD_PAPER---aimd-paper} \
  ${METHODS:+--methods "$METHODS"} ${SYSTEMS:+--systems "$SYSTEMS"} \
  --out "$OUTDIR/timing_${DEVICES:-cuda,cpu}.json"
