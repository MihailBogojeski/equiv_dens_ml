#!/bin/bash
# Per-step timing scan (benchmark_step_timing.py drive), R1.6 / R2.2 / R3.2 / R3.4.
#
# Whole node (the site rejects --exclusive; all four GPUs and all CPUs of a
# 64-core A100 node give the same isolation), gpu48 QOS:
#   OUTDIR=results/revision/timing_v2/whole_node sbatch --gres=gpu:4 --cpus-per-task=64 --mem=400G \
#     experiments/jacs_revision/slurm/submit_step_timing.sbatch
# CPU-only runs (DEVICES=cpu) on a CPU node: a GPU allocation left idle for hours is cancelled on Torch.
#   OUTDIR=results/revision/timing_v2/dedicated_gpu sbatch --time=2-01:00:00 \
#     experiments/jacs_revision/slurm/submit_step_timing.sbatch
# Only the first GPU and the first THREADS allocated cores are used.
#SBATCH --job-name=step-timing-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --output=logs/step_timing_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${OUTDIR:?set OUTDIR}"
THREADS=${THREADS:-16}
mkdir -p "$OUTDIR"
export CUDA_VISIBLE_DEVICES=""
export TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1 WANDB_MODE=offline
export CPATH="$PWD/tools/python312-headers${CPATH:+:$CPATH}"   # Triton headers for AIMNet2

{
  echo "host $(hostname)  job ${SLURM_JOB_ID}  cpus ${SLURM_CPUS_PER_TASK}  gpu ${CUDA_VISIBLE_DEVICES}"
  echo "CPU-only node"
  echo "other jobs on this node:"; squeue -w "$(hostname -s)" -h -o "%i %u %C %b %j" || true
  lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"
} > "$OUTDIR/node_${SLURM_JOB_ID}_${DEVICES:-cuda,cpu}.txt" 2>&1
cat "$OUTDIR/node_${SLURM_JOB_ID}_${DEVICES:-cuda,cpu}.txt"

./.venv/bin/python -B experiments/jacs_revision/analysis/benchmark_step_timing.py drive \
  --devices "${DEVICES:-cuda,cpu}" --threads "$THREADS" --n-warmup 10 --n-timed 100 --aimd-paper \
  --out "$OUTDIR/timing_${DEVICES:-cuda,cpu}.json"
