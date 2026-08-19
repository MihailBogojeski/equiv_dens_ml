#!/bin/bash
# One array task per shard of the water/OOD campaign, run in ORCA on CPU.
#
# The task is idempotent: a finished shard exits immediately, an unfinished one
# resumes at the first frame missing from status.jsonl, and a shard held by a
# live worker is skipped. Re-submitting the whole array is therefore always
# safe and is how the watchdog recovers walltime kills.
#SBATCH --job-name=water-orca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --output=logs/water_orca_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load orca/6.1.1
export OMP_NUM_THREADS=1
# The login environment preloads a CUDA runtime that does not exist on the CPU
# partition, and every ORCA subprocess reports the failure. Dropping it here
# keeps a real error visible in the log instead of buried under the noise.
unset LD_PRELOAD

SHARD_DIR=${SHARD_DIR:?set SHARD_DIR to the directory holding shard_*.json}
OUTDIR=${OUTDIR:?set OUTDIR to the results directory for this split}
THEORY=${THEORY:-wb97mv_def2tzvpd}
MAXCORE_MB=${MAXCORE_MB:-3000}
RETRY_ERRORS=${RETRY_ERRORS:-0}

SHARD=$(printf "%s/shard_%04d.json" "$SHARD_DIR" "${SLURM_ARRAY_TASK_ID}")
if [[ ! -f "$SHARD" ]]; then
  echo "no such shard: $SHARD"
  exit 0
fi

mkdir -p logs "$OUTDIR"

./.venv/bin/python scripts/revision/qm7x_orca_worker.py \
  --shard "$SHARD" \
  --outdir "$OUTDIR" \
  --theory "$THEORY" \
  --nprocs "${SLURM_CPUS_PER_TASK}" \
  --maxcore-mb "$MAXCORE_MB" \
  $(test "$RETRY_ERRORS" -ne 0 && echo "--retry-errors")
