#!/bin/bash
# The large-cluster tier of the ORCA campaign.
#
# A 24-water cluster in def2-TZVPD is about 1030 basis functions, and a single
# hybrid frame there does not fit in the 4 h window the small tier uses. The
# cpu48 QOS allows two days, so the large tier trades concurrency for walltime
# and more cores per task rather than fighting the limit with tiny shards.
#SBATCH --job-name=water-orca-lg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --output=logs/water_orca_lg_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load orca/6.1.1
export OMP_NUM_THREADS=1

SHARD_DIR=${SHARD_DIR:?set SHARD_DIR}
OUTDIR=${OUTDIR:?set OUTDIR}
THEORY=${THEORY:-wb97mv_def2tzvpd}
MAXCORE_MB=${MAXCORE_MB:-5000}
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
  --claim-stale-s 86400 \
  $(test "$RETRY_ERRORS" -ne 0 && echo "--retry-errors")
