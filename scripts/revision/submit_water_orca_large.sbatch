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
#SBATCH --output=logs/water_orca_lg_%A_%a.out
#
# As in the small tier, the partition is left to the site submit filter. At a
# 24 h walltime the filter returns `cs` alone -- `cpu_short` caps out at 6 h --
# so this tier gains nothing today, but naming the partition could only ever
# cost it somewhere to run if that changes.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load orca/6.1.1
export OMP_NUM_THREADS=1
# See submit_water_orca.sbatch: the CPU partition has no CUDA runtime and the
# inherited LD_PRELOAD makes every ORCA subprocess log a failure it survives.
unset LD_PRELOAD

SHARD_DIR=${SHARD_DIR:?set SHARD_DIR}
OUTDIR=${OUTDIR:?set OUTDIR}
THEORY=${THEORY:-wb97mv_def2tzvpd}
RETRY_ERRORS=${RETRY_ERRORS:-0}

# Derived from the allocation for the same reason as in the small tier: ORCA's
# %maxcore is per process, so a fixed value silently stops matching --mem the
# moment either is tuned. See submit_water_orca.sbatch for the 75 %.
MEM_MB=${SLURM_MEM_PER_NODE:-98304}
MAXCORE_MB=${MAXCORE_MB:-$((MEM_MB * 3 / 4 / ${SLURM_CPUS_PER_TASK:-16}))}
echo "allocation ${MEM_MB} MB over ${SLURM_CPUS_PER_TASK:-16} cores -> ORCA maxcore ${MAXCORE_MB} MB"

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
