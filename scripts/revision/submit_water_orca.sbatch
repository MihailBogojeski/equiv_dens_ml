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
#SBATCH --output=logs/water_orca_%A_%a.out
#
# No --partition on purpose. The site submit filter picks the partition list
# from the walltime, and for a request under 6 h it returns `cs,cpu_short`
# instead of the `cs` alone that naming the partition pins us to. That matters:
# this account's fair-share factor is 0.006, so on `cs` its tasks queue behind
# everyone else's, and an identical pair of probe jobs submitted together had the
# unpinned one running in a second while the `cs`-pinned one was still pending
# forty minutes later. Pinning the partition can only ever remove somewhere for
# the job to run.

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
RETRY_ERRORS=${RETRY_ERRORS:-0}

# ORCA's %maxcore is per MPI process, so the ceiling it implies is maxcore x
# nprocs. A fixed value is only safe for a fixed --mem, and the campaign now
# sizes --mem per split -- a flat 3000 MB here would have let ORCA believe it had
# 24 GB inside a 16 GB allocation and be OOM-killed mid-frame. Deriving it from
# what Slurm actually granted keeps the two in step whatever the split asks for.
#
# The 75 % leaves room for the worker's own Python process. It does not have to
# cover both stages at once: ORCA has exited by the time the density fit runs, so
# the peaks do not overlap, and the fit is the larger of the two only for the
# biggest clusters, which run in the large tier.
MEM_MB=${SLURM_MEM_PER_NODE:-16384}
MAXCORE_MB=${MAXCORE_MB:-$((MEM_MB * 3 / 4 / ${SLURM_CPUS_PER_TASK:-8}))}
echo "allocation ${MEM_MB} MB over ${SLURM_CPUS_PER_TASK:-8} cores -> ORCA maxcore ${MAXCORE_MB} MB"

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
