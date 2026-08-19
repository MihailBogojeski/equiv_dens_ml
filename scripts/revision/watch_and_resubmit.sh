#!/bin/bash
# Keep re-submitting a shard array until every shard is finished.
#
# Array tasks die at the walltime, nodes drain, and ORCA occasionally fails a
# single frame. Because the worker resumes per frame and skips finished shards,
# the recovery for all of those is the same: submit the array again. This loop
# does that on a timer and stops when nothing is left, so a campaign spanning
# many walltime windows needs no attention in between.
#
# Usage:
#   scripts/revision/watch_and_resubmit.sh SHARD_DIR OUTDIR THEORY [INTERVAL_S]

set -euo pipefail
cd "$(dirname "$0")/../.."

SHARD_DIR=${1:?usage: watch_and_resubmit.sh SHARD_DIR OUTDIR THEORY [INTERVAL_S]}
OUTDIR=${2:?}
THEORY=${3:-wb97mv_def2tzvpd}
INTERVAL=${4:-1800}
CONCURRENT=${CONCURRENT:-300}
JOB_NAME=${JOB_NAME:-water-orca}
MAX_ROUNDS=${MAX_ROUNDS:-200}

n_shards=$(find "$SHARD_DIR" -maxdepth 1 -name 'shard_*.json' | wc -l)
if [[ "$n_shards" -eq 0 ]]; then
  echo "no shards in $SHARD_DIR"
  exit 1
fi
last=$((n_shards - 1))

for ((round = 1; round <= MAX_ROUNDS; round++)); do
  done_count=$(find "$OUTDIR" -maxdepth 2 -name 'shard.done' 2>/dev/null | wc -l)
  echo "[$(date -Is)] round ${round}: ${done_count}/${n_shards} shards done"
  if [[ "$done_count" -ge "$n_shards" ]]; then
    echo "[$(date -Is)] all shards complete"
    break
  fi

  # Only submit when nothing of ours is already queued, so the array is not
  # duplicated on every tick while a previous wave is still draining.
  running=$(squeue -h -u "$USER" -n "$JOB_NAME" -o "%i" 2>/dev/null | wc -l)
  if [[ "$running" -eq 0 ]]; then
    echo "[$(date -Is)] submitting array 0-${last}%${CONCURRENT}"
    SHARD_DIR="$SHARD_DIR" OUTDIR="$OUTDIR" THEORY="$THEORY" \
      sbatch --job-name="$JOB_NAME" --array="0-${last}%${CONCURRENT}" \
      scripts/revision/submit_water_orca.sbatch
  else
    echo "[$(date -Is)] ${running} tasks still queued/running; waiting"
  fi

  sleep "$INTERVAL"
done
