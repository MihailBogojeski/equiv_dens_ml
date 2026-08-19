#!/bin/bash
# Keep re-submitting one shard array until every shard of it is finished.
#
# Array tasks die at the walltime, nodes drain, and ORCA occasionally fails a
# single frame. Because the worker resumes per frame and skips finished shards,
# the recovery for all of those is the same: submit the array again. This loop
# does that on a timer and stops when nothing is left, so a campaign spanning
# many walltime windows needs no attention in between.
#
# The job name is derived from the output directory rather than shared, because
# the "is a wave still draining?" check below is a squeue count by name: with one
# name across all splits, no split could ever re-submit while any other split was
# still running, and the campaign would stall on its slowest member.
#
# Usage:
#   scripts/revision/watch_and_resubmit.sh SHARD_DIR OUTDIR THEORY [INTERVAL_S]
# Env:
#   SBATCH_FILE  submit script (default the small tier)
#   JOB_NAME     override the derived name
#   ONESHOT=1    do a single round and exit (used by the campaign driver)

set -euo pipefail
cd "$(dirname "$0")/../.."

SHARD_DIR=${1:?usage: watch_and_resubmit.sh SHARD_DIR OUTDIR THEORY [INTERVAL_S]}
OUTDIR=${2:?}
THEORY=${3:-wb97mv_def2tzvpd}
INTERVAL=${4:-1800}
CONCURRENT=${CONCURRENT:-200}
SBATCH_FILE=${SBATCH_FILE:-scripts/revision/submit_water_orca.sbatch}
STALE_S=${STALE_S:-86400}
MAX_ROUNDS=${MAX_ROUNDS:-200}
ONESHOT=${ONESHOT:-0}

# The name has to carry the theory as well as the split: the same geometries are
# labelled at PBE-D4 and at wB97M-V, and a name that only named the split would
# make the second level look like it was already queued and silently skip it.
split=$(basename "$(dirname "$OUTDIR")")
JOB_NAME=${JOB_NAME:-w-${split}-${THEORY}}

n_shards=$(find "$SHARD_DIR" -maxdepth 1 -name 'shard_*.json' | wc -l)
if [[ "$n_shards" -eq 0 ]]; then
  echo "no shards in $SHARD_DIR"
  exit 1
fi

for ((round = 1; round <= MAX_ROUNDS; round++)); do
  status=$(./.venv/bin/python scripts/revision/campaign_status.py \
    --shard-dir "$SHARD_DIR" --outdir "$OUTDIR" --stale-s "$STALE_S")
  spec=$(./.venv/bin/python scripts/revision/campaign_status.py \
    --shard-dir "$SHARD_DIR" --outdir "$OUTDIR" --stale-s "$STALE_S" --array-spec)
  echo "[$(date -Is)] ${JOB_NAME} round ${round}: ${status}"

  if [[ -z "$spec" ]]; then
    echo "[$(date -Is)] ${JOB_NAME}: nothing outstanding"
    break
  fi

  running=$(squeue -h -u "$USER" -n "$JOB_NAME" -o "%i" 2>/dev/null | wc -l)
  if [[ "$running" -eq 0 ]]; then
    echo "[$(date -Is)] ${JOB_NAME}: submitting ${spec}%${CONCURRENT}"
    SHARD_DIR="$SHARD_DIR" OUTDIR="$OUTDIR" THEORY="$THEORY" \
      sbatch --job-name="$JOB_NAME" --array="${spec}%${CONCURRENT}" "$SBATCH_FILE" || true
  else
    echo "[$(date -Is)] ${JOB_NAME}: ${running} tasks still queued/running"
  fi

  if [[ "$ONESHOT" -ne 0 || "$round" -eq "$MAX_ROUNDS" ]]; then
    break
  fi
  sleep "$INTERVAL"
done
