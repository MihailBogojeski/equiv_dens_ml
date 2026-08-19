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
#   PARTITION/QOS/SBATCH_EXTRA/JOB_PREFIX  send this wave somewhere else
#                (see the scavenger wave in watch_water_campaign.sh)

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
JOB_PREFIX=${JOB_PREFIX:-w}

# Partition and QOS are overrides rather than edits to the submit script because
# the same shard list is drained by more than one wave at a time: a guaranteed
# wave on `cs` and a preemptible wave on `cpu_prem`. Options given to sbatch beat
# the #SBATCH directives in the file, so both waves share one submit script and
# there is no second copy to keep in sync.
SBATCH_ARGS=()
[[ -n "${PARTITION:-}" ]] && SBATCH_ARGS+=(--partition="$PARTITION")
[[ -n "${QOS:-}" ]] && SBATCH_ARGS+=(--qos="$QOS")
[[ -n "${SBATCH_EXTRA:-}" ]] && read -r -a extra <<< "$SBATCH_EXTRA" && SBATCH_ARGS+=("${extra[@]}")

# The name has to carry the theory as well as the split: the same geometries are
# labelled at PBE-D4 and at wB97M-V, and a name that only named the split would
# make the second level look like it was already queued and silently skip it.
# The prefix separates the waves for the same reason -- one name across both
# would let a queued scavenger task suppress the guaranteed submission.
split=$(basename "$(dirname "$OUTDIR")")
JOB_NAME=${JOB_NAME:-${JOB_PREFIX}-${split}-${THEORY}}

n_shards=$(find "$SHARD_DIR" -maxdepth 1 -name 'shard_*.json' | wc -l)
if [[ "$n_shards" -eq 0 ]]; then
  echo "no shards in $SHARD_DIR"
  exit 1
fi

for ((round = 1; round <= MAX_ROUNDS; round++)); do
  # One call, not one per output: each invocation asks the scheduler which
  # tasks are alive, and on a busy cluster that query alone can take a minute.
  read -r status spec < <(./.venv/bin/python scripts/revision/campaign_status.py \
    --shard-dir "$SHARD_DIR" --outdir "$OUTDIR" --stale-s "$STALE_S" --summary-and-spec)
  echo "[$(date -Is)] ${JOB_NAME} round ${round}: ${status//_/ }"

  if [[ -z "$spec" ]]; then
    echo "[$(date -Is)] ${JOB_NAME}: nothing outstanding"
    break
  fi

  running=$(squeue -h -u "$USER" -n "$JOB_NAME" -o "%i" 2>/dev/null | wc -l)
  if [[ "$running" -eq 0 ]]; then
    echo "[$(date -Is)] ${JOB_NAME}: submitting ${spec}%${CONCURRENT}"
    SHARD_DIR="$SHARD_DIR" OUTDIR="$OUTDIR" THEORY="$THEORY" \
      sbatch --job-name="$JOB_NAME" --array="${spec}%${CONCURRENT}" \
      "${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"}" "$SBATCH_FILE" || true
  else
    echo "[$(date -Is)] ${JOB_NAME}: ${running} tasks still queued/running"
  fi

  if [[ "$ONESHOT" -ne 0 || "$round" -eq "$MAX_ROUNDS" ]]; then
    break
  fi
  sleep "$INTERVAL"
done
