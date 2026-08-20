#!/bin/bash
# The campaign watchdog, hosted by the scheduler instead of by a login shell.
#
# Everything in this campaign that happens without a person -- re-submitting
# shards killed at the walltime, reclaiming orphaned ones, launching each
# training run when its inputs are ready, refreshing the error-versus-distance
# curve -- happens because watch_water_campaign.sh is running somewhere. Run
# from a login shell it is one `setsid` away from a process nobody owns, and it
# died exactly that way: the log stops mid-pass, no error, and the campaign sat
# for the rest of the afternoon with running jobs draining and nothing behind
# them. There is no alarm for this, because a watchdog that is gone writes
# nothing, and its last page looks the same as one that is merely between
# passes.
#
# So the scheduler holds it. Each job submits its successor with
# `--dependency=afterany` before doing any work, which covers all three ways
# this ends: the 2 day walltime, a node failure, or a cancellation. The chain is
# broken deliberately by creating STOP_FILE, which is also what to do before
# editing the campaign scripts, since bash reads a running script by byte offset
# and editing one in place corrupts the run.
#
# Usage:
#   sbatch scripts/revision/submit_water_watchdog.sbatch      # start the chain
#   touch results/revision/.watchdog_stop                     # stop it
#
# Short jobs, not one long one. The watchdog competes for the queue with the
# campaign it is supervising, and asking for the QOS maximum of two days put it
# behind everything: a one-core job backfills into almost any gap, but only if
# the gap is longer than its walltime, and two-day gaps do not come up. At four
# hours it starts within minutes. Hopping six times a day costs nothing, and the
# chain replaces a wedged watchdog within one hop rather than within two days.
#SBATCH --job-name=water-watchdog
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --output=logs/water_watchdog_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

unset LD_PRELOAD
mkdir -p logs results/revision

INTERVAL=${INTERVAL:-1800}
STOP_FILE=${STOP_FILE:-results/revision/.watchdog_stop}

if [[ -e "$STOP_FILE" ]]; then
  echo "[$(date -Is)] ${STOP_FILE} exists; not running and not chaining"
  exit 0
fi

# Only one chain. A second `sbatch` of this file would otherwise run forever
# alongside the first, and two watchdogs racing to submit the same array is a
# harder thing to notice than none at all. The older chain wins, and only
# *running* jobs count -- a chain's own successor sits pending by design.
mine=${SLURM_JOB_ID%%_*}
older=$(squeue -h -u "$USER" -n water-watchdog -t RUNNING -o "%A" 2>/dev/null |
  awk -v me="$mine" '$1+0 < me+0' | head -1)
if [[ -n "$older" ]]; then
  echo "[$(date -Is)] watchdog ${older} is already running; exiting without chaining"
  exit 0
fi

# Before the work, not after: a job killed at the walltime or by a node failure
# never reaches its own last line, and that is the case the chain exists for.
successor=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID}" \
  scripts/revision/submit_water_watchdog.sbatch 2>&1) &&
  echo "[$(date -Is)] successor queued as ${successor}" ||
  echo "[$(date -Is)] WARNING could not queue a successor: ${successor}"

# Run a copy. The repository script is edited while the campaign runs, and bash
# reads a script by byte offset as it goes, so an edit to the original mid-pass
# makes a running shell resume at the wrong place in the new text.
snapshot="logs/watchdog/campaign_${SLURM_JOB_ID}.sh"
mkdir -p logs/watchdog
cp scripts/revision/watch_water_campaign.sh "$snapshot"
chmod +x "$snapshot"

# One round is a full sweep of the shard tree plus whatever it submits; sized so
# the loop ends a little before the walltime rather than being cut mid-sbatch.
# Read from the scheduler rather than restating the --time above, which cannot
# be a variable and so would drift from this the first time one of them changed.
left=$(squeue -h -j "${SLURM_JOB_ID}" -o "%L" 2>/dev/null | tr -d ' ')
seconds_left=$(awk -F'[-:]' '
  NF == 4 { print $1 * 86400 + $2 * 3600 + $3 * 60 + $4; exit }
  NF == 3 { print $1 * 3600 + $2 * 60 + $3; exit }
  NF == 2 { print $1 * 60 + $2; exit }
  { print 0 }' <<< "$left")
(( seconds_left > 0 )) || seconds_left=$(( 4 * 3600 ))
rounds=$(( (seconds_left - 1200) / INTERVAL ))
(( rounds > 0 )) || rounds=1

echo "[$(date -Is)] watchdog ${SLURM_JOB_ID} on $(hostname): ${rounds} rounds of ${INTERVAL}s"
started=$(date +%s)
MAX_ROUNDS="$rounds" "$snapshot" "$INTERVAL"
status=$?
lived=$(( $(date +%s) - started ))
echo "[$(date -Is)] watchdog ${SLURM_JOB_ID} exited ${status} after ${lived}s"

# The successor starts the moment this job ends, so a run that dies on its first
# line would be replaced immediately and the chain would become a submission
# loop against the scheduler. A watchdog that cannot survive one round is broken
# in a way retrying will not fix, so hold the slot instead of handing it back.
MIN_LIFETIME_S=${MIN_LIFETIME_S:-600}
if (( lived < MIN_LIFETIME_S )); then
  echo "[$(date -Is)] died after ${lived}s, under the ${MIN_LIFETIME_S}s floor; holding before the successor starts"
  sleep $(( MIN_LIFETIME_S - lived ))
fi
