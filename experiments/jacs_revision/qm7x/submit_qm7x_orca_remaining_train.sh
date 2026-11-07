#!/usr/bin/env bash
# Launch the nohup QM7x campaign watchdog (ORCA arrays -> assemble -> energy stage).
#
# Replaces the old tmux sequential chunk driver. ORCA tasks use short walltimes
# and --resume; the watchdog re-submits outstanding shards until train is full.
#
# Usage: bash scripts/revision/qm7x/submit_qm7x_orca_remaining_train.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

mkdir -p logs
WATCH="${ROOT}/scripts/revision/qm7x/watch_qm7x_campaign.sh"
PIDFILE="${ROOT}/logs/qm7x_watchdog.pid"

if [[ -f "$PIDFILE" ]]; then
  old_pid=$(cat "$PIDFILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "watchdog already running (pid ${old_pid})"
    exit 0
  fi
fi

nohup bash "$WATCH" >> logs/qm7x_watchdog.log 2>&1 &
echo $! > "$PIDFILE"
echo "started nohup watchdog pid $(cat "$PIDFILE") -> logs/qm7x_watchdog.log"
