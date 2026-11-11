#!/usr/bin/env bash
# Print the newest resumable run directory under a --save_dir, or nothing.
#
# run.py does not train into --save_dir; it creates a timestamped subdirectory
# per launch (`results/revision/water_wb97mv_def2tzvpd/2026-08-19_a1B2c3D4/`) and
# puts args.txt, best_<ID>.pth and checkpoints/ in there. Everything downstream
# wants that inner directory: --restart loads `<dir>/checkpoints/` and
# csh_evaluate.py reads `<dir>/args.txt`.
#
# Both callers previously looked in --save_dir itself, where neither file ever
# appears. The failures were silent and opposite: training re-started from step
# zero on every 24 h window, so a 300k-step run could never finish, and the
# analysis reported "no trained model yet" forever -- which is the expected
# message early in a campaign and so would not have looked wrong.
#
# "Newest" is by mtime rather than by name so a resumed run, which touches its
# checkpoints without renaming its directory, stays the one that is picked up.
#
# Usage:  latest_run_dir.sh SAVE_DIR [--any]
#   default   only directories holding a checkpoint, i.e. ones worth resuming
#   --any     any directory holding args.txt, for evaluating a run whose
#             checkpoints were pruned

set -uo pipefail

SAVE_DIR=${1:?usage: latest_run_dir.sh SAVE_DIR [--any]}
MODE=${2:-checkpoint}

[[ -d "$SAVE_DIR" ]] || exit 0

best=""
best_mtime=0
for candidate in "$SAVE_DIR"/*/; do
  candidate=${candidate%/}
  [[ -d "$candidate" ]] || continue
  if [[ "$MODE" == "--any" ]]; then
    [[ -f "${candidate}/args.txt" ]] || continue
  else
    # A directory with args.txt but no checkpoint is a launch that died before
    # its first save; resuming from it would fail, so it is not a candidate.
    compgen -G "${candidate}/checkpoints/*.pth" > /dev/null 2>&1 ||
      compgen -G "${candidate}/best_*.pth" > /dev/null 2>&1 || continue
  fi
  mtime=$(stat -c %Y "$candidate" 2>/dev/null || echo 0)
  for f in "${candidate}/checkpoints/latest_checkpoint.pth" "${candidate}/args.txt"; do
    [[ -f "$f" ]] || continue
    t=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    (( t > mtime )) && mtime=$t
  done
  if (( mtime > best_mtime )); then
    best=$candidate
    best_mtime=$mtime
  fi
done

[[ -n "$best" ]] && echo "$best"
exit 0
