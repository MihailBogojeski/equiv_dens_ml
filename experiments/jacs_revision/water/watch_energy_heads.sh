#!/usr/bin/env bash
# Requeue the water and malonaldehyde energy packs across 24 h walls.
#
# One tre-pack at a time. Unfinished water heads are resubmitted first.
# Malonaldehyde energy starts only after both water parents have
# .training_done and density_log_finished.py accepts both malonaldehyde
# logs. Density fine-tune jobs are resumed separately and do not use the
# tre-pack name, so they can run beside the water energy pack.
#
# OUT_ROOT points at an empty tree so submit_water_train.sbatch's assemble
# step finds no ORCA results and does not rewrite datasets/revision/water_labels.
#
# Usage:  scripts/revision/water/watch_energy_heads.sh [INTERVAL_S]

set -euo pipefail

revision_repo_root() {
  local here d
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  d="$here"
  while [[ "$d" != "/" ]]; do
    if [[ -f "$d/pyproject.toml" ]]; then
      printf '%s\n' "$d"
      return 0
    fi
    d="$(dirname "$d")"
  done
  echo "could not find repo root from ${here}" >&2
  return 1
}
cd "$(revision_repo_root)"
# shellcheck source=scripts/revision/lib/gpu_queue_policy.sh
source scripts/revision/lib/gpu_queue_policy.sh

INTERVAL=${1:-1800}
MAX_ROUNDS=${MAX_ROUNDS:-1000}
THEORIES=(pbe_d4_avdz wb97mv_def2tzvpd)

mark_density_if_log_finished() {
  local name=$1
  local save_dir="results/revision/${name}"
  if ! ./.venv/bin/python scripts/revision/water/density_log_finished.py "$name"; then
    return 1
  fi
  if [[ -e "${save_dir}/.density_training_done" ]]; then
    return 0
  fi
  local run_dir
  run_dir=$(scripts/revision/lib/latest_run_dir.sh "$save_dir" || true)
  if [[ -z "$run_dir" ]]; then
    echo "[$(date -Is)] ${name}: log finished but no checkpoint directory" >&2
    return 1
  fi
  mkdir -p "$save_dir"
  echo "$run_dir" > "${save_dir}/.density_run_dir"
  touch "${save_dir}/.density_training_done"
  echo "[$(date -Is)] ${name}: density complete per log; wrote .density_training_done"
  return 0
}

ensure_malon_density() {
  local theory=$1
  local name="malonaldehyde_${theory}"
  local job="tr-malonaldehyde-${theory}"
  if mark_density_if_log_finished "$name"; then
    return 0
  fi
  local queued
  queued=$(squeue -h -u "$USER" -n "$job" -o "%i" 2>/dev/null | wc -l)
  if [[ "$queued" -gt 0 ]]; then
    echo "[$(date -Is)] ${job} still queued or running"
    return 0
  fi
  echo "[$(date -Is)] submitting density ${job}"
  # env, not --export: ENERGY_PACK and this OUT_ROOT stay intact. --export
  # splits on commas, which would break a two-spec pack.
  env OUT_ROOT=results/revision/_skip_assemble RUN=malonaldehyde THEORY="$theory" \
    sbatch --job-name="$job" \
      --partition="$EQUIV_GPU_PARTITIONS" --qos="$EQUIV_GPU_QOS" \
      scripts/revision/submit_water_train.sbatch || true
}

next_pack() {
  local malon_ready=$1
  shift
  QUEUED="$*" MALON_READY="$malon_ready" ./.venv/bin/python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, "scripts/revision/lib")
from energy_train_spec import (  # noqa: E402
    MALON_ENERGY_SPECS,
    WATER_ENERGY_SPECS,
    next_energy_pack,
    spec_is_done,
)

repo = Path(".").resolve()
done = [
    spec
    for spec in (*WATER_ENERGY_SPECS, *MALON_ENERGY_SPECS)
    if spec_is_done(repo, spec)
]
queued = os.environ.get("QUEUED", "").split()
ready = os.environ.get("MALON_READY") == "1"
print(",".join(next_energy_pack(done, queued, malon_density_ready=ready)))
PY
}

round=0
while (( round < MAX_ROUNDS )); do
  round=$((round + 1))
  malon_ready=1
  for theory in "${THEORIES[@]}"; do
    ensure_malon_density "$theory" || malon_ready=0
    if ! mark_density_if_log_finished "malonaldehyde_${theory}"; then
      malon_ready=0
    fi
  done

  queued_names=$(squeue -h -u "$USER" -o "%j" 2>/dev/null || true)
  pack=$(next_pack "$malon_ready" $queued_names)
  if [[ -n "$pack" ]]; then
    echo "[$(date -Is)] submitting energy pack tre-pack (${pack})"
    # env prefix: ENERGY_PACK contains commas, which --export would split.
    env ENERGY_PACK="$pack" SLOTS=2 sbatch --job-name=tre-pack \
      --partition="$EQUIV_GPU_PARTITIONS" --qos="$EQUIV_GPU_QOS" \
      scripts/revision/submit_water_train_energy.sbatch || true
  else
    echo "[$(date -Is)] no energy pack to submit (malon_density_ready=${malon_ready})"
  fi

  all_done=1
  for name in water_pbe_d4_avdz water_wb97mv_def2tzvpd \
      malonaldehyde_pbe_d4_avdz malonaldehyde_wb97mv_def2tzvpd; do
    if [[ ! -e "results/revision/${name}/.training_done" ]]; then
      all_done=0
    fi
  done
  if [[ "$all_done" == "1" ]]; then
    echo "[$(date -Is)] all four energy heads finished"
    exit 0
  fi
  sleep "$INTERVAL"
done
echo "[$(date -Is)] reached MAX_ROUNDS=${MAX_ROUNDS} with energy heads still unfinished" >&2
exit 1
