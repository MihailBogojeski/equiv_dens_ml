#!/usr/bin/env bash
# Nohup-friendly watchdog for QM7-X ORCA labelling, assembly, and energy training.
#
# ORCA: short-walltime CPU arrays (--resume skips finished one-frame shards).
# Training: strict two-stage recipe — density is pre-trained (bjOUNzrR); stage D
# is submitted once assembled labels exist.
#
# Usage:
#   nohup scripts/revision/qm7x/watch_qm7x_campaign.sh >> logs/qm7x_watchdog.log 2>&1 &
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

INTERVAL=${INTERVAL:-1800}
MAX_ROUNDS=${MAX_ROUNDS:-500}
CONCURRENT=${CONCURRENT:-250}
ACCOUNT="${SLURM_ACCOUNT:-torch_pr_283_chemistry}"
QOS="${SLURM_QOS:-cpu48}"
SBATCH_ORCA="${SBATCH_ORCA:-scripts/revision/submit_qm7x_orca.sbatch}"
SHARD_DIR="${SHARD_DIR:-datasets/revision/qm7x/shards/train}"
OUTDIR="${OUTDIR:-results/revision/qm7x_orca/train}"
DENSITY_PARENT="${DENSITY_PARENT:-results/revision/qm7x}"
ENERGY_JOB_NAME="${ENERGY_JOB_NAME:-tre-qm7x-energy}"

mkdir -p logs results/revision/qm7x_orca/train "$DENSITY_PARENT"

if [[ ! -f "${DENSITY_PARENT}/.density_training_done" ]]; then
  density_run="paper/models/qm7x/2024-04-22_bjOUNzrR"
  echo "$density_run" > "${DENSITY_PARENT}/.density_run_dir"
  touch "${DENSITY_PARENT}/.density_training_done"
  echo "[$(date -Is)] recorded pre-trained density at ${density_run}"
fi

submit_orca_jobs() {
  local n_missing
  n_missing=$(./.venv/bin/python scripts/revision/qm7x/qm7x_orca_status.py \
    --split train --json | ./.venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['n_missing'])")

  if [[ "$n_missing" -eq 0 ]]; then
    return 0
  fi

  local running
  running=$(squeue -h -u "$USER" -n qm7x-orca-train -o "%i" 2>/dev/null | wc -l)
  if [[ "$running" -gt 0 ]]; then
    echo "[$(date -Is)] qm7x-orca-train already queued/running (${running} entries)"
    return 0
  fi

  # Re-submit whole 2500-shard windows; worker --resume skips finished frames.
  local offset chunk_end array_end
  for offset in 10000 12500 15000 17500 0 2500 5000 7500; do
    chunk_end=$((offset + 2499))
    local left
    left=$(./.venv/bin/python -c "
from pathlib import Path
import sys
root = Path('results/revision/qm7x_orca/train')
offset, end = int(sys.argv[1]), int(sys.argv[2])
missing = sum(1 for i in range(offset, end + 1) if not (root / f'shard_{i:04d}' / 'results.npy').is_file())
print(missing)
" "$offset" "$chunk_end")
    if [[ "$left" -eq 0 ]]; then
      continue
    fi
    if [[ "$offset" -ge 10000 ]]; then
      array_end=2499
    else
      array_end=$((chunk_end - offset))
    fi
    echo "[$(date -Is)] submitting train offset=${offset} array=0-${array_end} (${left} missing)"
    sbatch --job-name=qm7x-orca-train \
      --account="$ACCOUNT" --qos="$QOS" \
      --cpus-per-task=8 --mem=16G --time=02:00:00 \
      --array="0-${array_end}%${CONCURRENT}" \
      --output="${OUTDIR}/slurm-%x-%A_%a.out" \
      --export=ALL,SPLIT=train,SHARD_DIR="${SHARD_DIR}",OUTDIR="${OUTDIR}",ARRAY_OFFSET="${offset}" \
      "$SBATCH_ORCA" || true
    return 0
  done
}

maybe_assemble() {
  local split="$1"
  local dens="datasets/qm7x_${split}_dft_augccpvdz.npy"
  if [[ -f "$dens" ]]; then
    return 0
  fi
  local n_missing
  n_missing=$(./.venv/bin/python scripts/revision/qm7x/qm7x_orca_status.py \
    --split "$split" --json | ./.venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['n_missing'])")
  if [[ "$n_missing" -ne 0 ]]; then
    return 0
  fi
  echo "[$(date -Is)] assembling ${split}"
  ./.venv/bin/python scripts/revision/qm7x/qm7x_assemble_npy.py \
    --results-dir "results/revision/qm7x_orca/${split}" \
    --split "$split" \
    --shard-dir "datasets/revision/qm7x/shards/${split}" \
    --min-complete 0.999
}

maybe_submit_energy() {
  local save_dir="results/revision/qm7x_energy"
  if [[ -e "${save_dir}/.training_done" ]]; then
    return 0
  fi
  for f in \
    datasets/qm7x_train_dft_augccpvdz.npy \
    datasets/qm7x_train_dft_augccpvdz_orca_base.npy \
    datasets/qm7x_valid_dft_augccpvdz.npy \
    datasets/qm7x_valid_dft_augccpvdz_orca_base.npy; do
    if [[ ! -f "$f" ]]; then
      echo "[$(date -Is)] energy waiting on ${f}"
      return 0
    fi
  done

  local queued
  queued=$(squeue -h -u "$USER" -n "$ENERGY_JOB_NAME" -o "%i" 2>/dev/null | wc -l)
  if [[ "$queued" -gt 0 ]]; then
    return 0
  fi
  echo "[$(date -Is)] submitting ${ENERGY_JOB_NAME}"
  sbatch --job-name="$ENERGY_JOB_NAME" scripts/revision/submit_qm7x_energy.sbatch || true
}

for ((round = 1; round <= MAX_ROUNDS; round++)); do
  echo "=========== [$(date -Is)] qm7x round ${round} ==========="
  ./.venv/bin/python scripts/revision/qm7x/qm7x_orca_status.py --split train || true
  ./.venv/bin/python scripts/revision/qm7x/qm7x_orca_status.py --split valid || true

  submit_orca_jobs
  maybe_assemble valid
  maybe_assemble train
  maybe_submit_energy

  train_left=$(./.venv/bin/python scripts/revision/qm7x/qm7x_orca_status.py \
    --split train --json | ./.venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['n_missing'])")
  if [[ "$train_left" -eq 0 ]] \
    && [[ -f datasets/qm7x_train_dft_augccpvdz.npy ]] \
    && [[ -e results/revision/qm7x_energy/.training_done ]]; then
    echo "[$(date -Is)] QM7x campaign complete"
    break
  fi

  sleep "$INTERVAL"
done
