#!/bin/bash
# CPU-only ORCA array for QM7-X DensNet labels (PBE0/aug-cc-pVDZ EnGrad).
# One Slurm task = one frame. Short walltime keeps tasks on cpu_short/cs and
# lets the watchdog re-submit safely (--resume skips finished shards).
#SBATCH --job-name=qm7x-orca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --output=results/revision/qm7x_orca/slurm-%x-%A_%a.out
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p results/revision/qm7x_orca

module load orca/6.1.1
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

SPLIT="${SPLIT:-smoke}"
SHARD_DIR="${SHARD_DIR:-${ROOT}/datasets/revision/qm7x/shards/${SPLIT}}"
OUTDIR="${OUTDIR:-${ROOT}/results/revision/qm7x_orca/${SPLIT}}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
TASK_ID=$((SLURM_ARRAY_TASK_ID + ARRAY_OFFSET))
SHARD="$(printf '%s/shard_%04d.json' "${SHARD_DIR}" "${TASK_ID}")"

if [[ ! -f "$SHARD" ]]; then
  echo "missing shard ${SHARD}" >&2
  exit 1
fi

export OMP_NUM_THREADS=1
export ORCA_BIN="${ORCA_BIN:-orca}"
export ORCA2JSON_BIN="${ORCA2JSON_BIN:-orca_2json}"
unset LD_PRELOAD

python scripts/revision/lib/qm7x_orca_worker.py \
  --shard "$SHARD" \
  --outdir "$OUTDIR" \
  --nprocs "${SLURM_CPUS_PER_TASK:-8}" \
  --resume
