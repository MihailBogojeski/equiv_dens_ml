#!/usr/bin/env bash
# Build shards and submit QM7-X ORCA CPU arrays.
# Usage: bash scripts/revision/submit_qm7x_orca.sh {download|smoke|train|val|assemble}
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ACCOUNT="${SLURM_ACCOUNT:-torch_pr_283_chemistry}"
PARTITION="${SLURM_PARTITION:-cs}"
QOS="${SLURM_QOS:-cpu48}"
CPUS="${SLURM_CPUS:-8}"
MEM="${SLURM_MEM:-32G}"
TIME="${SLURM_TIME:-24:00:00}"
CONCURRENT="${SLURM_CONCURRENT:-50}"
SBATCH_FILE="${ROOT}/scripts/revision/submit_qm7x_orca.sbatch"

submit_split() {
  local split="$1"
  local shard_dir="${ROOT}/datasets/revision/qm7x/shards/${split}"
  local outdir="${ROOT}/results/revision/qm7x_orca/${split}"
  local manifest="${shard_dir}/manifest.json"
  if [[ ! -f "$manifest" ]]; then
    echo "missing ${manifest}; build shards first" >&2
    exit 1
  fi
  local n_shards
  n_shards="$(python -c "import json; print(json.load(open('${manifest}'))['n_shards'])")"
  if [[ "$n_shards" -lt 1 ]]; then
    echo "no shards in ${shard_dir}" >&2
    exit 1
  fi
  local last=$((n_shards - 1))
  mkdir -p "$outdir"
  echo "Submitting ${split}: array 0-${last}%${CONCURRENT} on ${PARTITION}/${QOS}"
  sbatch --job-name="qm7x-orca-${split}" \
    --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
    --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
    --array="0-${last}%${CONCURRENT}" \
    --output="${outdir}/slurm-%x-%A_%a.out" \
    --export=ALL,SPLIT="${split}",SHARD_DIR="${shard_dir}",OUTDIR="${outdir}" \
    "$SBATCH_FILE"
}

cmd="${1:-smoke}"
case "$cmd" in
  download)
    bash "${ROOT}/scripts/revision/download_qm7x.sh"
    ;;
  smoke)
    python "${ROOT}/scripts/revision/qm7x_build_shards.py" \
      --split smoke --source npy --max-frames 100 --frames-per-shard 10
    submit_split smoke
    ;;
  train)
    python "${ROOT}/scripts/revision/qm7x_build_shards.py" \
      --split train --source npy --frames-per-shard 20
    submit_split train
    ;;
  val|valid)
    python "${ROOT}/scripts/revision/qm7x_build_shards.py" \
      --split valid --source npy --frames-per-shard 20
    submit_split valid
    ;;
  assemble)
    split="${2:-smoke}"
    python "${ROOT}/scripts/revision/qm7x_assemble_npy.py" \
      --results-dir "${ROOT}/results/revision/qm7x_orca/${split}" \
      --split "$split"
    ;;
  *)
    echo "usage: $0 {download|smoke|train|val|assemble [split]}" >&2
    exit 2
    ;;
esac
