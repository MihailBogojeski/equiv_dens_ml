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
MEM="${SLURM_MEM:-16G}"
TIME="${SLURM_TIME:-04:00:00}"
CONCURRENT="${SLURM_CONCURRENT:-250}"
# cpu48 MaxSubmitPU is 3000; Greene MaxArraySize is 10000. Stay under QOS.
MAX_ARRAY="${SLURM_MAX_ARRAY:-2500}"
SBATCH_FILE="${ROOT}/scripts/revision/submit_qm7x_orca.sbatch"

sbatch_range() {
  local split="$1"
  local range="$2"
  local shard_dir="$3"
  local outdir="$4"
  local wait_flag="${5:-}"
  sbatch ${wait_flag} --job-name="qm7x-orca-${split}" \
    --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS" \
    --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
    --array="${range}%${CONCURRENT}" \
    --output="${outdir}/slurm-%x-%A_%a.out" \
    --export=ALL,SPLIT="${split}",SHARD_DIR="${shard_dir}",OUTDIR="${outdir}" \
    "$SBATCH_FILE"
}

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
  mkdir -p "$outdir"
  local ranges
  ranges="$(python -c "import sys; sys.path.insert(0,'${ROOT}/scripts/revision'); from qm7x_build_shards import slurm_array_ranges; print(' '.join(f'{a}-{b}' for a,b in slurm_array_ranges(${n_shards}, ${MAX_ARRAY})))")"
  echo "${split}: ${n_shards} one-frame tasks, ranges ${ranges}, throttle %${CONCURRENT}"

  local n_ranges
  n_ranges="$(echo "$ranges" | wc -w)"
  if [[ "$n_ranges" -eq 1 ]]; then
    sbatch_range "$split" "$ranges" "$shard_dir" "$outdir"
    return
  fi

  local runner="${outdir}/run_chunks.sh"
  cat > "$runner" <<EOF
#!/bin/bash
set -euo pipefail
cd "${ROOT}"
for range in ${ranges}; do
  echo "\$(date -Is) ${split} chunk \${range}%${CONCURRENT}"
  sbatch --wait --job-name="qm7x-orca-${split}" \\
    --account="${ACCOUNT}" --partition="${PARTITION}" --qos="${QOS}" \\
    --cpus-per-task="${CPUS}" --mem="${MEM}" --time="${TIME}" \\
    --array="\${range}%${CONCURRENT}" \\
    --output="${outdir}/slurm-%x-%A_%a.out" \\
    --export=ALL,SPLIT="${split}",SHARD_DIR="${shard_dir}",OUTDIR="${outdir}" \\
    "${SBATCH_FILE}"
done
echo "\$(date -Is) ${split} all chunks done"
EOF
  chmod +x "$runner"
  local session="qm7x-orca-${split}"
  if tmux -f /exec-daemon/tmux.portal.conf has-session -t "=${session}" 2>/dev/null; then
    echo "tmux session ${session} already running"
    return
  fi
  tmux -f /exec-daemon/tmux.portal.conf new-session -d -s "$session" -c "$ROOT" -- bash "$runner"
  echo "started tmux ${session} -> ${runner}"
}

cmd="${1:-smoke}"
case "$cmd" in
  download)
    bash "${ROOT}/scripts/revision/download_qm7x.sh"
    ;;
  smoke)
    python "${ROOT}/scripts/revision/qm7x_build_shards.py" \
      --split smoke --source npy --max-frames 100 --frames-per-shard 1
    submit_split smoke
    ;;
  train)
    python "${ROOT}/scripts/revision/qm7x_build_shards.py" \
      --split train --source npy --frames-per-shard 1
    submit_split train
    ;;
  val|valid)
    python "${ROOT}/scripts/revision/qm7x_build_shards.py" \
      --split valid --source npy --frames-per-shard 1
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
