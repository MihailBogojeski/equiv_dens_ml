#!/usr/bin/env bash
# Merge ORCA shard outputs for one new-system campaign into DenSNet npy pairs.
#
# Small and large atom tiers share one source_index space (see build_shards_from_xyz.py),
# so train/val/id splits merge both tiers when present. Run during the campaign with
# --min-complete below 1.0 to watch labels accumulate.
#
# Usage:
#   scripts/revision/hbonds/assemble_hbond_labels.sh meoh_water [THEORY] [MIN_COMPLETE]
#   scripts/revision/hbonds/assemble_hbond_labels.sh --all [THEORY] [MIN_COMPLETE]

set -euo pipefail
cd "$(dirname "$0")/../../.."

SYSTEMS=(meoh_water acac paracetamol nma_dimer pgg crambin_frags)

if [[ "${1:-}" == "--all" ]]; then
  THEORY=${2:-wb97mv_def2tzvpd}
  MIN_COMPLETE=${3:-0.6}
  for system in "${SYSTEMS[@]}"; do
    echo
    "$0" "$system" "$THEORY" "$MIN_COMPLETE" || true
  done
  exit 0
fi

SYSTEM=${1:?usage: assemble_hbond_labels.sh SYSTEM [THEORY] [MIN_COMPLETE]}
THEORY=${2:-wb97mv_def2tzvpd}
MIN_COMPLETE=${3:-0.6}

OUT_ROOT="results/revision/${SYSTEM}_orca"
SHARD_ROOT="datasets/revision/shards_${SYSTEM}"
LABELS="datasets/revision/${SYSTEM}_labels"
DATA_ROOT="datasets/revision/${SYSTEM}"

OOD=$(python3 - <<PY
import json
from pathlib import Path
m = json.loads(Path("${DATA_ROOT}/manifest.json").read_text())
print(m.get("ood_split", "ood"))
PY
)

assemble() {
  local out_name=$1
  shift
  local results=() shards=()
  for split in "$@"; do
    local rdir="${OUT_ROOT}/${split}/${THEORY}"
    [[ -d "$rdir" ]] || continue
    results+=("$rdir")
    shards+=("${SHARD_ROOT}/${split}/${THEORY}")
  done
  if [[ ${#results[@]} -eq 0 ]]; then
    echo "  ${out_name}: no results yet, skipping"
    return 1
  fi
  mkdir -p "$LABELS"
  echo "== ${SYSTEM} ${out_name} (${*})"
  ./.venv/bin/python scripts/revision/qm7x/qm7x_assemble_npy.py \
    --results-dir "${results[@]}" \
    --shard-dir "${shards[@]}" \
    --split "$out_name" \
    --min-complete "$MIN_COMPLETE" \
    --dens-out "${LABELS}/${out_name}.npy" \
    --base-out "${LABELS}/${out_name}_base.npy"
}

echo "=== assembling ${SYSTEM} @ ${THEORY} (min-complete ${MIN_COMPLETE}) ==="
assemble "${SYSTEM}_train_${THEORY}" "${SYSTEM}_train" "${SYSTEM}_train_large" || true
assemble "${SYSTEM}_val_${THEORY}" "${SYSTEM}_val" "${SYSTEM}_val_large" || true
assemble "${SYSTEM}_id_test_${THEORY}" "${SYSTEM}_id_test" "${SYSTEM}_id_test_large" || true
assemble "${SYSTEM}_${OOD}_${THEORY}" "${SYSTEM}_${OOD}" || true
