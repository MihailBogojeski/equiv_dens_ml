#!/bin/bash
# Build shards and submit ORCA labels for one H-bond / new-system campaign tree.
#
# Usage:
#   SYSTEM=meoh_water THEORY=wb97mv_def2tzvpd \
#     scripts/revision/hbonds/launch_hbond_campaign.sh
#   SYSTEM=meoh_water THEORY=pbe_d4_avdz \
#     scripts/revision/hbonds/launch_hbond_campaign.sh --build-only

set -euo pipefail
cd "$(dirname "$0")/../../.."

SYSTEM=${SYSTEM:?set SYSTEM, e.g. meoh_water}
THEORY=${THEORY:-wb97mv_def2tzvpd}
DRY=${1:-}
DATA_ROOT=${DATA_ROOT:-datasets/revision/${SYSTEM}}
SHARD_ROOT=${SHARD_ROOT:-datasets/revision/shards_${SYSTEM}}
OUT_ROOT=${OUT_ROOT:-results/revision/${SYSTEM}_orca}
LARGE_ATOM_CUTOFF=${LARGE_ATOM_CUTOFF:-33}
CONCURRENT_SMALL=${CONCURRENT_SMALL:-100}
CONCURRENT_LARGE=${CONCURRENT_LARGE:-40}
# PGG is 30 atoms, so it misses LARGE_ATOM_CUTOFF 33 and would otherwise be
# submitted on the 4 h small tier. Hybrid SCF there is ~1 h/frame and the
# array tasks time out, leaving stale claims. Crambin fragments are already
# ≥40 atoms; keep them on the 48 h script even if a rebuild lands in the
# non-_large split directory.
ALWAYS_LARGE_SYSTEMS=${ALWAYS_LARGE_SYSTEMS:-pgg,crambin_frags}

submit_tier() {
  local split=$1 sbatch_file=$2 concurrency=$3
  local shard_dir="${SHARD_ROOT}/${split}/${THEORY}"
  local outdir="${OUT_ROOT}/${split}/${THEORY}"
  local n=0
  if [[ -d "$shard_dir" ]]; then
    n=$(find "$shard_dir" -maxdepth 1 -name 'shard_*.json' | wc -l)
  fi
  if [[ "$n" -eq 0 ]]; then
    echo "  ${split}: no shards, skipping"
    return
  fi
  echo "  ${split}: ${n} shards -> ${outdir}"
  if [[ "$DRY" == "--dry-run" || "$DRY" == "--build-only" ]]; then
    return
  fi
  ONESHOT=1 SBATCH_FILE="$sbatch_file" CONCURRENT="$concurrency" \
    scripts/revision/water/watch_and_resubmit.sh "$shard_dir" "$outdir" "$THEORY"
}

build() {
  local split=$1 xyz=$2
  shift 2
  if [[ ! -f "$xyz" ]]; then
    echo "  ${split}: ${xyz} not present, skipping"
    return 1
  fi
  ./.venv/bin/python scripts/revision/lib/build_shards_from_xyz.py \
    --xyz "$xyz" --split "${SYSTEM}_${split}" --theory "$THEORY" --outdir "$SHARD_ROOT" "$@"
}

manifest_ood() {
  if [[ ! -f "${DATA_ROOT}/manifest.json" ]]; then
    echo "ood"
    return
  fi
  python3 - <<PY
import json
from pathlib import Path
m = json.loads(Path("${DATA_ROOT}/manifest.json").read_text())
print(m.get("ood_split", "ood"))
PY
}

OOD=$(manifest_ood)

echo "=== ${SYSTEM} shards for ${THEORY} ==="
if [[ ! -f "${DATA_ROOT}/manifest.json" ]]; then
  echo "  ${SYSTEM}: no manifest at ${DATA_ROOT}, skipping"
  exit 0
fi
for split in train val id_test; do
  build "$split" "${DATA_ROOT}/${split}.xyz" \
    --max-atoms $((LARGE_ATOM_CUTOFF - 1)) --walltime-s 14400 --max-frames-per-shard 40 || true
  build "${split}_large" "${DATA_ROOT}/${split}.xyz" \
    --min-atoms "$LARGE_ATOM_CUTOFF" --walltime-s 86400 --max-frames-per-shard 8 || true
done
build "$OOD" "${DATA_ROOT}/${OOD}.xyz" \
  --walltime-s 86400 --max-frames-per-shard 8 || true

small_sbatch=scripts/revision/submit_water_orca.sbatch
small_conc=$CONCURRENT_SMALL
if [[ ",${ALWAYS_LARGE_SYSTEMS}," == *",${SYSTEM},"* ]]; then
  small_sbatch=scripts/revision/submit_water_orca_large.sbatch
  small_conc=$CONCURRENT_LARGE
fi

if [[ "$DRY" != "--build-only" && "$DRY" != "--dry-run" ]]; then
  echo
  echo "=== submitting ${SYSTEM} ==="
  for split in train val id_test; do
    submit_tier "${SYSTEM}_${split}" "$small_sbatch" "$small_conc"
    submit_tier "${SYSTEM}_${split}_large" scripts/revision/submit_water_orca_large.sbatch "$CONCURRENT_LARGE"
  done
  submit_tier "${SYSTEM}_${OOD}" scripts/revision/submit_water_orca_large.sbatch "$CONCURRENT_LARGE"
fi

echo
echo "monitor: ./.venv/bin/python scripts/revision/campaign_status.py --root ${SHARD_ROOT}"
