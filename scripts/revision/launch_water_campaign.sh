#!/bin/bash
# Build shards and submit the whole labelling campaign in one go.
#
# Frames are split into a small and a large tier because a 24-water cluster in
# def2-TZVPD costs orders of magnitude more than a water trimer, and giving both
# the same walltime would either waste most of the small tasks' allocation or
# kill every large one. The split is by atom count, and each tier gets its own
# core count and walltime.
#
# Everything is idempotent: re-running rebuilds the shard lists and re-submits,
# finished shards exit immediately, and unfinished ones resume mid-shard.
#
# Usage:  scripts/revision/launch_water_campaign.sh [THEORY] [--dry-run]

set -euo pipefail
cd "$(dirname "$0")/../.."

THEORY=${1:-wb97mv_def2tzvpd}
DRY=${2:-}
SHARD_ROOT=datasets/revision/shards
OUT_ROOT=results/revision/water_orca
CONCURRENT_SMALL=${CONCURRENT_SMALL:-200}
CONCURRENT_LARGE=${CONCURRENT_LARGE:-60}
LARGE_ATOM_CUTOFF=${LARGE_ATOM_CUTOFF:-33}

WATER=datasets/revision/water_clusters
OOD=datasets/revision/water_ood
MALON=datasets/revision/malonaldehyde

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
  echo "  ${split}: array 0-$((n - 1))%${concurrency} -> ${outdir}"
  if [[ "$DRY" == "--dry-run" ]]; then
    return
  fi
  SHARD_DIR="$shard_dir" OUTDIR="$outdir" THEORY="$THEORY" \
    sbatch --array="0-$((n - 1))%${concurrency}" "$sbatch_file"
}

build() {
  local split=$1 xyz=$2
  shift 2
  if [[ ! -f "$xyz" ]]; then
    echo "  ${split}: ${xyz} not present yet, skipping"
    return 1
  fi
  ./.venv/bin/python scripts/revision/build_shards_from_xyz.py \
    --xyz "$xyz" --split "$split" --theory "$THEORY" --outdir "$SHARD_ROOT" "$@"
}

echo "=== building shards for ${THEORY} ==="

# Small tier: everything up to and including a 10-water cluster, plus the
# 9-atom malonaldehyde frames, at 8 cores in a 4 h window.
for split in train val id_test ood_size; do
  build "water_${split}_small" "${WATER}/${split}.xyz" \
    --max-atoms $((LARGE_ATOM_CUTOFF - 1)) --walltime-s 14400 --max-frames-per-shard 40 || true
done
for split in train val ood_proton_transfer; do
  build "malonaldehyde_${split}" "${MALON}/${split}.xyz" \
    --walltime-s 14400 --max-frames-per-shard 100 || true
done

# Large tier: 11 waters and up, at 16 cores in a 24 h window.
build water_ood_size_large "${WATER}/ood_size.xyz" \
  --min-atoms "$LARGE_ATOM_CUTOFF" --walltime-s 86400 --max-frames-per-shard 8 || true
build water_ood_order "${OOD}/ood_order.xyz" \
  --walltime-s 86400 --max-frames-per-shard 4 || true
build water_ood_density "${OOD}/ood_density.xyz" \
  --walltime-s 86400 --max-frames-per-shard 4 || true

echo
echo "=== submitting ==="
for split in water_train_small water_val_small water_id_test_small water_ood_size_small \
             malonaldehyde_train malonaldehyde_val malonaldehyde_ood_proton_transfer; do
  submit_tier "$split" scripts/revision/submit_water_orca.sbatch "$CONCURRENT_SMALL"
done
for split in water_ood_size_large water_ood_order water_ood_density; do
  submit_tier "$split" scripts/revision/submit_water_orca_large.sbatch "$CONCURRENT_LARGE"
done

echo
echo "monitor with: squeue -u $USER -n water-orca,water-orca-lg"
echo "resubmit with: scripts/revision/watch_and_resubmit.sh <shard_dir> <outdir> ${THEORY}"
