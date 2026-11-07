#!/bin/bash
# ORCA labels for extreme-mode and water distance-scale OOD geometries.
#
# Writes only sibling trees (shards_ood_geom / ood_geom_orca). Does not touch
# the finished 3k shards/ or water_orca/ trees, and does not submit tr-*
# training. The 12-mer is energy/forces only (NO_DF=1); everything else keeps
# the density fit.
#
# Usage:
#   scripts/revision/launch_ood_geometry_dft.sh --build-only
#   scripts/revision/launch_ood_geometry_dft.sh --submit
#   THEORY=pbe_d4_avdz scripts/revision/launch_ood_geometry_dft.sh --submit

set -euo pipefail
cd "$(dirname "$0")/../../.."

MODE=${1:---build-only}
THEORY=${THEORY:-}
SHARD_ROOT=${SHARD_ROOT:-datasets/revision/shards_ood_geom}
OUT_ROOT=${OUT_ROOT:-results/revision/ood_geom_orca}
LARGE_ATOM_CUTOFF=${LARGE_ATOM_CUTOFF:-33}
CONCURRENT_SMALL=${CONCURRENT_SMALL:-80}
CONCURRENT_LARGE=${CONCURRENT_LARGE:-60}
JOB_PREFIX=${JOB_PREFIX:-og}
PARTITION="${PARTITION:-cs}"

MODE_SYSTEMS=(
  water_n2
  water_n6
  ethanol
  ethanethiol
  resorcinol
  thiophene_2mer
  malonaldehyde
)

theories=()
if [[ -n "$THEORY" ]]; then
  theories=("$THEORY")
else
  theories=(pbe_d4_avdz wb97mv_def2tzvpd)
fi

submit_tier() {
  local split=$1 sbatch_file=$2 concurrency=$3
  local no_df=${4:-0}
  local partition=${5:-$PARTITION}
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
  if [[ "$MODE" == "--dry-run" || "$MODE" == "--build-only" ]]; then
    return
  fi
  ONESHOT=1 SBATCH_FILE="$sbatch_file" CONCURRENT="$concurrency" \
    JOB_PREFIX="$JOB_PREFIX" NO_DF="$no_df" PARTITION="$partition" \
    scripts/revision/water/watch_and_resubmit.sh "$shard_dir" "$outdir" "$THEORY"
}

build_split() {
  local split=$1 xyz=$2
  shift 2
  if [[ ! -f "$xyz" ]]; then
    echo "  ${split}: ${xyz} not present, skipping"
    return 1
  fi
  ./.venv/bin/python scripts/revision/lib/build_shards_from_xyz.py \
    --xyz "$xyz" --split "$split" --theory "$THEORY" --outdir "$SHARD_ROOT" "$@"
}

launch_theory() {
  echo "=== ood-geom DFT ${THEORY} -> ${SHARD_ROOT} ==="

  build_split ood_scale datasets/revision/ood_scale/ood_distance_scale.xyz \
    --max-atoms $((LARGE_ATOM_CUTOFF - 1)) --walltime-s 14400 --max-frames-per-shard 40 || true
  build_split ood_scale_large datasets/revision/ood_scale/ood_distance_scale.xyz \
    --min-atoms "$LARGE_ATOM_CUTOFF" --walltime-s 86400 --max-frames-per-shard 8 || true

  local name
  for name in "${MODE_SYSTEMS[@]}"; do
    build_split "ood_modes_${name}" \
      "datasets/revision/ood_modes/${name}/ood_modes.xyz" \
      --max-atoms $((LARGE_ATOM_CUTOFF - 1)) --walltime-s 14400 --max-frames-per-shard 40 || true
  done

  build_split ood_modes_thiophene_12mer \
    datasets/revision/ood_modes/thiophene_12mer/ood_modes.xyz \
    --min-atoms "$LARGE_ATOM_CUTOFF" --walltime-s 86400 --max-frames-per-shard 4 || true

  if [[ "$MODE" != "--submit" ]]; then
    return
  fi

  echo
  echo "=== submitting ${THEORY} ==="
  submit_tier ood_scale scripts/revision/submit_water_orca.sbatch "$CONCURRENT_SMALL"
  submit_tier ood_scale_large scripts/revision/submit_water_orca_large.sbatch \
    "$CONCURRENT_LARGE" 0 cs
  for name in "${MODE_SYSTEMS[@]}"; do
    submit_tier "ood_modes_${name}" scripts/revision/submit_water_orca.sbatch "$CONCURRENT_SMALL"
  done
  submit_tier ood_modes_thiophene_12mer \
    scripts/revision/submit_water_orca_large.sbatch "$CONCURRENT_LARGE" 1 cs
}

for THEORY in "${theories[@]}"; do
  launch_theory
done

echo
echo "monitor: ./.venv/bin/python scripts/revision/water/campaign_status.py --root ${SHARD_ROOT} --out-root ${OUT_ROOT}"
echo "3k trees were not used. No tr-* jobs submitted."
