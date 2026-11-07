#!/bin/bash
# Dual-theory DFT on the md40x geometry trees (PBE all, then hybrid all).
#
# Does not touch the finished 3k shards/ or water_orca/ trees.
# Default is --build-only. Pass --submit to hand the wave to the watchdog.
#
# Usage:
#   scripts/revision/water/launch_md40x_waves.sh pbe
#   scripts/revision/water/launch_md40x_waves.sh hybrid
#   scripts/revision/water/launch_md40x_waves.sh pbe --submit

set -euo pipefail
cd "$(dirname "$0")/../../.."

WAVE=${1:-pbe}
MODE=${2:---build-only}

WATER=${WATER:-datasets/revision/water_clusters_md40x}
OOD=${OOD:-datasets/revision/water_ood_md40x}
MALON=${MALON:-datasets/revision/malonaldehyde_md40x}
SHARD_ROOT=${SHARD_ROOT:-datasets/revision/shards_md40x}
OUT_ROOT=${OUT_ROOT:-results/revision/water_orca_md40x}

launch_tree() {
  local theory=$1
  echo "=== md40x ${WAVE}: theory=${theory} water=${WATER} ==="
  WATER="$WATER" OOD="$OOD" MALON="$MALON" \
    SHARD_ROOT="$SHARD_ROOT" OUT_ROOT="$OUT_ROOT" \
    scripts/revision/water/launch_water_campaign.sh "$theory" "$MODE"
}

case "$WAVE" in
  pbe|wave_md40x_pbe)
    launch_tree pbe_d4_avdz
    ;;
  hybrid|wave_md40x_hybrid)
    # Same XYZ as Wave A, not a 10% subset.
    launch_tree wb97mv_def2tzvpd
    ;;
  *)
    echo "unknown wave: ${WAVE} (expected pbe|hybrid)" >&2
    exit 1
    ;;
esac

echo
echo "shards only unless MODE=--submit. Monitor with:"
echo "  ./.venv/bin/python scripts/revision/water/campaign_status.py --root ${SHARD_ROOT}"
echo "assemble with REQUIRE_DIPOLE=1 and md40x OUT_ROOT/LABELS."
