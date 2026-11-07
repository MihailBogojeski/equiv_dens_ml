#!/bin/bash
# Staged DFT on the MD-expanded geometry trees.
#
# Wave 1: PBE-D4 on every Phase A (or Phase B) frame.
# Wave 2: ωB97M-V on a stratified 10% of those frames.
# Wave 3: same pattern on the new-motif tree after it has been MD-thickened.
#
# Geometry generation is local. This script only *builds shards* by default
# (--build-only) so it cannot accidentally queue ~300k hybrid single points.
# Pass --submit to hand the wave to the existing watchdog.
#
# Usage:
#   scripts/revision/water/launch_md_expansion_waves.sh wave1
#   scripts/revision/water/launch_md_expansion_waves.sh wave2
#   scripts/revision/water/launch_md_expansion_waves.sh wave3
#   scripts/revision/water/launch_md_expansion_waves.sh wave1 --submit

set -euo pipefail
cd "$(dirname "$0")/../../.."

WAVE=${1:-wave1}
MODE=${2:---build-only}

launch_tree() {
  local theory=$1 water=$2 ood=$3 malon=$4 shard=$5 out=$6
  echo "=== ${WAVE}: theory=${theory} water=${water} ==="
  WATER="$water" OOD="$ood" MALON="$malon" \
    SHARD_ROOT="$shard" OUT_ROOT="$out" \
    scripts/revision/water/launch_water_campaign.sh "$theory" "$MODE"
}

case "$WAVE" in
  wave1)
    # Phase A 10×, PBE-D4 on every frame.
    launch_tree pbe_d4_avdz \
      datasets/revision/water_clusters_md10x \
      datasets/revision/water_ood_md10x \
      datasets/revision/malonaldehyde_md10x \
      datasets/revision/shards_md10x \
      results/revision/water_orca_md10x
    ;;
  wave2)
    # Stratified 10% of Phase A, then ωB97M-V only.
    ./.venv/bin/python scripts/revision/water/select_stratified_subset.py \
      --src-water datasets/revision/water_clusters_md10x \
      --src-ood datasets/revision/water_ood_md10x \
      --src-malon datasets/revision/malonaldehyde_md10x \
      --out-water datasets/revision/water_clusters_md10x_hybrid10 \
      --out-ood datasets/revision/water_ood_md10x_hybrid10 \
      --out-malon datasets/revision/malonaldehyde_md10x_hybrid10 \
      --frac 0.1
    launch_tree wb97mv_def2tzvpd \
      datasets/revision/water_clusters_md10x_hybrid10 \
      datasets/revision/water_ood_md10x_hybrid10 \
      datasets/revision/malonaldehyde_md10x_hybrid10 \
      datasets/revision/shards_md10x_hybrid10 \
      results/revision/water_orca_md10x_hybrid10
    ;;
  wave3)
    # Phase B parents after MD thickening: PBE all, then the operator runs wave2
    # pointing at these trees if a hybrid subset is wanted.
    launch_tree pbe_d4_avdz \
      datasets/revision/water_clusters_newmotifs_md10x \
      datasets/revision/water_ood_newmotifs_md10x \
      datasets/revision/malonaldehyde_newmotifs_md10x \
      datasets/revision/shards_newmotifs_md10x \
      results/revision/water_orca_newmotifs_md10x
    ;;
  wave_md40x_pbe)
    exec scripts/revision/water/launch_md40x_waves.sh pbe "$MODE"
    ;;
  wave_md40x_hybrid)
    exec scripts/revision/water/launch_md40x_waves.sh hybrid "$MODE"
    ;;
  *)
    echo "unknown wave: ${WAVE} (expected wave1|wave2|wave3|wave_md40x_pbe|wave_md40x_hybrid)" >&2
    exit 1
    ;;
esac

echo
echo "shards only unless MODE=--submit. Monitor with:"
echo "  ./.venv/bin/python scripts/revision/water/campaign_status.py --root datasets/revision/shards_md10x"
