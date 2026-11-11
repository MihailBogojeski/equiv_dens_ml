#!/bin/bash
# After md40x thicken (and Phase B if it produced parents): merge trees and
# submit the PBE-D4 wave. Does not start 3k training. Does not overwrite 3k paths.

set -euo pipefail
cd "$(dirname "$0")/../../.."

WATER_A=datasets/revision/water_clusters_md40x
OOD_A=datasets/revision/water_ood_md40x
MALON_A=datasets/revision/malonaldehyde_md40x
WATER_B=datasets/revision/water_clusters_newmotifs_md40x
OOD_B=datasets/revision/water_ood_newmotifs_md40x
MALON_B=datasets/revision/malonaldehyde_newmotifs_md40x
WATER_R=datasets/revision/water_clusters_rackers
OOD_R=datasets/revision/water_ood_rackers
DEST_W=datasets/revision/water_clusters_md40x_merged
DEST_O=datasets/revision/water_ood_md40x_merged
DEST_M=datasets/revision/malonaldehyde_md40x_merged

if [[ ! -f "${WATER_A}/train.xyz" ]]; then
  echo "missing ${WATER_A}/train.xyz; thicken has not finished" >&2
  exit 1
fi

if [[ -d datasets/revision/water_clusters_newmotifs && ! -f "${WATER_B}/train.xyz" ]]; then
  echo "thickening Phase B parents to md40x"
  WORKERS="${WORKERS:-16}" scripts/revision/water/run_geometry_expansion.sh phase_b_md40x
fi

RACKERS_DIR=datasets/revision/external/rackers_water
if [[ ! -f "${RACKERS_DIR}/raw.xyz" && -n "$(compgen -G "${RACKERS_DIR}/w*_energy_force.pkl" || true)" ]]; then
  echo "extracting Rackers XYZ from Zenodo pickles"
  ./.venv/bin/python scripts/revision/water/extract_rackers_pkls.py \
    --src "$RACKERS_DIR" --xyz-out "${RACKERS_DIR}/raw.xyz"
fi
if [[ -f "${RACKERS_DIR}/raw.xyz" && ! -f "${WATER_R}/manifest.json" ]]; then
  echo "importing Rackers parents"
  ./.venv/bin/python scripts/revision/water/import_rackers_parents.py \
    --xyz "${RACKERS_DIR}/raw.xyz" \
    --existing-xyz datasets/revision/water_clusters/train.xyz \
                   datasets/revision/water_clusters/ood_size.xyz \
                   datasets/revision/water_ood/ood_order.xyz
fi

water_src=("$WATER_A")
ood_src=("$OOD_A")
malon_src=("$MALON_A")
[[ -f "${WATER_B}/train.xyz" ]] && water_src+=("$WATER_B")
[[ -f "${OOD_B}/ood_order.xyz" || -f "${OOD_B}/ood_density.xyz" ]] && ood_src+=("$OOD_B")
[[ -f "${MALON_B}/train.xyz" ]] && malon_src+=("$MALON_B")
[[ -f "${WATER_R}/train.xyz" ]] && water_src+=("$WATER_R")
[[ -f "${OOD_R}/ood_rackers.xyz" ]] && ood_src+=("$OOD_R")

./.venv/bin/python scripts/revision/water/merge_xyz_trees.py \
  --src "${water_src[@]}" --dest "$DEST_W" \
  --splits train,val,id_test,ood_size --rmsd-cutoff 0.12
./.venv/bin/python scripts/revision/water/merge_xyz_trees.py \
  --src "${ood_src[@]}" --dest "$DEST_O" \
  --splits ood_order,ood_density,ood_rackers --rmsd-cutoff 0.20
./.venv/bin/python scripts/revision/water/merge_xyz_trees.py \
  --src "${malon_src[@]}" --dest "$DEST_M" \
  --splits train,val,ood_proton_transfer --rmsd-cutoff 0.08

echo "=== PBE shard build + submit ==="
WATER="$DEST_W" OOD="$DEST_O" MALON="$DEST_M" \
  SHARD_ROOT=datasets/revision/shards_md40x \
  OUT_ROOT=results/revision/water_orca_md40x \
  scripts/revision/water/launch_md40x_waves.sh pbe --submit

echo "PBE wave submitted. Monitor with:"
echo "  ./.venv/bin/python scripts/revision/water/campaign_status.py --root datasets/revision/shards_md40x --out-root results/revision/water_orca_md40x"
