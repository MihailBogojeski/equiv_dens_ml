#!/usr/bin/env bash
# Turn the ORCA shard output for one level of theory into DenSNet npy pairs.
#
# Separate from training because assembly is cheap, needs no GPU, and is worth
# running repeatedly during a campaign to watch the size histogram fill in.
#
# The size-OOD split is assembled from both tiers at once. It was deliberately
# cut into a small and a large array so the 8- and 10-water frames would not sit
# behind 24-water frames in a 24 h queue, but it is one split scientifically, and
# assembling only one tier would silently narrow exactly the size range the
# extrapolation claim rests on. build_shards_from_xyz.py numbers frames before
# it filters by atom count, so the two tiers share one index space and merge
# cleanly.
#
# Usage:  scripts/revision/assemble_water_labels.sh [THEORY] [MIN_COMPLETE]

set -euo pipefail
cd "$(dirname "$0")/../.."

THEORY=${1:-wb97mv_def2tzvpd}
# Deliberately not 1.0: a campaign is judged on whether enough of every cluster
# size arrived, and holding out for the last stragglers would block training on
# a dataset that is already complete enough. Under this, the size range really
# is truncated and the assembler says so.
MIN_COMPLETE=${2:-0.6}

OUT_ROOT=results/revision/water_orca
SHARD_ROOT=datasets/revision/shards
LABELS=datasets/revision/water_labels
MALON_LABELS=datasets/revision/malonaldehyde_labels

assemble() {
  local out_dir=$1 out_name=$2
  shift 2
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
  mkdir -p "$out_dir"
  echo "== ${out_name} (${*})"
  ./.venv/bin/python scripts/revision/qm7x_assemble_npy.py \
    --results-dir "${results[@]}" \
    --shard-dir "${shards[@]}" \
    --split "$out_name" \
    --min-complete "$MIN_COMPLETE" \
    --dens-out "${out_dir}/${out_name}.npy" \
    --base-out "${out_dir}/${out_name}_base.npy"
}

echo "=== assembling ${THEORY} (min-complete ${MIN_COMPLETE}) ==="
assemble "$LABELS" "water_train_${THEORY}" water_train_small || true
assemble "$LABELS" "water_val_${THEORY}" water_val_small || true
assemble "$LABELS" "water_id_test_${THEORY}" water_id_test_small || true
assemble "$LABELS" "water_ood_size_${THEORY}" water_ood_size_small water_ood_size_large || true
assemble "$LABELS" "water_ood_order_${THEORY}" water_ood_order || true
assemble "$LABELS" "water_ood_density_${THEORY}" water_ood_density || true
assemble "$MALON_LABELS" "malonaldehyde_train_${THEORY}" malonaldehyde_train || true
assemble "$MALON_LABELS" "malonaldehyde_val_${THEORY}" malonaldehyde_val || true
assemble "$MALON_LABELS" "malonaldehyde_ood_proton_transfer_${THEORY}" malonaldehyde_ood_proton_transfer || true
