#!/usr/bin/env bash
# Everything downstream of the labels, in the order it has to happen.
#
# Split into two halves by what it needs. The collective variables and the
# overlap report need only geometries, so they run now and are what establish
# that the OOD splits really are out of distribution -- the claim Reviewer 1
# challenged. The error curves need a trained model, so they are skipped with a
# message until one exists, and this script is safe to re-run as the campaign
# fills in.
#
# Usage:  scripts/revision/run_ood_analysis.sh [THEORY]

set -uo pipefail
cd "$(dirname "$0")/../.."

THEORY=${1:-wb97mv_def2tzvpd}
PY=./.venv/bin/python
CV=results/revision/cv
EVAL=results/revision/eval/${THEORY}
WATER=datasets/revision/water_clusters
OOD=datasets/revision/water_ood
MALON=datasets/revision/malonaldehyde
RUN_DIR=results/revision/water_wb97mv_001
MALON_RUN_DIR=results/revision/malonaldehyde_pt_001
LABELS=datasets/revision/water_labels
MALON_LABELS=datasets/revision/malonaldehyde_labels

mkdir -p "$CV" "$EVAL" results/revision/figures

echo "=== collective variables ==="
cv_water() {
  local label=$1 xyz=$2
  [[ -f "$xyz" ]] || { echo "  ${label}: no geometries"; return; }
  [[ -f "${CV}/${label}.json" ]] && { echo "  ${label}: already computed"; return; }
  $PY scripts/revision/water_collective_variables.py --xyz "$xyz" --label "$label" --out "${CV}/${label}.json"
}
cv_water water_train "${WATER}/train.xyz"
cv_water water_val "${WATER}/val.xyz"
cv_water water_id_test "${WATER}/id_test.xyz"
cv_water water_ood_size "${WATER}/ood_size.xyz"
cv_water ood_order "${OOD}/ood_order.xyz"
cv_water ood_density "${OOD}/ood_density.xyz"

for split in train val ood_proton_transfer; do
  out="${CV}/malon_${split}_pt.json"
  [[ -f "$out" ]] && continue
  $PY scripts/revision/malonaldehyde_collective_variables.py \
    --xyz "${MALON}/${split}.xyz" --manifest "${MALON}/manifest.json" \
    --label "malon_${split}" --out "$out"
done

echo
echo "=== train/test separation ==="
$PY scripts/revision/ood_overlap_report.py \
  --train "${CV}/water_train.json" \
  --test "${CV}/water_id_test.json" "${CV}/water_ood_size.json" \
         "${CV}/ood_order.json" "${CV}/ood_density.json" \
  --out results/revision/ood_overlap.json

# Malonaldehyde gets its own run: the water descriptors treat its two oxygens as
# a pair of waters and report nothing meaningful about a proton transfer.
$PY scripts/revision/ood_overlap_report.py \
  --train "${CV}/malon_train_pt.json" \
  --test "${CV}/malon_val_pt.json" "${CV}/malon_ood_proton_transfer_pt.json" \
  --cvs delta_pt,r_oo,co_alternation,r_donor_h,r_acceptor_h \
  --joint delta_pt,co_alternation \
  --out results/revision/ood_overlap_malonaldehyde.json

echo
echo "=== density errors ==="
if [[ ! -d "${RUN_DIR}" || ! -f "${RUN_DIR}/args.txt" ]]; then
  echo "  no trained water model at ${RUN_DIR} yet; stopping before the error curves"
  exit 0
fi

evaluate() {
  local label=$1 base=$2 dens=$3 run_dir=$4 elements=$5
  if [[ ! -f "$base" || ! -f "$dens" ]]; then
    echo "  ${label}: labels not assembled yet"
    return 1
  fi
  $PY scripts/revision/csh_evaluate.py \
    --args-file "${run_dir}/args.txt" --run-dir "$run_dir" \
    --np-dataset "$base" --dens-dataset "$dens" \
    --elements "$elements" --label "$label" --out "${EVAL}/${label}.json"
}

for split in water_id_test water_ood_size water_ood_order water_ood_density; do
  evaluate "$split" \
    "${LABELS}/${split}_${THEORY}_base.npy" "${LABELS}/${split}_${THEORY}.npy" \
    "$RUN_DIR" 1,8
done
if [[ -d "$MALON_RUN_DIR" ]]; then
  evaluate malonaldehyde_ood_proton_transfer \
    "${MALON_LABELS}/malonaldehyde_ood_proton_transfer_${THEORY}_base.npy" \
    "${MALON_LABELS}/malonaldehyde_ood_proton_transfer_${THEORY}.npy" \
    "$MALON_RUN_DIR" 1,6,8
fi

echo
echo "=== error versus distance from training ==="
$PY scripts/revision/error_vs_distance.py \
  --train-cv "${CV}/water_train.json" \
  --tier "id_test:${CV}/water_id_test.json:${EVAL}/water_id_test.json" \
  --tier "ood_size:${CV}/water_ood_size.json:${EVAL}/water_ood_size.json" \
  --tier "ood_order:${CV}/ood_order.json:${EVAL}/water_ood_order.json" \
  --tier "ood_density:${CV}/ood_density.json:${EVAL}/water_ood_density.json" \
  --physical-axis "malonaldehyde:${CV}/malon_ood_proton_transfer_pt.json:${EVAL}/malonaldehyde_ood_proton_transfer.json:abs_delta_pt" \
  --out "results/revision/error_vs_distance_${THEORY}.json" \
  --figure "results/revision/figures/error_vs_distance_${THEORY}.png"
