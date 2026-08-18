#!/usr/bin/env bash
# CPU PBE+D4 (+ optional PBE0) labels. Resume-safe. Do not use node GPUs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

run_split() {
  local traj="$1"
  local prefix="$2"
  local outdir="$3"
  local xc="$4"
  echo "=== DFT ${xc} ${prefix} ==="
  "${PY}" scripts/revision/generate_dft_labels.py \
    --trajectory "${traj}" \
    --output-dir "${outdir}" \
    --output-prefix "${prefix}" \
    --xc "${xc}" --d4 --df --no-gpu --resume --save-interval 5
}

case "${1:-pbe}" in
  pbe-train)
    run_split datasets/revision/water_clusters/train.xyz water_train datasets/revision/water_clusters pbe
    ;;
  pbe-rest)
    run_split datasets/revision/ood/ethanol_ood.xyz ethanol_ood datasets/revision/ood pbe
    run_split datasets/revision/water_clusters/val.xyz water_val datasets/revision/water_clusters pbe
    run_split datasets/revision/water_clusters/id_test.xyz water_id_test datasets/revision/water_clusters pbe
    run_split datasets/revision/water_clusters/ood_size.xyz water_ood_size datasets/revision/water_clusters pbe
    ;;
  pbe0)
    "${PY}" scripts/revision/prepare_pbe0_subset.py
    run_split datasets/revision/pbe0/ethanol_water_pbe0_subset.xyz ethanol_water_pbe0 datasets/revision/pbe0 pbe0
    ;;
  smoke)
    "${PY}" scripts/revision/generate_dft_labels.py \
      --trajectory datasets/revision/water_clusters/minima/n2_dimer.xyz \
      --output-dir datasets/revision/water_clusters \
      --output-prefix water_dimer_smoke \
      --xc pbe --d4 --df --no-gpu --resume --save-interval 1
    ;;
  *)
    echo "usage: $0 {smoke|pbe-train|pbe-rest|pbe0}" >&2
    exit 2
    ;;
esac
