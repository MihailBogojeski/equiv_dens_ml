#!/usr/bin/env bash
# Wait for DenSNet relax jobs, then evaluate H-bond metrics (R1.1).
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=./.venv/bin/python
REF=datasets/revision/water_clusters/ood_size.xyz

wait_for() {
  local pattern=$1
  while pgrep -f "$pattern" >/dev/null; do
    sleep 30
  done
}

echo "waiting for wb97mv relax..."
wait_for "relax_water_clusters.*water_wb97mv"
echo "waiting for PBE relax..."
wait_for "relax_water_clusters.*water_pbe_d4"

$PY scripts/revision/water/evaluate_hbond_metrics.py \
  --ref "$REF" \
  --pred-xyz results/revision/water_relaxed_wb97mv/ood_size_relaxed.xyz \
  --out results/revision/water_hbond_densnet_wb97mv.json

$PY scripts/revision/water/evaluate_hbond_metrics.py \
  --ref "$REF" \
  --pred-xyz results/revision/water_relaxed_pbe/ood_size_relaxed.xyz \
  --out results/revision/water_hbond_densnet_pbe.json

echo "hbond evaluation complete"
