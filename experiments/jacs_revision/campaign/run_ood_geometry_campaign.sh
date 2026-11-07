#!/usr/bin/env bash
# Detached OOD-geometry campaign: finish missing extreme-mode systems,
# project train clouds, score the water lambda scan if needed, write figures.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
LOGDIR="${ROOT}/logs/ood_geometry"
mkdir -p "${LOGDIR}" "${ROOT}/results/revision/mode_cv" "${ROOT}/results/revision/eval"

wait_for_pidfile_or_pattern() {
  local pattern="$1"
  while pgrep -f "${pattern}" >/dev/null 2>&1; do
    echo "$(date -Is) waiting for ${pattern}"
    sleep 30
  done
}

generate_if_missing() {
  local name="$1"
  local dest="${ROOT}/datasets/revision/ood_modes/${name}/modes.npz"
  if [[ -f "${dest}" ]]; then
    echo "$(date -Is) ${name}: modes.npz present"
    return 0
  fi
  echo "$(date -Is) ${name}: generating extreme modes"
  "${PY}" scripts/revision/analysis/generate_extreme_modes.py --system "${name}"
}

project_system() {
  local name="$1"
  local train="$2"
  local modes="${ROOT}/datasets/revision/ood_modes/${name}/modes.npz"
  local test="${ROOT}/datasets/revision/ood_modes/${name}/ood_modes.xyz"
  local out="${ROOT}/results/revision/mode_cv/${name}.json"
  if [[ ! -f "${modes}" ]]; then
    echo "$(date -Is) ${name}: skip projection, no modes.npz"
    return 0
  fi
  if [[ -f "${out}" ]]; then
    echo "$(date -Is) ${name}: projection already written"
    return 0
  fi
  echo "$(date -Is) ${name}: projecting onto modes"
  extra=()
  if [[ -n "${train}" && -e "${train}" ]]; then
    extra+=(--train "${train}")
  fi
  "${PY}" scripts/revision/analysis/project_mode_cv.py \
    --modes "${modes}" \
    --test "${test}" \
    "${extra[@]}" \
    --max-train 400 \
    --out "${out}"
}

echo "$(date -Is) campaign start"

# Project whatever Hessians are already on disk, then wait for the in-flight
# 12-mer rather than starting a second copy of that Hessian.
project_system water_n2 datasets/revision/water_clusters/train.xyz
project_system water_n6 datasets/revision/water_clusters/train.xyz
project_system ethanol datasets/ethanol_train_10.xyz
project_system ethanethiol datasets/ethanethiol_combo_Aidx-1000.npy
project_system resorcinol datasets/resorcinol_combo_kmeansidx-1000_train.npy
project_system thiophene_2mer datasets/thiophene_all_train_d4.npy
project_system malonaldehyde datasets/revision/malonaldehyde/train.xyz

if [[ ! -f "${ROOT}/results/revision/eval/ood_distance_scale.json" ]]; then
  echo "$(date -Is) scoring water distance scale with GFN2-xTB"
  "${PY}" scripts/revision/analysis/eval_distance_scaling.py \
    --manifest datasets/revision/ood_scale/manifest.json \
    --xtb \
    --out results/revision/eval/ood_distance_scale.json
fi

wait_for_pidfile_or_pattern "generate_extreme_modes.py --system thiophene_12mer"
generate_if_missing thiophene_12mer
project_system thiophene_12mer paper/trajectories/polythiophene/simulation_local_all_12mer_md_001_0.npy

echo "$(date -Is) writing response figures"
"${PY}" scripts/revision/analysis/plot_response_letter_figures.py || true

echo "$(date -Is) campaign done"
