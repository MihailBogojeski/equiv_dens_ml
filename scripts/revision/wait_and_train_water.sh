#!/usr/bin/env bash
# Poll CPU DFT outputs, then train water DenSNet + cutoff/seed sweeps (R1.1, R3.5).
# Safe to run on a submitted GPU node or locally after labels finish.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export WANDB_MODE="${WANDB_MODE:-offline}"
LOGDIR="results/revision/gpu_campaign"
mkdir -p "$LOGDIR"

need=(
  "datasets/revision/water_clusters/water_train_pyscf_augccpvdz_pbe.npy:1250"
  "datasets/revision/water_clusters/water_val_pyscf_augccpvdz_pbe.npy:250"
  "datasets/revision/water_clusters/water_ood_size_pyscf_augccpvdz_pbe.npy:300"
)

ready() {
  python - <<'PY'
from pathlib import Path
import numpy as np
import os
pairs = [x for x in os.environ["WATER_NEED"].split(";") if x]
ok = True
for item in pairs:
    path, target = item.split(":")
    target = int(target)
    p = Path(path)
    if not p.exists():
        print(f"missing {path} / {target}", flush=True)
        ok = False
        continue
    n = len(np.load(p, allow_pickle=True))
    print(f"{path}: {n}/{target}", flush=True)
    if n < target:
        ok = False
print("READY" if ok else "WAIT")
PY
}

WATER_NEED="$(printf '%s;' "${need[@]}")"
export WATER_NEED="${WATER_NEED%;}"
echo "$(date -Is) waiting for water PBE labels" | tee -a "$LOGDIR/gpu_hours.log"
while true; do
  status="$(ready | tee -a "$LOGDIR/water_wait.log" | tail -1)"
  if [[ "${status}" == READY ]]; then
    break
  fi
  sleep 600
done

echo "$(date -Is) water labels ready; training" | tee -a "$LOGDIR/gpu_hours.log"
python run.py train @config/training/water_clusters_001.txt \
  | tee "$LOGDIR/water_clusters_001.log"

for cut in 4 5 6 8; do
  echo "== cutoff ${cut} =="
  python run.py train @config/training/water_clusters_001.txt \
    @config/training/water_clusters_cutoff_${cut}.txt \
    --args_file_name=water_clusters_cutoff_${cut} \
    --save_dir=results/revision/water_clusters_cutoff_${cut} \
    | tee "$LOGDIR/water_cutoff_${cut}.log"
done

for seed in 7 123; do
  echo "== seed ${seed} =="
  python run.py train @config/training/water_clusters_001.txt \
    --split_seed="${seed}" \
    --args_file_name=water_clusters_seed_${seed} \
    --save_dir=results/revision/water_clusters_seed_${seed} \
    | tee "$LOGDIR/water_seed_${seed}.log"
done

echo "$(date -Is) water campaign finished" | tee -a "$LOGDIR/gpu_hours.log"
