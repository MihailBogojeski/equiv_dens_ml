#!/usr/bin/env bash
# Wave C: DenSNet train / IR / geo-opt on a FREE GPU node (not gl056).
# Refuse to start if this host is gl056 or CUDA is already claimed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
HOST="$(hostname -s || hostname)"
if [[ "${HOST}" == gl056* ]]; then
  echo "Refusing Wave C on ${HOST}: GPUs are reserved for other jobs."
  echo "Submit scripts/revision/submit_gpu_campaign.sbatch to another Greene GPU node."
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && ! -e /dev/nvidia0 ]]; then
  echo "No GPU device visible."
  exit 2
fi
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
LOGDIR="results/revision/gpu_campaign"
mkdir -p "$LOGDIR"
date -Is | tee -a "$LOGDIR/gpu_hours.log"
echo "host=${HOST} cuda=${CUDA_VISIBLE_DEVICES:-all}" | tee -a "$LOGDIR/gpu_hours.log"

echo "== water cluster train =="
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
    --split_seed=${seed} \
    --args_file_name=water_clusters_seed_${seed} \
    --save_dir=results/revision/water_clusters_seed_${seed} \
    | tee "$LOGDIR/water_seed_${seed}.log"
done

if [[ -f datasets/revision/pbe0/ethanol_water_pbe0_pyscf_augccpvdz_pbe0.npy ]]; then
  echo "== PBE0 ethanol subset =="
  python run.py train @config/training/ethanol_pbe0_001.txt \
    | tee "$LOGDIR/ethanol_pbe0_001.log"
fi

echo "== ethanol 500 ps MD =="
python run.py md @config/md/nn/ethanol_500ps.txt \
  | tee "$LOGDIR/ethanol_500ps.log"

echo "== thiophene 2-mer 500 ps MD =="
python run.py md @config/md/nn/polythiophene_2mer_500ps.txt \
  | tee "$LOGDIR/thiophene2_500ps.log"

echo "== production geo-opt / NVE =="
python scripts/revision/optimize_densnet.py \
  --structure datasets/ethanol_train_10.xyz \
  --model paper/models/ethanol/2024-03-22_96w7KyGG \
  --use-gpu \
  --out results/revision/geoopt_ethanol_gpu.json \
  | tee "$LOGDIR/geoopt_ethanol.log"

date -Is | tee -a "$LOGDIR/gpu_hours.log"
echo "Wave C finished" | tee -a "$LOGDIR/gpu_hours.log"
