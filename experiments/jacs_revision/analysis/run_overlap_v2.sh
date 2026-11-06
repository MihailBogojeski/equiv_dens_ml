#!/usr/bin/env bash
# Train/test overlap for R1.5, with train-train baselines (train_test_overlap.py).
# SOAP needs dscribe, installed apart from .venv in tools/pydeps-soap.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=tools/pydeps-soap
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
PY=./.venv/bin/python
OUT=results/revision/overlap_v2
RUN="$PY -B experiments/jacs_revision/analysis/train_test_overlap.py"
mkdir -p "$OUT"

$RUN --system ethanethiol --train datasets/ethanethiol_combo_Aidx-1000.npy \
  --test datasets/ethanethiol_rand-1000.npy --out $OUT/ethanethiol.json
$RUN --system resorcinol --train datasets/resorcinol_combo_kmeansidx-1000_train.npy \
  --test datasets/resorcinol_augccpvdz_test.npy --out $OUT/resorcinol.json
for seed in 42 1 2; do
  $RUN --system ethanol_md17 --train datasets/external/md17/md17_ethanol.npz \
    --train-indices $OUT/md17_ethanol_seed${seed}_train_idx.npy \
    --test datasets/external/md17/md17_ethanol.npz \
    --test-indices $OUT/md17_ethanol_seed${seed}_test_idx.npy --seed $seed \
    --note "paper split not on disk: 1000/1000 disjoint random MD17 frames, seed ${seed}, the SI procedure" \
    --out $OUT/ethanol_md17_seed${seed}.json
done
$RUN --system thiophene_4mer --train datasets/thiophene_all_train_d4.npy \
  --test datasets/thiophene4mer_test_d4.npy --out $OUT/thiophene_4mer.json
$RUN --system thiophene_6mer --train datasets/thiophene_all_train_d4.npy \
  --test datasets/thiophene6mer_test_d4.npy --out $OUT/thiophene_6mer.json
$RUN --system thiophene_12mer --train datasets/thiophene_all_train_d4.npy \
  --test paper/trajectories/polythiophene/simulation_local_all_12mer_md_001_0.npy --max-test 300 \
  --note "12-mer frames from the published 12-mer ML-MD trajectory; no 12-mer in training, SOAP only" \
  --out $OUT/thiophene_12mer.json
$RUN --system water_id --train datasets/revision/water_clusters/train.xyz \
  --test datasets/revision/water_clusters/id_test.xyz --out $OUT/water_id.json
$RUN --system water_ood_size --train datasets/revision/water_clusters/train.xyz \
  --test datasets/revision/water_clusters/ood_size.xyz \
  --note "n=8,10,12 have no same-size training cluster; SOAP only" --out $OUT/water_ood_size.json
