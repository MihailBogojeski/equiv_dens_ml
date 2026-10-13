#!/bin/bash
# Baseline sweep: h2o_small and water clusters on GPU (all sizes that fit) and CPU (small sizes).
# usage: benchmarks/run_baseline.sh [tag]   (tag prefixes the result files, default baseline_)
cd "$(dirname "$0")"
TAG=${1:-baseline_}
export WANDB_MODE=offline
PY=/opt/conda/envs/equiv_dens_ml/bin/python
$PY bench_model.py --dataset h2o_small --device cuda --batch_size 5 --tag $TAG
for n in 1 2 4 8; do $PY bench_model.py --dataset water$n --device cuda --batch_size 2 --tag $TAG; done
for n in 16 32 64; do $PY bench_model.py --dataset water$n --device cuda --batch_size 1 --reps 3 --tag $TAG; done
$PY bench_model.py --dataset h2o_small --device cpu --batch_size 5 --tag $TAG
for n in 1 2 4; do $PY bench_model.py --dataset water$n --device cpu --batch_size 2 --reps 3 --tag $TAG; done
