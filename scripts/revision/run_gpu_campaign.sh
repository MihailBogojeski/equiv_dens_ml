#!/usr/bin/env bash
# Wave C GPU jobs. TASK=pbe0|ethanol-md|thiophene-md|water|quick|all
# Set ALLOW_GL056=1 to run on this allocation (share remaining L40S VRAM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
HOST="$(hostname -s || hostname)"
TASK="${1:-${TASK:-all}}"
if [[ "${HOST}" == gl056* && "${ALLOW_GL056:-0}" != 1 ]]; then
  echo "Refusing Wave C on ${HOST} unless ALLOW_GL056=1."
  echo "Submit scripts/revision/submit_revision_gpu_jobs.sh to a free Greene GPU, or export ALLOW_GL056=1."
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && ! -e /dev/nvidia0 ]]; then
  echo "No GPU device visible."
  exit 2
fi
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCHINDUCTOR_DISABLE="${TORCHINDUCTOR_DISABLE:-1}"
LOGDIR="results/revision/gpu_campaign"
mkdir -p "$LOGDIR"
date -Is | tee -a "$LOGDIR/gpu_hours.log"
echo "host=${HOST} cuda=${CUDA_VISIBLE_DEVICES:-all} task=${TASK}" | tee -a "$LOGDIR/gpu_hours.log"

run_quick() {
  echo "== GPU Figure 2 =="
  python scripts/revision/benchmark_figure2.py \
    --methods densnet,maceoff,aimnet2,so3lr \
    --use-gpu --n-rep 5 \
    --out results/revision/figure2_timing_gpu.json \
    | tee "$LOGDIR/figure2_gpu.log"

  echo "== GPU ethanol geo-opt =="
  python scripts/revision/optimize_densnet.py \
    --structure datasets/ethanol_train_10.xyz \
    --model paper/models/ethanol/2024-03-22_96w7KyGG \
    --args-file config/md/nn/ethanol_500ps.txt \
    --skip-dft --use-gpu \
    --out results/revision/geoopt_ethanol_gpu.json \
    | tee "$LOGDIR/geoopt_ethanol_gpu.log"

  echo "== GPU ethanol NVE 0.1 ps =="
  python scripts/revision/cpu_mlip_suite.py \
    --methods densnet --device cuda --skip-geoopt \
    --nve-steps 200 \
    --out-dir results/revision/mlip_gpu_densnet \
    | tee "$LOGDIR/nve_gpu.log"
  python scripts/revision/nve_energy_drift.py \
    --hdf5 results/revision/mlip_gpu_densnet/densnet_nve.jsonl \
    --n-atoms 9 \
    --out results/revision/nve_drift_gpu.json \
    | tee -a "$LOGDIR/nve_gpu.log"

  echo "== score 96w7KyGG forces on labeled OOD frames =="
  python scripts/revision/score_ood_forces.py \
    --use-gpu \
    --out results/revision/eval_ethanol_ood_forces.json \
    | tee "$LOGDIR/eval_ethanol_ood.log"
}

run_pbe0() {
  echo "== PBE0 ethanol/water subset train =="
  python run.py train @config/training/ethanol_pbe0_001.txt \
    | tee "$LOGDIR/ethanol_pbe0_001.log"
}

run_ethanol_md() {
  echo "== ethanol 500 ps MD =="
  python run.py md @config/md/nn/ethanol_500ps.txt \
    | tee "$LOGDIR/ethanol_500ps.log"
}

run_thiophene_md() {
  echo "== thiophene 2-mer 500 ps MD =="
  python run.py md @config/md/nn/polythiophene_2mer_500ps.txt \
    | tee "$LOGDIR/thiophene2_500ps.log"
}

run_water() {
  bash scripts/revision/wait_and_train_water.sh
}

case "${TASK}" in
  quick) run_quick ;;
  pbe0) run_pbe0 ;;
  ethanol-md) run_ethanol_md ;;
  thiophene-md) run_thiophene_md ;;
  water) run_water ;;
  all)
    run_quick
    run_pbe0
    run_ethanol_md
    run_thiophene_md
    run_water
    ;;
  *)
    echo "usage: $0 {quick|pbe0|ethanol-md|thiophene-md|water|all}" >&2
    exit 2
    ;;
esac

date -Is | tee -a "$LOGDIR/gpu_hours.log"
echo "Wave C task ${TASK} finished" | tee -a "$LOGDIR/gpu_hours.log"
