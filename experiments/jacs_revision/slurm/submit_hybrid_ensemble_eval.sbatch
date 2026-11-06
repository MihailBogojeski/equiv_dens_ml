#!/bin/bash
# R3.3: score two extra ωB97M-V init_seed members on water_id_test, refresh table.
#SBATCH --job-name=hybrid-ens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --output=logs/hybrid_ensemble_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=offline
PY=./.venv/bin/python
LABELS=datasets/revision/water_labels
BASE=${LABELS}/water_id_test_wb97mv_def2tzvpd_base.npy
DENS=${LABELS}/water_id_test_wb97mv_def2tzvpd.npy

eval_member() {
  local run_dir=$1
  local out_dir=$2
  mkdir -p "$out_dir"
  if [[ -f "${out_dir}/water_id_test.json" ]]; then
    echo "keeping ${out_dir}/water_id_test.json"
    return 0
  fi
  $PY scripts/revision/csh/csh_evaluate.py \
    --args-file "${run_dir}/args.txt" --run-dir "$run_dir" \
    --np-dataset "$BASE" --dens-dataset "$DENS" \
    --elements 1,8 --label water_id_test \
    --out "${out_dir}/water_id_test.json"
}

# Primary member already scored.
mkdir -p results/revision/eval/wb97mv_def2tzvpd_seed7 results/revision/eval/wb97mv_def2tzvpd_seed123
if [[ -f "$BASE" && -f "$DENS" ]]; then
  eval_member results/revision/water_wb97mv_def2tzvpd_seed7/2026-08-27_VkxnzRjm \
    results/revision/eval/wb97mv_def2tzvpd_seed7
  eval_member results/revision/water_wb97mv_def2tzvpd_seed123/2026-08-27_fwdfP2ic \
    results/revision/eval/wb97mv_def2tzvpd_seed123
else
  echo "id-test labels missing; refreshing table from existing single-member evals"
fi

THEORY_WB97="wb97mv_def2tzvpd:results/revision/eval/wb97mv_def2tzvpd"
if [[ -f results/revision/eval/wb97mv_def2tzvpd_seed7/water_id_test.json ]]; then
  THEORY_WB97="${THEORY_WB97},results/revision/eval/wb97mv_def2tzvpd_seed7"
fi
if [[ -f results/revision/eval/wb97mv_def2tzvpd_seed123/water_id_test.json ]]; then
  THEORY_WB97="${THEORY_WB97},results/revision/eval/wb97mv_def2tzvpd_seed123"
fi

$PY scripts/revision/analysis/pbe_vs_hybrid_table.py \
  --theory "pbe_d4_avdz:results/revision/eval/pbe_d4_avdz" \
  --theory "$THEORY_WB97" \
  --out results/revision/pbe_vs_hybrid.json \
  --markdown results/revision/pbe_vs_hybrid.md
