#!/usr/bin/env bash
# Polythiophene 12-mer (86 atoms) single-point timings for the response letter.
# GPU QOS on Greene is gpu48 (MaxWall 48 h). No shorter GPU QOS exists.
# Request 1 h so backfill can slip this past the 24–48 h trains.
#
# Time estimate (1 warmup + 5 timed calls):
#   GPU kernels: ~1 min (DenSNet SI fit is 755 ms/step).
#   GPU load / JAX compile: 10–15 min (SO3LR is the wildcard).
#   CPU xTB / DFTB / MACE-POLAR: 15–25 min.
#   Buffer to 60 min.
#SBATCH --job-name=thio12-time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=results/revision/gpu_campaign/slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=scripts/revision/lib/gpu_queue_policy.sh
source scripts/revision/lib/gpu_queue_policy.sh
mkdir -p results/revision/gpu_campaign
source .venv/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCHINDUCTOR_DISABLE="${TORCHINDUCTOR_DISABLE:-1}"

STRUCTURE=datasets/thiophene12mer_test.xyz
MODEL=paper/models/polythiophene/2024-03-23_1XDL67zp_ext
ARGS=config/md/nn/polythiophene_12mer.txt
NPY=datasets/thiophene_all_train_d4.npy
NREP=5

echo "== 12-mer GPU timings =="
python scripts/revision/analysis/benchmark_figure2.py \
  --structure "$STRUCTURE" \
  --densnet-model "$MODEL" \
  --densnet-args "$ARGS" \
  --densnet-np-dataset "$NPY" \
  --methods densnet,maceoff,aimnet2,so3lr,macepolar \
  --use-gpu --n-rep "$NREP" \
  --out results/revision/figure2_timing_gpu_12mer.json

echo "== 12-mer CPU timings =="
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
python scripts/revision/analysis/benchmark_figure2.py \
  --structure "$STRUCTURE" \
  --densnet-model "$MODEL" \
  --densnet-args "$ARGS" \
  --densnet-np-dataset "$NPY" \
  --methods densnet,maceoff,aimnet2,so3lr,macepolar,gfn2xtb,gxtb,dftb \
  --n-rep "$NREP" \
  --out results/revision/figure2_timing_12mer.json

echo "finished $(date -Is)"
