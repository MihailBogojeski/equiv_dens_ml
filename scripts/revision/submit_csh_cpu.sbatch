#!/bin/bash
#SBATCH --job-name=csh_df_cpu
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/csh_df_cpu_%A_%a.out
#SBATCH --array=0-36

# Density-fit one shard of the CSH neutral subset on CPU.
#
# Shards are claimed with an exclusive-create lock, so this array and the GPU
# array can be submitted at the same time over the same shard list: whichever
# queue drains faster simply completes more shards, and reruns are no-ops.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/mh7373/projects/equiv_dens_ml}"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_VISIBLE_DEVICES=""

MANIFEST=${MANIFEST:-datasets/revision/omol_csh/manifests/train_neutral_hcnos.json}
H5=${H5:-datasets/revision/omol_csh/omol_csh_58k_train.h5}
OUTDIR=${OUTDIR:-datasets/revision/csh_df/train}
SHARD_SIZE=${SHARD_SIZE:-64}

./.venv/bin/python scripts/revision/csh_shard_to_df.py \
  --manifest "$MANIFEST" \
  --h5 "$H5" \
  --outdir "$OUTDIR" \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --shard-size "$SHARD_SIZE" \
  --device cpu
