#!/bin/bash
#SBATCH --job-name=csh_df_gpu
#SBATCH --partition=l40s_public
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/csh_df_gpu_%A_%a.out
#SBATCH --array=0-36

# Same work as submit_csh_cpu.sbatch on a GPU. Both arrays can be queued
# simultaneously against the same shard list; the exclusive-create claim
# prevents duplicated effort, so this is purely a hedge against queue latency.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/mh7373/projects/equiv_dens_ml}"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

MANIFEST=${MANIFEST:-datasets/revision/omol_csh/manifests/train_neutral_hcnos.json}
H5=${H5:-datasets/revision/omol_csh/omol_csh_58k_train.h5}
OUTDIR=${OUTDIR:-datasets/revision/csh_df/train}
SHARD_SIZE=${SHARD_SIZE:-64}
TABLE=${TABLE:-datasets/revision/omol_csh/ao_permutation_8el.json}

./.venv/bin/python scripts/revision/csh_shard_to_df.py \
  --manifest "$MANIFEST" \
  --h5 "$H5" \
  --outdir "$OUTDIR" \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --shard-size "$SHARD_SIZE" \
  --table "$TABLE" \
  --device gpu
