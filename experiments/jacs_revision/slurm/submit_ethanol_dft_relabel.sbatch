#!/bin/bash
#SBATCH --job-name=ethanol-dft-relabel
#SBATCH --output=logs/ethanol_dft_relabel_%A_%a.out
#SBATCH --error=logs/ethanol_dft_relabel_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-89%20
# R2.11: regenerate ~900-frame ethanol PySCF/PBE aug-cc-pVDZ training pack.
# Each array task labels 10 consecutive frames; merge with merge_ethanol_dft_shards.py.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/mh7373/projects/equiv_dens_ml}"
mkdir -p logs datasets/revision/ethanol_dft_shards

PY=./.venv/bin/python
FRAMES_PER_TASK=10
START=$(( SLURM_ARRAY_TASK_ID * FRAMES_PER_TASK ))
END=$(( START + FRAMES_PER_TASK ))

# Prefer recovered MD17 ethanol geometries; fall back to paper stub if absent.
TRAJ=${ETHANOL_TRAJECTORY:-paper/trajectories/ethanol/simulation_cluster_all_001_compressed_0.npy}
OUT=datasets/revision/ethanol_dft_shards/shard_${SLURM_ARRAY_TASK_ID}.npy

echo "task ${SLURM_ARRAY_TASK_ID}: frames ${START}-${END} from ${TRAJ}"

$PY scripts/revision/lib/generate_dft_labels.py \
  --trajectory "$TRAJ" \
  --output-prefix "ethanol_relabel_${SLURM_ARRAY_TASK_ID}" \
  --output-dir datasets/revision/ethanol_dft_shards \
  --xc pbe \
  --basis aug-cc-pVDZ \
  --d4 --df \
  --start-index "$START" \
  --end-index "$END" \
  --no-gpu

# Rename geometry pack to shard name expected by merge_ethanol_dft_shards.py
mv "datasets/revision/ethanol_dft_shards/ethanol_relabel_${SLURM_ARRAY_TASK_ID}_npy.npy" "$OUT"
