#!/bin/bash
# gpu4pyscf labelling for one contiguous index range of a trajectory.
#
# Overflow capacity for the PBE-D4 level only. wB97M-V labels are ORCA-only
# (see theory_levels.py) because PySCF has no COSX to match ORCA's RIJCOSX,
# and a size-correlated switch between integral schemes would corrupt the very
# size-scaling trend this dataset exists to measure.
#
# Each array task owns its own output part and resumes into it, so a task
# killed at the walltime restarts at the first unlabelled frame of its range.
#SBATCH --job-name=water-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=l40s_public
#SBATCH --output=logs/water_gpu_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TRAJ=${TRAJ:?set TRAJ to the multi-frame XYZ}
OUTDIR=${OUTDIR:?set OUTDIR}
PREFIX=${PREFIX:?set PREFIX}
XC=${XC:-pbe}
BASIS=${BASIS:-augccpvdz}
CHUNK=${CHUNK:-50}
D4=${D4:-1}

START=$((SLURM_ARRAY_TASK_ID * CHUNK))
END=$((START + CHUNK))

mkdir -p logs "$OUTDIR"

./.venv/bin/python scripts/revision/generate_dft_labels.py \
  --trajectory "$TRAJ" \
  --output-dir "$OUTDIR" \
  --output-prefix "${PREFIX}_part$(printf '%04d' "${SLURM_ARRAY_TASK_ID}")" \
  --xc "$XC" \
  --basis "$BASIS" \
  --df \
  --resume \
  --save-interval 5 \
  --start-index "$START" \
  --end-index "$END" \
  $(test "$D4" -ne 0 && echo "--d4")
