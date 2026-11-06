#!/bin/bash
# GFN2 / motif geometry for the md40x expansion. Does not touch the 3k trees.
#
#   STAGE=phase_md40x     sbatch scripts/revision/submit_md40x_geometry.sbatch
#   STAGE=phase_b_parents sbatch scripts/revision/submit_md40x_geometry.sbatch
#   STAGE=phase_b_md40x   sbatch scripts/revision/submit_md40x_geometry.sbatch
#SBATCH --job-name=md40x-geom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --output=logs/md40x_geom_%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
unset LD_PRELOAD
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p logs/revision

STAGE=${STAGE:-phase_md40x}
WORKERS=${WORKERS:-16}

echo "[$(date -Is)] ${SLURM_JOB_ID} stage=${STAGE} workers=${WORKERS} host=$(hostname)"
WORKERS="$WORKERS" scripts/revision/water/run_geometry_expansion.sh "$STAGE"
echo "[$(date -Is)] ${STAGE} finished"
