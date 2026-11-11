#!/bin/bash
#SBATCH --job-name=thio-dpm
#SBATCH --output=logs/thio_dpm_%A_%a.out
#SBATCH --error=logs/thio_dpm_%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9
# Pilot: 10-mer rep0, 50k frames in 5k chunks (enough for 25/50 ps IR windows).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/mh7373/projects/equiv_dens_ml}"
mkdir -p logs results/revision/polythiophene_dipoles_10mer

CHUNK=5000
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))
OUT="results/revision/polythiophene_dipoles_10mer/rep0_${START}_${END}.npy"

./.venv/bin/python scripts/revision/analysis/recompute_polythiophene_dipoles_npy.py \
  --npy paper/trajectories/polythiophene/simulation_local_all_10mer_md_001_0.npy \
  --oligomer 10mer \
  --start-frame "$START" \
  --end-frame "$END" \
  --device cuda \
  --out "$OUT"
