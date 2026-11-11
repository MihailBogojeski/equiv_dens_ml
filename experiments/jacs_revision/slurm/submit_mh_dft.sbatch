#!/bin/bash
# After short MH finishes: pick A-D + empty bin, DFT + plot (CPU).
#SBATCH --job-name=thio-mh-dft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --output=logs/thiophene_mh_dft_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUDA_VISIBLE_DEVICES=""
MINIMA=${MINIMA:-results/revision/thiophene_mh/minima.json}
if [[ ! -f "$MINIMA" ]]; then
  echo "no $MINIMA yet"; exit 1
fi
n=$(./.venv/bin/python -c "import json; print(json.load(open('$MINIMA')).get('n_unique',0))")
if [[ "${n:-0}" -lt 2 ]]; then
  echo "only $n unique minima; need a longer hop run"; exit 1
fi

./.venv/bin/python scripts/revision/thiophene/analyze_mh_minima.py \
  --minima "$MINIMA" \
  --out results/revision/thiophene_mh/selected_for_dft.json

./.venv/bin/python scripts/revision/thiophene/validate_mh_minima_dft.py \
  --selected results/revision/thiophene_mh/selected_for_dft.json \
  --run-dft --relax \
  --out results/revision/thiophene_mh/dft_validation.json

./.venv/bin/python scripts/revision/thiophene/plot_thiophene_mh_figures.py \
  --train-cv results/revision/thiophene_mh/train_cv.json \
  --minima "$MINIMA" \
  --selected results/revision/thiophene_mh/selected_for_dft.json \
  --dft results/revision/thiophene_mh/dft_validation.json \
  --out-dir results/revision/figures/response
