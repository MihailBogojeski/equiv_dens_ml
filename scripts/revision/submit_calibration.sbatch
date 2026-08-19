#!/bin/bash
#SBATCH --job-name=water-calib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --output=results/revision/calibration/slurm-%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load orca/6.1.1
export OMP_NUM_THREADS=1

SIZES=${SIZES:-3,6}
THEORIES=${THEORIES:-pbe_d4_avdz,wb97mv_def2tzvpd}
ENGINES=${ENGINES:-orca,pyscf}
OUT=${OUT:-results/revision/calibration/calibration.json}
EXTRA=${EXTRA:---reference-orca}

mkdir -p "$(dirname "$OUT")"

./.venv/bin/python scripts/revision/calibrate_theory.py \
  --sizes "$SIZES" \
  --theories "$THEORIES" \
  --engines "$ENGINES" \
  --nprocs "${SLURM_CPUS_PER_TASK}" \
  --scratch "${SLURM_TMPDIR:-/tmp}/water_calib_${SLURM_JOB_ID}" \
  --out "$OUT" \
  ${EXTRA}
