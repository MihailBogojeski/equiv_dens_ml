#!/bin/bash
#SBATCH --job-name=jacs-ir-compare
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --output=logs/jacs_ir_compare_%j.out
set -euo pipefail
: "${CODE_ROOT:?set CODE_ROOT to the recovery checkout}"
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
out=results/revision/jacs_recovery_20260927/ethanethiol_ir
./.venv/bin/python -B "$CODE_ROOT/scripts/revision/analysis/plot_ethanethiol_ir_compare.py" \
  --spectrum "DenSNet=$out/densnet/ir_50ps.csv" \
  --spectrum "MACE-OFF + DenSNet dipole=$out/maceoff/ir_50ps.csv" \
  --spectrum "MACE-POLAR-1-M=$out/macepolar/ir_50ps.csv" \
  --spectrum "SO3LR=$out/so3lr/ir_50ps.csv" \
  --spectrum "NIST=datasets/revision/external/ethanethiol_nist_gas_ir.csv" \
  --figure "$out/comparison.png" --metrics "$out/metrics.json"
