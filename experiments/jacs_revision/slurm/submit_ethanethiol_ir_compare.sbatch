#!/usr/bin/env bash
# Reviewer 3 point 2: matched 50 ps ethanethiol IR trajectories.
# CPU-only: a 9-atom molecule cannot hold 50% SM util on Torch gl* GPUs,
# and those nodes cancel the job. Submit all methods:
#   sbatch --array=0-3 scripts/revision/submit_ethanethiol_ir_compare.sbatch
# Or one method:
#   METHOD=macepolar sbatch scripts/revision/submit_ethanethiol_ir_compare.sbatch
#SBATCH --job-name=etsh-ir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --partition=cs
#SBATCH --output=logs/ethanethiol_ir_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results/revision/ethanethiol_ir_mlips

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) array_method=densnet ;;
  1) array_method=maceoff ;;
  2) array_method=macepolar ;;
  3) array_method=so3lr ;;
  *) echo "invalid array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac
METHOD=${METHOD:-$array_method}
OUT_ROOT=${OUT_ROOT:-results/revision/ethanethiol_ir_mlips}
PRODUCTION_PS=${PRODUCTION_PS:-50}
EQUILIBRATION_PS=${EQUILIBRATION_PS:-2}

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

./.venv/bin/python scripts/revision/analysis/run_mlip_ir_md.py \
  --method "$METHOD" \
  --device cpu \
  --structure results/revision/ethanethiol_train_frame0.xyz \
  --out-dir "${OUT_ROOT}/${METHOD}" \
  --production-ps "$PRODUCTION_PS" --equilibration-ps "$EQUILIBRATION_PS" \
  --windows-ps 25 50

NIST=datasets/revision/external/ethanethiol_nist_gas_ir.csv
if [[ -f "$NIST" ]] &&
   [[ -f "${OUT_ROOT}/densnet/ir_50ps.csv" ]] &&
   [[ -f "${OUT_ROOT}/maceoff/ir_50ps.csv" ]] &&
   [[ -f "${OUT_ROOT}/macepolar/ir_50ps.csv" ]] &&
   [[ -f "${OUT_ROOT}/so3lr/ir_50ps.csv" ]]; then
  ./.venv/bin/python scripts/revision/analysis/plot_ethanethiol_ir_compare.py \
    --spectrum "DenSNet=${OUT_ROOT}/densnet/ir_50ps.csv" \
    --spectrum "MACE-OFF + DenSNet dipole=${OUT_ROOT}/maceoff/ir_50ps.csv" \
    --spectrum "MACE-POLAR-1-M=${OUT_ROOT}/macepolar/ir_50ps.csv" \
    --spectrum "SO3LR=${OUT_ROOT}/so3lr/ir_50ps.csv" \
    --spectrum "NIST=${NIST}" \
    --figure results/revision/figures/response/ethanethiol_ir_mlips.png \
    --metrics "${OUT_ROOT}/metrics.json"
  install -D -m 0644 \
    results/revision/figures/response/ethanethiol_ir_mlips.png \
    docs/response-to-jacs-review/figures/response/ethanethiol_ir_mlips.png
fi
