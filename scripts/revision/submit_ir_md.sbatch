#!/bin/bash
# Long DenSNet MD with logged dipoles, for the IR-versus-length curve (R3.6).
#
# Usage:  MOL=ethanol|thiophene2 sbatch scripts/revision/submit_ir_md.sbatch
#
# Reviewer 3 asked whether the spectra are converged with respect to trajectory
# length, which needs one long run sampled at several lengths rather than
# several runs. The MD writes HDF5 as it goes, so ir_vs_length.py can be pointed
# at the file at any time and will report whatever windows have been reached --
# a run cut off at the walltime still answers the question for the lengths it
# did reach, and resubmitting extends it.
#
# Both MD configs were pointing at datasets/revision/sad_pbe_augccpvdz.npy,
# a prior that carries no df_coeffs. For ethanol that silently substituted the
# wrong delta-learning reference; for the thiophene _ext checkpoint it also
# halved the energy head's L=0 input width from 60 to 30 and the load aborted
# outright. Both configs now name the priors their checkpoints were trained
# against and load with zero missing tensors.
#
# 48 h because a 1e6-step trajectory at half-femtosecond steps does not finish
# in 24 h at any throughput this model has shown.
#SBATCH --job-name=ir-md
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public,l40s_public
#SBATCH --output=logs/ir_md_%x_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE=offline

MOL=${MOL:-ethanol}
case "$MOL" in
  ethanol)    CFG=config/md/nn/ethanol_500ps.txt ;;
  thiophene2) CFG=config/md/nn/polythiophene_2mer_500ps.txt ;;
  *) echo "unknown MOL ${MOL}"; exit 1 ;;
esac

LOGDIR=results/revision/ir_md/${MOL}
mkdir -p "$LOGDIR" logs

echo "=== ${MOL} MD from ${CFG} on $(hostname) at $(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Throughput is measured rather than assumed. The single-point benchmark says
# 467 ms/step, which would put 1e6 steps at five days; the MD loop is faster
# than that, and how much faster decides whether 500 ps is reachable or whether
# the curve gets reported at the lengths that were.
start=$(date +%s)
./.venv/bin/python run.py md "@${CFG}" 2>&1 | tee "${LOGDIR}/md.log"
rc=${PIPESTATUS[0]}
elapsed=$(( $(date +%s) - start ))
echo "md exited ${rc} after ${elapsed}s"

hdf5=$(find results scratch -name "*${MOL}*.hdf5" -newermt "-${elapsed} seconds" 2>/dev/null | head -1)
[[ -z "$hdf5" ]] && hdf5=$(find results scratch -name "*.hdf5" -newermt "-${elapsed} seconds" 2>/dev/null | head -1)

if [[ -n "$hdf5" ]]; then
  echo "=== IR versus trajectory length from ${hdf5} ==="
  ./.venv/bin/python scripts/revision/ir_vs_length.py \
    --hdf5 "$hdf5" --windows "50,100,200,500" \
    --out "results/revision/ir_convergence_${MOL}"
else
  echo "no HDF5 trajectory found; nothing to transform"
fi
echo "finished $(date -Is)"
