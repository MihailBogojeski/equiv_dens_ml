#!/bin/bash
# SAD-prior, DF and model density errors plus LDA-exchange and von Weizsäcker
# errors against the full SCF density (score_density_functionals.py).
# Three processes share one GPU, as in submit_rescore_parallel.sbatch, so the
# job stays GPU-bound (Torch cancels jobs averaging <50% utilisation).
#
#   sbatch experiments/jacs_revision/slurm/submit_density_functionals.sbatch
#   MALON_DIRECT=1 sbatch ...   # after the direct malonaldehyde models exist
#SBATCH --job-name=dens-func
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --output=logs/dens_func_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 WANDB_MODE=offline TORCHDYNAMO_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=results/revision/rescore_fullref
S="./.venv/bin/python -B experiments/jacs_revision/analysis/score_density_functionals.py"
pids=()
if [[ -n "${MALON_DIRECT:-}" ]]; then
  $S --manifest "$M/manifest_malon_direct.json" > "logs/dens_func_${SLURM_JOB_ID}_malon_direct.log" 2>&1 &
  pids+=($!)
else
  $S --manifest "$M/manifest_pbe_water.json" --models pbe,direct_pbe \
    > "logs/dens_func_${SLURM_JOB_ID}_pbe_water.log" 2>&1 &
  pids+=($!)
  $S --manifest "$M/manifest_hybrid_water_malon.json" \
    --models wb97mv,direct_wb97mv,malon_pbe_d4_avdz,malon_wb97mv_def2tzvpd \
    > "logs/dens_func_${SLURM_JOB_ID}_hybrid_malon.log" 2>&1 &
  pids+=($!)
  $S --manifest "$M/manifest_extreme_modes.json" --no-grad \
    > "logs/dens_func_${SLURM_JOB_ID}_extreme.log" 2>&1 &
  pids+=($!)
fi
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "done, status $status"
exit $status
