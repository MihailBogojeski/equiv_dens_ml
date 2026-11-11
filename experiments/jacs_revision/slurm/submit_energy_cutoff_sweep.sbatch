#!/bin/bash
# Energy-head cutoff sweep, water PBE-D4 Δ: fresh v3 heads (train_energy_head.sh recipe) on the frozen density
# networks of the density cutoff sweep (results/revision/water_pbe_orca_cutoff_{4,5,6}), with the same --cutoff,
# 40k steps each, so 4 / 5 / 6 Å are compared at an equal budget (the 6 Å head is the like-for-like control; the
# production 6 Å head is at ~77k steps). Question: does a cutoff that leaves message passing work to do inside the
# 2-6-water training clusters transfer better to the larger clusters? The three heads share one GPU.
# Save roots results/revision/water_pbe_d4_avdz_energy_cutoff_<c>; resumable: re-submit (or chain) to continue.
# When all three have finished training, each is scored on the full sets like the _cluster heads:
#   results/revision/energy_forces_v3/water_pbe_d4_avdz_cutoff_<c>/ (+ locality.json).
#SBATCH --job-name=energy-cutoff
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/energy_cutoff_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=offline TORCHDYNAMO_DISABLE=1
# Python headers for Triton's gcc-built runtime helper (see train_energy_head.sh); needed by the scoring calls below
export CPATH=${CPATH:-/scratch/mh7373/tools/python3.12-include}
th=pbe_d4_avdz
declare -A DENS=([4]=results/revision/water_pbe_orca_cutoff_4/2026-08-26_R6lpIhIM
                 [5]=results/revision/water_pbe_orca_cutoff_5/2026-08-26_SLxdRhRB
                 [6]=results/revision/water_pbe_orca_cutoff_6/2026-08-26_Yg5567tx)
pids=()
for c in 4 5 6; do
  SCORE=0 LOG_TAG=cutoff${c}_${SLURM_JOB_ID:-local} SAVE_ROOT=results/revision/water_${th}_energy_cutoff_$c \
    DENSITY=${DENS[$c]} experiments/jacs_revision/slurm/train_energy_head.sh sad $th \
    --cutoff=$c.0 --max_steps=40000 --num_workers=2 &
  pids+=($!)
done
st=0; for p in "${pids[@]}"; do wait $p || st=1; done
[ $st -ne 0 ] && { echo "training did not finish (status $st); re-submit to resume"; exit $st; }
O=results/revision/energy_forces_v3; mkdir -p $O/logs
for c in 4 5 6; do
  run=$(dirname "$(dirname "$(ls -dt results/revision/water_${th}_energy_cutoff_$c/20*/checkpoints/latest_checkpoint.pth | head -1)")")
  out=$O/water_${th}_cutoff_$c; mkdir -p $out
  ( ./.venv/bin/python -B experiments/jacs_revision/analysis/energy_head_locality.py --run $run --theory $th \
      --out $out/locality.json > $O/logs/water_${th}_cutoff_${c}_locality.log 2>&1
    ./.venv/bin/python -B experiments/jacs_revision/analysis/score_energy_forces.py --run $run --theory $th --out $out \
      --splits id_test ood_size ood_order ood_density --batch 8 --max-atoms 240 --check-single 4 --single-tol 0.05 \
      > $O/logs/water_${th}_cutoff_$c.log 2>&1
    echo "cutoff $c scored (exit $?) -> $out" ) &
done
wait
