#!/bin/bash
# Score a pair of water energy heads: additivity/locality test, then every split one structure at a time.
# Usage: PAIR=sad|direct SUFFIX=_energy_ft_nonorm sbatch --dependency=afterok:<ft job> submit_energy_score.sbatch
#SBATCH --job-name=energy-score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --qos=gpu48
#SBATCH --output=logs/energy_score_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${PAIR:?}"; SUFFIX=${SUFFIX:-_energy_ft_nonorm}
export OMP_NUM_THREADS=4 WANDB_MODE=offline TORCHDYNAMO_DISABLE=1
pre=""; [ "$PAIR" = direct ] && pre="direct_"
pids=()
for th in pbe_d4_avdz wb97mv_def2tzvpd; do
  run=$(ls -dt results/revision/water_${pre}${th}${SUFFIX}/20* | head -1)
  out=results/revision/energy_forces_v2/water_${pre}${th}${SUFFIX#_energy}
  mkdir -p $out
  ( ./.venv/bin/python -B experiments/jacs_revision/analysis/energy_head_locality.py --run $run --theory $th --out $out/locality.json \
      > logs/energy_score_${SLURM_JOB_ID}_${th}_locality.log 2>&1
    ./.venv/bin/python -B experiments/jacs_revision/analysis/score_energy_forces.py --run $run --theory $th --out $out \
      --batch 1 --check-single 0 > logs/energy_score_${SLURM_JOB_ID}_${th}.log 2>&1 ) &
  pids+=($!)
  echo "$th: $run -> $out"
done
st=0; for p in "${pids[@]}"; do wait $p || st=1; done
exit $st
