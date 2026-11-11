#!/bin/bash
# Full-set energy/force scores of the cluster energy heads for the response letter (score set "_cluster"):
#   water Delta = *_energy_v3_slurm, water direct = water_direct_*_energy_v3, malonaldehyde = malon_energy_v2_slurm.
# Water: ID test + three OOD tiers, then the extreme-mode frames (dimer, hexamer). Malonaldehyde: train sample,
# validation, proton-transfer scan and extreme modes. Best checkpoint of each run at the time the job starts.
# Output results/revision/energy_forces_v3/<name>_cluster/. Figures: ENERGY_SUFFIX=_cluster plot_response_figures_v2.py.
#SBATCH --job-name=energy-score-cl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/energy_score_cluster_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=4 WANDB_MODE=offline TORCHDYNAMO_DISABLE=1
O=results/revision/energy_forces_v3
L=datasets/revision/ood_geom_labels
mkdir -p $O/logs
log() { echo "$(date +%T) $*" >> $O/logs/progress_cluster.log; }
water_run() {  # kind theory
  if [ "$1" = sad ]; then ls -dt results/revision/water_$2_energy_v3_slurm/20*/ | head -1
  else ls -dt results/revision/water_direct_$2_energy_v3/20*/ | head -1; fi
}
chain() {  # theory [gpu]  (gpu: run locally on that card instead of the job's one GPU)
  local th=$1
  [ -n "${2:-}" ] && export CUDA_VISIBLE_DEVICES=$2
  for kind in sad direct; do
    local pre=""; [ $kind = direct ] && pre=direct_
    local run; run=$(water_run $kind $th)
    log "start water $pre$th ($run)"
    ./.venv/bin/python -B experiments/jacs_revision/analysis/score_energy_forces.py --run $run --theory $th \
      --out $O/water_${pre}${th}_cluster --splits id_test ood_size ood_order ood_density --batch 8 --max-atoms 240 \
      --check-single 4 --single-tol 0.05 > $O/logs/water_${pre}${th}_cluster.log 2>&1
    log "end water $pre$th exit=$?"
    ./.venv/bin/python -B experiments/jacs_revision/analysis/score_energy_forces.py --run $run --theory $th \
      --out $O/modes_water_${pre}${th}_cluster --splits \
      --label-file ood_modes_water_n2=$L/ood_modes_water_n2_${th}_base.npy --label-file ood_modes_water_n6=$L/ood_modes_water_n6_${th}_base.npy \
      --batch 8 --max-atoms 240 --check-single 2 --single-tol 0.5 > $O/logs/modes_water_${pre}${th}_cluster.log 2>&1
    log "end modes $pre$th exit=$?"
  done
  local mrun; mrun=$(ls -dt results/revision/malon_energy_v2_slurm/malonaldehyde_$th/energy/20*/ | head -1)
  log "start malon $th ($mrun)"
  ./.venv/bin/python -B experiments/jacs_revision/analysis/score_malon_energy.py --run $mrun --theory $th \
    --out $O/malon_${th}_cluster --batch 16 --label-file ood_modes_malonaldehyde=$L/ood_modes_malonaldehyde_${th}_base.npy \
    > $O/logs/malon_${th}_cluster.log 2>&1
  log "end malon $th exit=$?"
}
if [ -n "${LOCAL_GPUS:-}" ]; then  # e.g. LOCAL_GPUS="0 1" bash submit_energy_score_cluster.sbatch (interactive allocation)
  read -r g0 g1 <<< "$LOCAL_GPUS"
  chain pbe_d4_avdz "$g0" &
  chain wb97mv_def2tzvpd "${g1:-$g0}" &
else
  chain pbe_d4_avdz &
  chain wb97mv_def2tzvpd &
fi
wait
log "cluster scoring done"
