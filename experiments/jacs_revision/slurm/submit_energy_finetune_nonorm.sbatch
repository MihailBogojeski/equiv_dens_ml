#!/bin/bash
# Final energy stage without the structure-wide layer norm (--normalize_en=0), as the paper's
# last-stage checkpoints were trained (polythiophene _ext, resorcinol _ext1, ethanethiol _ext4).
# With normalize_en=2 (config/training/water_wb97mv_001.txt) every atom's features are standardized
# over the whole structure, so the head is non-local: two ID clusters 100 A apart are non-additive by
# 1.6 kcal/mol and out-of-distribution clusters fail (energy_head_locality.py). Each head starts
# from its 2026-09-30 fine-tuned head (normalize_en=2, 20k steps); the norm has no parameters.
#
# Usage:
#   PAIR=sad    sbatch experiments/jacs_revision/slurm/submit_energy_finetune_nonorm.sbatch
#   PAIR=direct sbatch experiments/jacs_revision/slurm/submit_energy_finetune_nonorm.sbatch
#
# The density is rebuilt as it was fine-tuned (charge constraint; positive
# coefficients for direct), the model whose ε_ρ the letter reports.
#SBATCH --job-name=energy-ft0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --output=logs/energy_ft0_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${PAIR:?set PAIR=sad or PAIR=direct}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=offline TORCHDYNAMO_DISABLE=1
BASE="@config/training/water_energy_base.txt"
DIRECT="@config/training/water_direct_multiphase_overlay.txt"
TUNE=(--learning_rate=2e-4 --max_steps=20000 --integral_constraint=coeffs_in_coeffs_net --normalize_en=0)

launch() {  # name theory-config load_from init_from positive_coeffs [overlay]
  local name=$1 cfg=$2 density=$3 init=$4 pos=$5 overlay=${6:-}
  # Resume: if this head already has a run with a checkpoint, continue it (weights, optimizer, step,
  # data split and its saved arguments) instead of starting again from --init_from.
  local resume=() last
  last=$(ls -dt results/revision/${name}_energy_ft_nonorm/20*/checkpoints/latest_checkpoint.pth 2>/dev/null | head -1)
  [ -n "$last" ] && resume=(--restart="$(dirname "$(dirname "$last")")") && echo "$name: resuming ${resume[0]#--restart=}"
  ./.venv/bin/python -B run.py train --mode energy "$BASE" "$cfg" ${overlay:+"$overlay"} \
    --args_file_name="${name}_energy_ft_nonorm" --save_dir="results/revision/${name}_energy_ft_nonorm" \
    --load_from="$density" --init_from="$init" --positive_coeffs="$pos" "${TUNE[@]}" "${resume[@]}" \
    > "logs/energy_ft0_${SLURM_JOB_ID}_${name}.log" 2>&1 &
  pids+=($!)
}
pids=()
case "$PAIR" in
  sad)
    launch water_pbe_d4_avdz @config/training/water_energy_pbe.txt \
      results/revision/water_pbe_d4_avdz/2026-08-26_DQX1HUW0 \
      results/revision/water_pbe_d4_avdz_energy_ft/2026-09-30_5IkECCVf False
    launch water_wb97mv_def2tzvpd @config/training/water_energy_wb97mv.txt \
      results/revision/water_wb97mv_def2tzvpd/2026-08-26_itJSxLYq \
      results/revision/water_wb97mv_def2tzvpd_energy_ft/2026-09-30_7SPhQ3DD False ;;
  direct)
    launch water_direct_pbe_d4_avdz @config/training/water_energy_pbe.txt \
      results/revision/water_direct_pbe_d4_avdz/2026-08-26_jpx8oZi5 \
      results/revision/water_direct_pbe_d4_avdz_energy_ft/2026-09-30_lFn3acKA True "$DIRECT"
    launch water_direct_wb97mv_def2tzvpd @config/training/water_energy_wb97mv.txt \
      results/revision/water_direct_wb97mv_def2tzvpd/2026-08-26_SebXjydH \
      results/revision/water_direct_wb97mv_def2tzvpd_energy_ft/2026-09-30_6KFRlJjy True "$DIRECT" ;;
  *) echo "unknown PAIR $PAIR"; exit 2 ;;
esac
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "done, status $status"
exit $status
