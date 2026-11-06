#!/bin/bash
# Fine-tune water energy heads after the 2026-09-28 energy-loss and layer-norm
# fixes (errors.py energy shape; per-molecule layer norm). Two heads
# share one GPU; each starts from its old head (--init_from) as a new run.
#
# Usage:
#   PAIR=sad    sbatch experiments/jacs_revision/slurm/submit_energy_finetune.sbatch
#   PAIR=direct sbatch experiments/jacs_revision/slurm/submit_energy_finetune.sbatch
#
# The density is rebuilt as it was fine-tuned (charge constraint; positive
# coefficients for direct), the model whose ε_ρ the letter reports.
#SBATCH --job-name=energy-ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --partition=a100_chemistry
#SBATCH --output=logs/energy_ft_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${PAIR:?set PAIR=sad or PAIR=direct}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=offline TORCHDYNAMO_DISABLE=1
BASE="@config/training/water_energy_base.txt"
DIRECT="@config/training/water_direct_multiphase_overlay.txt"
TUNE=(--learning_rate=2e-4 --max_steps=20000 --integral_constraint=coeffs_in_coeffs_net)

launch() {  # name theory-config load_from init_from positive_coeffs [overlay]
  local name=$1 cfg=$2 density=$3 init=$4 pos=$5 overlay=${6:-}
  # Resume: if this head already has a run with a checkpoint, continue it (weights, optimizer, step,
  # data split and its saved arguments) instead of starting again from --init_from.
  local resume=() last
  last=$(ls -dt results/revision/${name}_energy_ft/20*/checkpoints/latest_checkpoint.pth 2>/dev/null | head -1)
  [ -n "$last" ] && resume=(--restart="$(dirname "$(dirname "$last")")") && echo "$name: resuming ${resume[0]#--restart=}"
  ./.venv/bin/python -B run.py train --mode energy "$BASE" "$cfg" ${overlay:+"$overlay"} \
    --args_file_name="${name}_energy_ft" --save_dir="results/revision/${name}_energy_ft" \
    --load_from="$density" --init_from="$init" --positive_coeffs="$pos" "${TUNE[@]}" "${resume[@]}" \
    > "logs/energy_ft_${SLURM_JOB_ID}_${name}.log" 2>&1 &
  pids+=($!)
}
pids=()
case "$PAIR" in
  sad)
    launch water_pbe_d4_avdz @config/training/water_energy_pbe.txt \
      results/revision/water_pbe_d4_avdz/2026-08-26_DQX1HUW0 \
      results/revision/water_pbe_d4_avdz_energy/2026-09-14_Dw6lXobF False
    launch water_wb97mv_def2tzvpd @config/training/water_energy_wb97mv.txt \
      results/revision/water_wb97mv_def2tzvpd/2026-08-26_itJSxLYq \
      results/revision/water_wb97mv_def2tzvpd_energy/2026-09-14_w837jxPx False ;;
  direct)
    launch water_direct_pbe_d4_avdz @config/training/water_energy_pbe.txt \
      results/revision/water_direct_pbe_d4_avdz/2026-08-26_jpx8oZi5 \
      results/revision/water_direct_pbe_d4_avdz_energy/2026-09-14_tSBmCFsJ True "$DIRECT"
    launch water_direct_wb97mv_def2tzvpd @config/training/water_energy_wb97mv.txt \
      results/revision/water_direct_wb97mv_def2tzvpd/2026-08-26_SebXjydH \
      results/revision/water_direct_wb97mv_def2tzvpd_energy/2026-09-14_4MBOa5kM True "$DIRECT" ;;
  *) echo "unknown PAIR $PAIR"; exit 2 ;;
esac
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "done, status $status"
exit $status
