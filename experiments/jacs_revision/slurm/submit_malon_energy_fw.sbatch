#!/bin/bash
# Malonaldehyde energy heads with the paper's force-weighted loss (energy 0.001 / forces 0.999, as the published
# ethanethiol, resorcinol and polythiophene heads) instead of the revision's 1.0 / 0.99. Fresh heads, normalize_en=0,
# 300k steps; lr patience 10 validations (5k steps, the paper's in steps) so a noisy 80-frame validation does not stop
# training early, EMA from epoch 300 as in the paper. Output: results/revision/malon_energy_fw/. Both theories share
# one GPU. Resumable: re-submit (or chain) to continue; the density stage is copied from the local root, read only.
#SBATCH --job-name=malon-en-fw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/malon_energy_fw_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 WANDB_MODE=offline
OUT=malon_energy_fw
for th in pbe_d4_avdz wb97mv_def2tzvpd; do
  src=results/revision/malon_energy_v2/malonaldehyde_$th; dst=results/revision/$OUT/malonaldehyde_$th
  if [ ! -d "$dst/density" ]; then
    [ -f "$src/density.done" ] || { echo "$th: local density stage not finished"; exit 2; }
    d=$(ls -d "$src"/density/20* | head -1); name=$(basename "$d")
    mkdir -p "$dst/density/$name.copying/checkpoints"
    cp "$d"/args.txt "$d"/best_*.pth "$dst/density/$name.copying/"
    cp "$d"/checkpoints/latest_checkpoint.pth "$d"/checkpoints/best_*.pth "$dst/density/$name.copying/checkpoints/" 2>/dev/null
    mv "$dst/density/$name.copying" "$dst/density/$name"
    touch "$dst/density.done"
    echo "{\"seeded_from\": \"$src\"}" > "$dst/source.json"
  fi
done
FW=(--train-arg=--energy_weight=0.001 --train-arg=--forces_weight=0.999 --train-arg=--max_steps=300000
    --train-arg=--decay_patience=10 --train-arg=--ema_start_epoch=300)
pids=()
for th in pbe_d4_avdz wb97mv_def2tzvpd; do
  ./.venv/bin/python -B experiments/jacs_revision/campaign/run_malon_energy_v2.py --theory $th --out-name $OUT "${FW[@]}" &
  pids+=($!)
done
st=0; for p in "${pids[@]}"; do wait $p || st=1; done
exit $st
