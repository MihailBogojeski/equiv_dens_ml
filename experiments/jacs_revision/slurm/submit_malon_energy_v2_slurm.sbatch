#!/bin/bash
# Independent cluster copy of the malonaldehyde energy-head runs (results/revision/malon_energy_v2_slurm), alongside the
# local-GPU runs in results/revision/malon_energy_v2. Fresh energy heads (normalize_en=0); the finished density stage is
# copied (checkpoints only) from the local root, which never writes to it again. Resumable: re-submit to continue.
#SBATCH --job-name=malon-en-slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/malon_energy_v2_slurm_%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 WANDB_MODE=offline
OUT=malon_energy_v2_slurm
for th in pbe_d4_avdz wb97mv_def2tzvpd; do
  src=results/revision/malon_energy_v2/malonaldehyde_$th; dst=results/revision/$OUT/malonaldehyde_$th
  if [ ! -d "$dst/density" ]; then
    [ -f "$src/density.done" ] || { echo "$th: local density stage not finished"; exit 2; }
    mkdir -p "$dst/density"
    d=$(ls -d "$src"/density/20* | head -1); name=$(basename "$d")
    mkdir -p "$dst/density/$name.copying/checkpoints"
    cp "$d"/args.txt "$d"/best_*.pth "$dst/density/$name.copying/"
    cp "$d"/checkpoints/latest_checkpoint.pth "$d"/checkpoints/best_*.pth "$dst/density/$name.copying/checkpoints/" 2>/dev/null
    mv "$dst/density/$name.copying" "$dst/density/$name"
    touch "$dst/density.done"
    echo "{\"seeded_from\": \"$src\"}" > "$dst/source.json"
  fi
done
./.venv/bin/python -B experiments/jacs_revision/campaign/run_malon_energy_v2.py --theory pbe_d4_avdz --out-name $OUT &
p1=$!
./.venv/bin/python -B experiments/jacs_revision/campaign/run_malon_energy_v2.py --theory wb97mv_def2tzvpd --out-name $OUT &
p2=$!
st=0; wait $p1 || st=1; wait $p2 || st=1
exit $st
