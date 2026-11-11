#!/bin/bash
# Train one water energy head with the v3 recipe, then score it. Runs locally or inside an sbatch.
#
#   CUDA_VISIBLE_DEVICES=1 experiments/jacs_revision/slurm/train_energy_head.sh sad pbe_d4_avdz
#   experiments/jacs_revision/slurm/train_energy_head.sh direct wb97mv_def2tzvpd [extra train args...]
#
# Recipe (the paper's final energy stage, adapted): a fresh energy head (no --init_from) on the frozen,
# charge-constrained density network; normalize_en=0, which keeps the head local and additive.
# A normalize_en=2 head is not a valid start for normalize_en=0: the flag also switches the √½ residual
# scaling, rbf scale, 1/num_neighbours and PairMixing norm factors, so the loaded weights compute a
# different function. AMSGrad lr 1e-3, halved after 5 flat validations, EMA from epoch 30 (~4.7k steps).
# Loss weights stay 1.0 (energy) / 0.99 (forces); the paper used 0.001/0.999.
#
# Resume: if SAVE_ROOT already holds a run with a checkpoint, it continues that run with its saved
# arguments (--restart overrides the command line). A changed recipe needs a new SAVE_ROOT.
# Env: SAVE_ROOT (default results/revision/water_[direct_]<theory>_energy_v3), SCORE=0 skips scoring,
#      DENSITY overrides the frozen density run (its --cutoff must then be passed as an extra arg),
#      LOG_TAG names the log file, SCORE_OUT the scoring output folder.
set -uo pipefail
cd "$(dirname "$0")/../../.."
kind=${1:?sad|direct}; th=${2:?pbe_d4_avdz|wb97mv_def2tzvpd}; shift 2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} WANDB_MODE=offline TORCHDYNAMO_DISABLE=1
# torch 2.13 routes some CUDA bmm calls to Triton, whose runtime helper is gcc-built against Python.h on first use;
# the compute nodes have no python3-devel, so point gcc at a copy of the 3.12 headers (only needed when the
# Triton cache is empty, e.g. after a cache clean-up; 2026-10-17).
export CPATH=${CPATH:-/scratch/mh7373/tools/python3.12-include}

case "$kind/$th" in
  sad/pbe_d4_avdz)         density=results/revision/water_pbe_d4_avdz/2026-08-26_DQX1HUW0 ;;
  sad/wb97mv_def2tzvpd)    density=results/revision/water_wb97mv_def2tzvpd/2026-08-26_itJSxLYq ;;
  direct/pbe_d4_avdz)      density=results/revision/water_direct_pbe_d4_avdz/2026-08-26_jpx8oZi5 ;;
  direct/wb97mv_def2tzvpd) density=results/revision/water_direct_wb97mv_def2tzvpd/2026-08-26_SebXjydH ;;
  *) echo "unknown head $kind/$th"; exit 2 ;;
esac
density=${DENSITY:-$density}  # e.g. a cutoff-sweep density network; pass its --cutoff as an extra arg
cfg=@config/training/water_energy_pbe.txt
[ "$th" = wb97mv_def2tzvpd ] && cfg=@config/training/water_energy_wb97mv.txt
pre=""; overlay=(); pos=False
[ "$kind" = direct ] && pre=direct_ && overlay=(@config/training/water_direct_multiphase_overlay.txt) && pos=True
name=water_${pre}${th}
SAVE_ROOT=${SAVE_ROOT:-results/revision/${name}_energy_v3}
log=logs/energy_v3_${LOG_TAG:-${SLURM_JOB_ID:-local}}_${name}.log

RECIPE=(--normalize_en=0 --integral_constraint=coeffs_in_coeffs_net --positive_coeffs=$pos
        --learning_rate=1e-3 --decay_factor=0.5 --decay_patience=5 --stop_at_learning_rate=1e-5
        --max_steps=100000 --validation_interval=1000 --checkpoint_interval=1000
        --ema_start_epoch=30 --energy_weight=1.0 --forces_weight=0.99 --train_eval_size=250)

resume=()
last=$(ls -dt "$SAVE_ROOT"/20*/checkpoints/latest_checkpoint.pth 2>/dev/null | head -1)
[ -n "$last" ] && resume=(--restart="$(dirname "$(dirname "$last")")") && echo "$name: resuming ${resume[0]#--restart=}"

echo "$name: training -> $SAVE_ROOT (log $log, GPU ${CUDA_VISIBLE_DEVICES:-all})"
./.venv/bin/python -B run.py train --mode energy @config/training/water_energy_base.txt "$cfg" "${overlay[@]}" \
  --args_file_name="${name}_energy_v3" --save_dir="$SAVE_ROOT" --load_from="$density" \
  "${RECIPE[@]}" "$@" "${resume[@]}" > "$log" 2>&1
status=$?
echo "$name: training exited with status $status"
[ "$status" -ne 0 ] || [ "${SCORE:-1}" = 0 ] && exit $status

run=$(dirname "$(dirname "$(ls -dt "$SAVE_ROOT"/20*/checkpoints/latest_checkpoint.pth | head -1)")")
out=${SCORE_OUT:-results/revision/energy_forces_v2/${name}_v3}
mkdir -p "$out"
./.venv/bin/python -B experiments/jacs_revision/analysis/energy_head_locality.py --run "$run" --theory "$th" \
  --out "$out/locality.json" > "${log%.log}_locality.log" 2>&1
./.venv/bin/python -B experiments/jacs_revision/analysis/score_energy_forces.py --run "$run" --theory "$th" \
  --out "$out" --batch 8 --check-single 8 > "${log%.log}_score.log" 2>&1
echo "$name: scored -> $out (status $?)"
