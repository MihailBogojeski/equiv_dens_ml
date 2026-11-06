#!/bin/bash
# One DenSNet training run of the water campaign, resumable across walltimes.
#
# One run per job rather than a list of runs in one job. At 300k steps a single
# training does not reliably finish inside a 24 h window, so a job holding six
# of them in sequence would be cut off partway through the second and the rest
# would never start -- silently, since the job itself exits cleanly at the wall.
# Split up, each run resumes from its own checkpoint on re-submission, which is
# the same recovery the labelling arrays use.
#
# Not submitted with --dependency on the labelling arrays either: the watchdog
# re-submits those as tasks are killed, so the ids present at submission are not
# the ones that finish the campaign. The watchdog launches these once
# campaign_status.py reports nothing outstanding.
#
# Usage:  THEORY=... RUN=... sbatch scripts/revision/submit_water_train.sbatch
#   RUN is one of: water, malonaldehyde, water_direct, cutoff_4, cutoff_5,
#   cutoff_6, cutoff_8, water_n50, water_direct_n50, water_curve,
#   water_direct_curve (and the other N_train sizes 125/250/500/1250).
#SBATCH --job-name=water-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
# Keep revision training on A100/H200. l40s_public is reserved for the
# tps-cofolding production campaign and its long-lived wrap allocation.
# --gres stays generic (gpu:1) so this script works on both partitions.
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/water_train_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

THEORY=${THEORY:-wb97mv_def2tzvpd}
RUN=${RUN:-water}
MIN_COMPLETE=${MIN_COMPLETE:-0.6}

# A note on the configs these stacks build, since the config files themselves
# cannot carry it -- argparse reads them with fromfile_prefix_chars, so every
# line is an argument and a `#` comment is a parse error.
#
# --np_dataset_test points at the *in-distribution* test set, not at an OOD
# tier. The OOD numbers that go in the paper come from csh_evaluate.py, run per
# tier by run_ood_analysis.sh once the model exists; run.py's own test pass is
# incidental to that. It used to point at the ice tier, which meant training
# could not start until the slowest 24 h arrays in the campaign had finished --
# days after the train and validation labels it actually optimises against were
# ready on disk.

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p logs

# Idempotent and cheap next to the training itself, and running it here rather
# than in a separate job removes a dependency edge; the assembler writes
# atomically so concurrent training jobs cannot tear each other's output.
scripts/revision/water/assemble_water_labels.sh "$THEORY" "$MIN_COMPLETE"

case "$THEORY" in
  wb97mv_def2tzvpd) base=(@config/training/water_wb97mv_001.txt) ; overlay=() ;;
  # Overlays on the wB97M-V configs so the two levels differ only in their
  # labels: the PBE-vs-hybrid comparison is only about the reference data if
  # every architecture and optimisation setting is identical.
  pbe_d4_avdz) base=(@config/training/water_wb97mv_001.txt)
               overlay=(@config/training/water_pbe_orca_overlay.txt) ;;
  *) echo "no training config registered for theory ${THEORY}"; exit 1 ;;
esac

case "$RUN" in
  water)
    cfg=("${base[@]}" "${overlay[@]}")
    name="water_${THEORY}"
    ;;
  malonaldehyde)
    cfg=(@config/training/malonaldehyde_pt_001.txt)
    if [[ "$THEORY" == "pbe_d4_avdz" ]]; then
      cfg+=(@config/training/malonaldehyde_pbe_orca_overlay.txt)
    fi
    name="malonaldehyde_${THEORY}"
    ;;
  malonaldehyde_direct)
    # Direct-ML counterpart of `malonaldehyde`, as water_direct is of water.
    cfg=(@config/training/malonaldehyde_pt_001.txt)
    if [[ "$THEORY" == "pbe_d4_avdz" ]]; then
      cfg+=(@config/training/malonaldehyde_pbe_orca_overlay.txt)
    fi
    cfg+=(@config/training/water_direct_overlay.txt)
    name="malonaldehyde_direct_${THEORY}"
    ;;
  water_direct)
    # R3.4: the same model predicting the total density instead of the SAD
    # correction. Everything but --remove_atom_density is held fixed, so the
    # difference between this run and `water` is the delta-learning target and
    # nothing else. Reported against analyze_density_metrics.py, which measures
    # how much of the density the prior already supplies.
    cfg=("${base[@]}" "${overlay[@]}" @config/training/water_direct_overlay.txt)
    name="water_direct_${THEORY}"
    ;;
  cutoff_*)
    # R3.5 asks how far the message passing has to reach to capture a
    # hydrogen-bond network. That is a question about the architecture, not the
    # functional, so it is asked at PBE, which costs an order of magnitude less
    # on the same geometries.
    if [[ "$THEORY" != "pbe_d4_avdz" ]]; then
      echo "cutoff sweep runs at PBE only, not ${THEORY}"
      exit 0
    fi
    cut=${RUN#cutoff_}
    cfg=("${base[@]}" "${overlay[@]}" "@config/training/water_clusters_cutoff_${cut}.txt")
    name="water_pbe_orca_cutoff_${cut}"
    ;;
  water_direct_n*|water_n*)
    # Data-efficiency arm: stratified train npy of size N, same val/test.
    # Distinct save_dir so these cannot resume the full-N campaign.
    if [[ "$RUN" == water_direct_n* ]]; then
      n=${RUN#water_direct_n}
      direct=1
    else
      n=${RUN#water_n}
      direct=0
    fi
    stem="datasets/revision/water_labels/learning_curves/water_train_n${n}_${THEORY}"
    if [[ ! -f "${stem}_base.npy" || ! -f "${stem}.npy" ]]; then
      echo "missing learning-curve train npy ${stem}*.npy" >&2
      echo "run scripts/revision/water/make_learning_curve_splits.py first" >&2
      exit 1
    fi
    cfg=("${base[@]}" "${overlay[@]}"
         --np_dataset="${stem}_base.npy"
         --dens_dataset="${stem}.npy"
         --num_train="$n")
    if [[ "$direct" -eq 1 ]]; then
      cfg+=(@config/training/water_direct_overlay.txt)
      name="water_direct_n${n}_${THEORY}"
    else
      name="water_n${n}_${THEORY}"
    fi
    ;;
  water_direct_curve|water_curve)
    # Full-N run that keeps every checkpoint so OOD-vs-step can be scored.
    # Separate save_dir from the campaign water_${THEORY} job.
    cfg=("${base[@]}" "${overlay[@]}" @config/training/water_learning_curve_ckpt.txt)
    if [[ "$RUN" == water_direct_curve ]]; then
      cfg+=(@config/training/water_direct_overlay.txt)
      name="water_direct_${THEORY}_curve"
    else
      name="water_${THEORY}_curve"
    fi
    ;;
  *)
    echo "unknown RUN ${RUN}"
    exit 1
    ;;
esac

# R3.3 wants an uncertainty band, which needs several independent fits of the
# same data. INIT_SEED varies only the weight initialisation and batch order,
# and each member gets its own save_dir so the members do not resume from each
# other's checkpoints -- which would collapse the ensemble to one model.
if [[ -n "${INIT_SEED:-}" ]]; then
  name="${name}_seed${INIT_SEED}"
  cfg+=(--init_seed="$INIT_SEED")
fi

save_dir="results/revision/${name}"
extra=(--args_file_name="$name" --save_dir="$save_dir")
# --restart wants the timestamped run directory run.py creates *inside*
# --save_dir, which is where checkpoints/ actually lives. Pointing it at the
# save_dir found nothing and started from step zero, and since 300k steps do not
# fit in one 24 h window, the run would have been re-trained from scratch on
# every re-submission and never converged -- while looking, each time, like a
# healthy training job.
resume_dir=$(scripts/revision/lib/latest_run_dir.sh "$save_dir")
if [[ -n "$resume_dir" ]]; then
  echo "resuming ${name} density stages from ${resume_dir}"
  extra+=(--restart="$resume_dir")
fi

# Paper recipe: multiphase density (DF + AFE + constraint fine-tune) only.
# Stage-D energy training is submit_water_train_energy.sbatch.
cfg+=(@config/training/water_multiphase_overlay.txt)
if [[ "$RUN" == "water_direct" || "$RUN" == water_direct_n* || "$RUN" == "water_direct_curve" || "$RUN" == "malonaldehyde_direct" ]]; then
  cfg+=(@config/training/water_direct_multiphase_overlay.txt)
fi

./.venv/bin/python run.py train --mode multiphase "${cfg[@]}" "${extra[@]}" \
  2>&1 | tee -a "logs/train_${name}.log"

# Only reached when run.py returns cleanly, which under `set -e` means the
# training actually ran to its stopping criterion rather than being cut off at
# the walltime. The watchdog re-submits any run without this marker, so the
# distinction is what stops a truncated run from being mistaken for a finished
# one.
density_run=$(scripts/revision/lib/latest_run_dir.sh "$save_dir")
if [[ -z "$density_run" ]]; then
  echo "density multiphase finished but no checkpoint under ${save_dir}" >&2
  exit 1
fi
echo "$density_run" > "${save_dir}/.density_run_dir"
touch "${save_dir}/.density_training_done"
echo "$(date -Is) ${name} density stages finished (${density_run})"
