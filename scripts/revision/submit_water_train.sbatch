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
#   RUN is one of: water, malonaldehyde, cutoff_4, cutoff_5, cutoff_6, cutoff_8
#SBATCH --job-name=water-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
# a100_chemistry first: it is this account's own partition, it had 43 of 172
# GPUs free while the shared queues were full, and its A100s carry 80 GB against
# the L40S's 46 GB. A probe job started there in under six seconds.
#
# Naming l40s_public alone is what left these trainings pending for a day.
# l40s_public was at 263 of 272 GPUs with the rest PLANNED for higher-priority
# users, and h200_public's partition QOS caps everyone at 24 GPUs in aggregate.
# With this account's fair-share at 0.006 the pinned queue was never going to
# clear -- the same trap the labelling arrays hit with --partition=cs.
#
# --gres stays generic (gpu:1, above) so one script fits all three partitions.
#SBATCH --partition=a100_chemistry,h200_public,l40s_public
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
scripts/revision/assemble_water_labels.sh "$THEORY" "$MIN_COMPLETE"

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
resume_dir=$(scripts/revision/latest_run_dir.sh "$save_dir")
if [[ -n "$resume_dir" ]]; then
  echo "resuming ${name} from ${resume_dir}"
  extra+=(--restart="$resume_dir")
fi

./.venv/bin/python run.py train "${cfg[@]}" "${extra[@]}" 2>&1 | tee -a "logs/train_${name}.log"

# Only reached when run.py returns cleanly, which under `set -e` means the
# training actually ran to its stopping criterion rather than being cut off at
# the walltime. The watchdog re-submits any run without this marker, so the
# distinction is what stops a truncated run from being mistaken for a finished
# one.
touch "${save_dir}/.training_done"
echo "$(date -Is) ${name} finished"
