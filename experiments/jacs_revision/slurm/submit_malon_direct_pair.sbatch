#!/bin/bash
# Both direct malonaldehyde density trainings (PBE-D4 and wB97M-V) on one GPU.
#
# Run alone, a 9-atom training keeps an A100 near 50% busy, and Torch cancels
# jobs that average below that over 2 h (18751517 was cancelled by root after
# 3.5 h). Two processes on one GPU keep it busy and use one allocation instead
# of two. Each run resumes from its own latest checkpoint through
# submit_water_train.sbatch, so re-submitting this job continues both.
#
#   sbatch --dependency=afterany:<running single job> \
#     experiments/jacs_revision/slurm/submit_malon_direct_pair.sbatch
#SBATCH --job-name=malon-direct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public
#SBATCH --output=logs/malon_direct_pair_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SLURM_CPUS_PER_TASK=8   # each run takes half the CPUs (OMP_NUM_THREADS in the inner script)

pids=()
for theory in pbe_d4_avdz wb97mv_def2tzvpd; do
  THEORY="$theory" RUN=malonaldehyde_direct \
    bash experiments/jacs_revision/slurm/submit_water_train.sbatch \
    > "logs/malon_direct_pair_${SLURM_JOB_ID}_${theory}.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
