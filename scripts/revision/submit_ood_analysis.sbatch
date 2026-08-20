#!/bin/bash
# The analysis chain for one level of theory, as its own job.
#
# It used to run inline in the watchdog loop, which does not fit: evaluating a
# tier is a forward pass per structure and there are roughly nine hundred of
# them across the tiers, measured at about six minutes per twenty structures on
# a core -- some four hours. Inline, that is four hours in which no shard is
# re-submitted and no training is launched, on a loop whose whole purpose is to
# do those two things every half hour.
#
# CPU rather than GPU deliberately. The account's GPU allowance is fully
# consumed by other work, so a GPU request here would queue behind -- and ahead
# of the training runs, which are what the analysis is waiting for in the first
# place.
#
# Usage:  THEORY=... sbatch scripts/revision/submit_ood_analysis.sbatch
#SBATCH --job-name=ood-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --output=logs/ood_analysis_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

THEORY=${THEORY:-wb97mv_def2tzvpd}

# The login shell exports a CUDA preload that does not exist on the CPU
# partition, and every child process prints a loader error before doing
# anything. See _clean_env in qm7x_orca_common.py for the same problem.
unset LD_PRELOAD

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export CUDA_VISIBLE_DEVICES=""
mkdir -p logs

echo "[$(date -Is)] ood analysis for ${THEORY} on $(hostname)"
scripts/revision/run_ood_analysis.sh "$THEORY"
echo "[$(date -Is)] ood analysis for ${THEORY} finished"
