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
# This asked for CPU only, on the grounds that the account's GPU allowance was
# fully consumed and a GPU request would queue ahead of the very trainings the
# analysis waits for. That premise no longer holds: a100_chemistry is this
# account's own partition and had 43 of 172 GPUs free. It matters now because
# the chain also relaxes 300 water clusters of up to 24 molecules under the
# model, which is minutes on a GPU and many hours on a core.
#
# Usage:  THEORY=... sbatch scripts/revision/submit_ood_analysis.sbatch
#SBATCH --job-name=ood-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public,l40s_public
#SBATCH --output=logs/ood_analysis_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

THEORY=${THEORY:-wb97mv_def2tzvpd}

# The login shell exports a CUDA preload that does not exist on the CPU
# partition, and every child process prints a loader error before doing
# anything. See _clean_env in qm7x_orca_common.py for the same problem.
unset LD_PRELOAD

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p logs

echo "[$(date -Is)] ood analysis for ${THEORY} on $(hostname)"
scripts/revision/run_ood_analysis.sh "$THEORY"
echo "[$(date -Is)] ood analysis for ${THEORY} finished"
