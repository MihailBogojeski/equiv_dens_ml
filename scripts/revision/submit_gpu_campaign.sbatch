#!/bin/bash
#SBATCH --job-name=densnet-rev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=results/revision/gpu_campaign/slurm-%j.out
# Do not submit this to gl056. Pick a free Greene GPU partition/account.
# Example: sbatch --partition=gpu --account=YOUR_ACCOUNT scripts/revision/submit_gpu_campaign.sbatch

set -euo pipefail
mkdir -p results/revision/gpu_campaign
if [[ "$(hostname -s)" == gl056* ]]; then
  echo "Allocated gl056; aborting so we do not steal occupied GPUs."
  exit 2
fi
cd "${SLURM_SUBMIT_DIR:-.}"
bash scripts/revision/run_gpu_campaign.sh
