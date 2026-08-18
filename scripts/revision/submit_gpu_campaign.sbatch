#!/bin/bash
#SBATCH --job-name=densnet-rev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=l40s_public
#SBATCH --output=results/revision/gpu_campaign/slurm-%x-%j.out
# TASK=pbe0|ethanol-md|thiophene-md|water|quick|all
set -euo pipefail
mkdir -p results/revision/gpu_campaign
cd "${SLURM_SUBMIT_DIR:-.}"
export ALLOW_GL056="${ALLOW_GL056:-1}"
export TASK="${TASK:-${1:-all}}"
bash scripts/revision/run_gpu_campaign.sh "${TASK}"
