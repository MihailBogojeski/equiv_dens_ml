#!/usr/bin/env bash
# Queue reviewer GPU jobs on Greene. They will pend if qos gpu=4 is already full.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p results/revision/gpu_campaign
SBATCH="scripts/revision/submit_gpu_campaign.sbatch"

submit() {
  local task="$1"
  local name="$2"
  sbatch --job-name="${name}" --export=ALL,TASK="${task}",ALLOW_GL056=1 \
    --account=torch_pr_283_chemistry --qos=gpu48 --partition=l40s_public \
    --gres=gpu:l40s:1 --cpus-per-task=8 --mem=64G --time=48:00:00 \
    --output=results/revision/gpu_campaign/slurm-%x-%j.out \
    --wrap="cd ${ROOT} && export ALLOW_GL056=1 TASK=${task} && bash scripts/revision/run_gpu_campaign.sh ${task}"
}

echo "Submitting revision GPU jobs from $(hostname) ..."
submit pbe0 dens-pbe0
submit ethanol-md dens-etohmd
submit thiophene-md dens-thio2
submit water dens-water
echo "Queue:"
squeue -u "${USER}" -o '%.10i %.12P %.18j %.8T %.10M %R %b' || true
