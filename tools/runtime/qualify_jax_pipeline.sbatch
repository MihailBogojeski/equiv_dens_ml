#!/usr/bin/env bash
#SBATCH --job-name=densnet-jax-qualify
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=scratch/jax_pipeline/qualification-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export DENSNET_NUM_THREADS=2
flock /tmp/densnet-jax-migration-numerics.lock \
  tools/runtime/run.sh jax env JAX_PLATFORMS="${DENSNET_JAX_PLATFORM:-cpu}" \
    python -m pytest tests/jax_pipeline/test_pipeline.py \
    tests/jax_pipeline/test_training.py tests/jax_pipeline/test_host.py -q -x \
    --junitxml="scratch/jax_pipeline/qualification-${SLURM_JOB_ID}.xml"
