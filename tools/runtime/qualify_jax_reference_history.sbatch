#!/usr/bin/env bash
#SBATCH --job-name=densnet-jax-history
#SBATCH --partition=cpu-2h
#SBATCH --exclude=head015
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=scratch/jax_pipeline/history-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export DENSNET_NUM_THREADS=2
tools/runtime/run.sh torch python tests/jax_pipeline/generate_pipeline_reference.py \
  --case batch --output scratch/jax_pipeline/pipeline-batch
tools/runtime/run.sh torch python tests/jax_pipeline/generate_pipeline_reference.py \
  --case tiny --source-root /home/johannes/equiv_dens_ml-paper-reference \
  --output scratch/jax_pipeline/pipeline-paper-tiny
for reference in pipeline-batch pipeline-paper-tiny; do
  flock /tmp/densnet-jax-migration-numerics.lock \
    tools/runtime/run.sh jax env JAX_PLATFORMS=cpu \
      DENSNET_JAX_PIPELINE_REFERENCE="scratch/jax_pipeline/$reference" \
      python -m pytest tests/jax_pipeline/test_pipeline.py -q -x \
      --junitxml="scratch/jax_pipeline/history-${SLURM_JOB_ID}-${reference}.xml"
done
