#!/usr/bin/env bash
# Independent initialized Torch oracles; run before native qualification.
#SBATCH --job-name=densnet-jax-oracles
#SBATCH --partition=cpu-2h
#SBATCH --exclude=head015
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=scratch/jax_pipeline/oracles-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export DENSNET_NUM_THREADS=2
tools/runtime/run.sh torch python tests/jax_pipeline/generate_layer_fixtures.py
tools/runtime/run.sh torch python tests/jax_pipeline/generate_pipeline_reference.py \
  --case tiny --output scratch/jax_pipeline/pipeline-tiny
tools/runtime/run.sh torch python tests/jax_pipeline/generate_training_reference.py
tools/runtime/run.sh torch python tests/jax_pipeline/generate_sparse_optimizer_reference.py
for variant in learned fixed positive constraint weighted core delta; do
  tools/runtime/run.sh torch python tests/jax_pipeline/generate_density_reference.py \
    --kind head --variant "$variant" --output "scratch/jax_pipeline/density-head-$variant"
done
tools/runtime/run.sh torch python tests/jax_pipeline/generate_density_reference.py \
  --kind df --output scratch/jax_pipeline/density-df
tools/runtime/run.sh torch python tests/jax_pipeline/generate_density_reference.py \
  --kind freeatom --output scratch/jax_pipeline/density-freeatom
tools/runtime/run.sh torch python tests/jax_pipeline/generate_density_reference.py \
  --kind expansion --output scratch/jax_pipeline/density-expansion-sp
tools/runtime/run.sh torch python tests/jax_pipeline/generate_energy_reference.py \
  --output scratch/jax-pipeline/energy-reference
tools/runtime/run.sh torch python tests/jax_pipeline/generate_functional_reference.py \
  --output scratch/jax-pipeline/functional-reference
