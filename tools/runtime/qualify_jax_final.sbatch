#!/usr/bin/env bash
# Freeze the source tree before submission; reference bundles remain separate.
#SBATCH --job-name=densnet-jax-final
#SBATCH --partition=cpu-2h
#SBATCH --exclude=head015
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=scratch/jax_pipeline/final-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source_snapshot=${DENSNET_JAX_SOURCE_SNAPSHOT:?Set the frozen source snapshot directory}
platform=${DENSNET_JAX_PLATFORM:-cpu}
export EQUIV_DENS_PROJECT_ROOT="$source_snapshot"
export DENSNET_NUM_THREADS=2
sha256sum --check "$source_snapshot/source.sha256" > "scratch/jax_pipeline/final-${SLURM_JOB_ID}-source-check.txt"
if [[ "$platform" == cuda ]]; then
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
  suite=("$source_snapshot/tests/jax_pipeline/test_pipeline.py"
         "$source_snapshot/tests/jax_pipeline/test_training.py"
         "$source_snapshot/tests/jax_pipeline/test_host.py"
         "$source_snapshot/tests/jax_pipeline/test_trainer.py"
         "$source_snapshot/tests/jax_pipeline/test_loss_boundaries.py"
         "$source_snapshot/tests/jax_pipeline/test_optional_pipeline.py"
         "$source_snapshot/tests/jax_pipeline/test_md_preparation.py"
         "$source_snapshot/tests/jax_pipeline/test_native_md.py")
else
  suite=("$source_snapshot/tests/jax_pipeline")
fi
tools/runtime/run.sh jax env JAX_PLATFORMS="$platform" python -c \
  'import jax,numpy,scipy,pyscf,ase,dftpy; print("JAX_DEVICES",jax.devices()); print("VERSIONS",jax.__version__,numpy.__version__,scipy.__version__,pyscf.__version__,ase.__version__,dftpy.__version__)'
tools/runtime/run.sh jax env JAX_PLATFORMS="$platform" \
  python -m pytest "${suite[@]}" -q -x \
  --junitxml="scratch/jax_pipeline/final-${SLURM_JOB_ID}.xml"
sha256sum --check "$source_snapshot/source.sha256" >> "scratch/jax_pipeline/final-${SLURM_JOB_ID}-source-check.txt"
