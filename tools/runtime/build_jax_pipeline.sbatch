#!/usr/bin/env bash
#SBATCH --job-name=densnet-jax-image
#SBATCH --partition=cpu-2h
#SBATCH --exclude=head015
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=scratch/jax_pipeline/image-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
build_root="/temp/johannes/densnet-jax-pipeline-${SLURM_JOB_ID}"
mkdir -p "$build_root/cache"
env -u APPTAINER_BIND -u APPTAINER_BINDPATH -u SINGULARITY_BIND -u SINGULARITY_BINDPATH \
  APPTAINER_TMPDIR="$build_root" APPTAINER_CACHEDIR="$build_root/cache" GOMAXPROCS=1 \
  apptainer build --fakeroot \
  --mksquashfs-args '-comp zstd -Xcompression-level 3 -mem 1500M -processors 2' \
  "$build_root/pipeline.sif" containers/jax-pipeline.def
cp "$build_root/pipeline.sif" /home/johannes/cont/.densnet-jax-pipeline-2026-10-11.sif.tmp
mv /home/johannes/cont/.densnet-jax-pipeline-2026-10-11.sif.tmp \
   /home/johannes/cont/densnet-jax-pipeline-2026-10-11.sif
sha256sum /home/johannes/cont/densnet-jax-pipeline-2026-10-11.sif \
  > scratch/jax_pipeline/pipeline-image.sha256
export DENSNET_JAX_IMAGE=/home/johannes/cont/densnet-jax-pipeline-2026-10-11.sif
tools/runtime/run.sh jax env JAX_PLATFORMS=cpu python -c \
  'import dftpy,jax,pyscf,ase; from dftpy.grid import DirectGrid; print("HOST_AND_JAX_IMPORT_OK",jax.__version__); print(DirectGrid(lattice=[[4.,0.,0.],[0.,4.,0.],[0.,0.,4.]],nr=[4,4,4]).nr)'
