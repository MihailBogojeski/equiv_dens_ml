#!/usr/bin/env bash
# Project-specific clean runtime; keeps Torch and JAX in separate processes.
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ $# -lt 2 ]]; then
  echo 'Usage: tools/runtime/run.sh torch|jax COMMAND [ARGS...]' >&2
  exit 2
fi
case "$1" in
  torch) image=${DENSNET_TORCH_IMAGE:-$HOME/cont/densnet-torch-2026-10-11.sif} ;;
  jax)
    default_jax_image=$HOME/cont/densnet-jax-pipeline-2026-10-11.sif
    [[ -f "$default_jax_image" ]] || default_jax_image=$HOME/cont/densnet-jax-2026-10-11.sif
    image=${DENSNET_JAX_IMAGE:-$default_jax_image}
    ;;
  *) echo 'Runtime must be torch or jax' >&2; exit 2 ;;
esac
shift
[[ -f "$image" ]] || { echo "Runtime image missing: $image" >&2; exit 1; }
runtime_flags=(--cleanenv)
if command -v nvidia-smi >/dev/null 2>&1; then runtime_flags+=(--nv); fi
for bind_path in /temp /home/space; do
  [[ ! -d "$bind_path" ]] || runtime_flags+=(--bind "$bind_path:$bind_path")
done
shared_datasets=${DENSNET_SHARED_DATASETS-/home/ml-dft/equiv_dens/datasets}
if [[ -n "$shared_datasets" && -d "$shared_datasets" ]]; then
  runtime_flags+=(--bind "$shared_datasets:$shared_datasets:ro")
fi
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
  runtime_flags+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi
project_root=${EQUIV_DENS_PROJECT_ROOT:-$project_root}
thread_count=${DENSNET_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-2}}
runtime_flags+=(--env "PYTHONPATH=$project_root/src:$project_root:$project_root/scripts"
  --env "OMP_NUM_THREADS=$thread_count" --env "MKL_NUM_THREADS=$thread_count"
  --env "OPENBLAS_NUM_THREADS=$thread_count" --env "MAX_JOBS=$thread_count"
  --env "TORCHINDUCTOR_COMPILE_THREADS=$thread_count")
exec env -u APPTAINER_BIND -u APPTAINER_BINDPATH \
  -u SINGULARITY_BIND -u SINGULARITY_BINDPATH apptainer exec \
  "${runtime_flags[@]}" "$image" "$@"
