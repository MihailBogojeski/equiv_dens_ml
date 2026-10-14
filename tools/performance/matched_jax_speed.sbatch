#!/usr/bin/env bash
#SBATCH --job-name=densnet-jax-speed
#SBATCH --partition=gpu-2h
#SBATCH --time=01:45:00
#SBATCH --gpus=40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
cd "${EQUIV_DENS_PROJECT_ROOT:-/home/johannes/equiv_dens_ml}"
report=${1:?report directory}
shift
mkdir -p "$report" /temp/johannes
hostname > "$report/node.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$report/job-id.txt"
git rev-parse HEAD > "$report/source-head.txt"
git -C ../research-infra rev-parse HEAD > "$report/shared-head.txt"
find src -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum > "$report/source-files.sha256"
sha256sum benchmarks/jax_backend_speed.py tools/performance/matched_jax_speed.py tools/performance/matched_jax_speed.sbatch > "$report/instruments.sha256"
cp /home/johannes/cont/densnet-torch-2026-10-11.sif /temp/johannes/densnet-jax-speed-torch-${SLURM_JOB_ID}.sif
cp /home/johannes/cont/densnet-jax-pipeline-2026-10-11.sif /temp/johannes/densnet-jax-speed-jax-${SLURM_JOB_ID}.sif
export DENSNET_TORCH_IMAGE=/temp/johannes/densnet-jax-speed-torch-${SLURM_JOB_ID}.sif
export DENSNET_JAX_IMAGE=/temp/johannes/densnet-jax-speed-jax-${SLURM_JOB_ID}.sif
trap 'rm -f "$DENSNET_TORCH_IMAGE" "$DENSNET_JAX_IMAGE"' EXIT
sha256sum "$DENSNET_TORCH_IMAGE" "$DENSNET_JAX_IMAGE" > "$report/images.sha256"
export DENSNET_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=true JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=default
# Each fresh arm uses its own cache directory; repeated pairs retain cold JIT cost.
export DENSNET_CACHE_ROOT=/temp/johannes/jax-speed-cache-${SLURM_JOB_ID}
exec 9>/tmp/johannes_measurement.lock
flock -x 9
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?}" --query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total --format=csv > "$report/gpu-before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$report/processes-before.csv"
python3 - "$report" <<'PY'
import csv,sys
from pathlib import Path
p=Path(sys.argv[1]); g=list(csv.DictReader((p/'gpu-before.csv').open())); a=list(csv.DictReader((p/'processes-before.csv').open()))
uuids={r[' uuid'].strip() for r in g}
if any(r['gpu_uuid'].strip() in uuids for r in a) or any(int(r[' utilization.gpu [%]'].split()[0])>0 for r in g):
    raise SystemExit('Busy allocated GPU: timing refused')
PY
printf '%s JAX matched speed start\n' "$(date -Is)" >> /tmp/agents-ressource-usage-${SLURM_JOB_ID}.txt
printf '%s %s JAX matched speed\n' "$(date -Is)" "$SLURM_JOB_ID" >> /tmp/johannes_measurement.lock.holders
# Host Python only orchestrates and compares small completed NumPy arrays.
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
set +e
python3 -u tools/performance/matched_jax_speed.py --output "$report" "$@" > "$report/run.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$report/exit-code.txt"
sha256sum -c "$report/source-files.sha256" > "$report/source-verification-after.txt"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,utilization.gpu,memory.used --format=csv > "$report/gpu-after.csv"
printf '%s JAX matched speed done exit=%s\n' "$(date -Is)" "$result" >> /tmp/agents-ressource-usage-${SLURM_JOB_ID}.txt
exit "$result"
