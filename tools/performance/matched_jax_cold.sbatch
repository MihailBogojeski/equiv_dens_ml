#!/usr/bin/env bash
#SBATCH --job-name=densnet-jax-cold
#SBATCH --partition=gpu-2h
#SBATCH --time=00:20:00
#SBATCH --gpus=40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
set -euo pipefail
cd "${EQUIV_DENS_PROJECT_ROOT:?frozen baseline source required}"
report=${1:?report directory}
fixture=${2:?portable wide fixture}
mkdir -p "$report" /temp/johannes
hostname > "$report/node.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$report/job-id.txt"
git rev-parse HEAD > "$report/source-head.txt"
find src -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum > "$report/source-files.sha256"
sha256sum benchmarks/jax_backend_speed.py benchmarks/jax_backend_cold.py tools/performance/matched_jax_cold.sbatch > "$report/instruments.sha256"
local_image=/temp/johannes/densnet-jax-cold-${SLURM_JOB_ID}.sif
cp /home/johannes/cont/densnet-jax-pipeline-2026-10-11.sif "$local_image"
trap 'rm -f "$local_image"' EXIT
export DENSNET_JAX_IMAGE=$local_image DENSNET_NUM_THREADS=1
sha256sum "$local_image" > "$report/images.sha256"
exec 9>/tmp/johannes_measurement.lock
flock -x 9
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?}" --query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total --format=csv > "$report/gpu-before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$report/processes-before.csv"
python3 - "$report" <<'PY'
import csv,sys
from pathlib import Path
p=Path(sys.argv[1]);g=list(csv.DictReader((p/'gpu-before.csv').open()));a=list(csv.DictReader((p/'processes-before.csv').open()));uids={r[' uuid'].strip() for r in g}
if any(r['gpu_uuid'].strip() in uids for r in a) or any(int(r[' utilization.gpu [%]'].split()[0])>0 for r in g):raise SystemExit('Busy allocated GPU: cold probe refused')
PY
for mode in ef density_update energy_force_update calculator; do
 cache=/temp/johannes/densnet-jax-cold-cache-${SLURM_JOB_ID}/${mode}
 mkdir -p "$cache"
 tools/runtime/run.sh jax env SLURM_JOB_ID="$SLURM_JOB_ID" JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  JAX_COMPILATION_CACHE_DIR="$cache" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=default \
  python -u -m benchmarks.jax_backend_cold --backend jax --fixture "$fixture" --mode "$mode" \
  --warmup 2 --iterations 3 --output "$report/wide-${mode}-cold.json" > "$report/wide-${mode}-cold.log" 2>&1
 done
sha256sum -c "$report/source-files.sha256" > "$report/source-verification-after.txt"
printf '0\n' > "$report/exit-code.txt"
