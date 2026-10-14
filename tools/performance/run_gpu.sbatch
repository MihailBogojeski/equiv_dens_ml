#!/usr/bin/env bash
# Serialized controlled tests/benchmarks on a short owned GPU allocation.
#SBATCH --job-name=densnet-performance
#SBATCH --partition=gpu-9m
#SBATCH --time=00:09:00
#SBATCH --gpus=40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
if [[ $# != 3 ]]; then
    echo 'Usage: sbatch run_gpu.sbatch PROJECT_ROOT REPORT_DIR COMMAND_FILE' >&2
    exit 2
fi
project_root=$1
report_dir=$2
command_file=$3
cd "$project_root"
mkdir -p "$report_dir" "/temp/$(id -un)"
local_image="/temp/$(id -un)/densnet-performance-${SLURM_JOB_ID}.sif"
trap 'rm -f "$local_image"' EXIT
cp /home/johannes/cont/densnet-torch-2026-10-11.sif "$local_image"
export DENSNET_TORCH_IMAGE=$local_image DENSNET_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$SLURM_JOB_ID" > "$report_dir/job-id.txt"
hostname > "$report_dir/node.txt"
date -Is > "$report_dir/start.txt"
git rev-parse HEAD > "$report_dir/development-head.txt"
git -C /home/johannes/equiv_dens_ml-paper-reference rev-parse HEAD > "$report_dir/reference-head.txt"
git -C ../research-infra rev-parse HEAD > "$report_dir/shared-head.txt"
git diff --binary -- src > "$report_dir/development.patch"
find src -name '*.py' -type f -print0 | sort -z | xargs -0 sha256sum > "$report_dir/source-files.sha256"
cp "$command_file" "$report_dir/commands.sh"
exec 9>/tmp/johannes_measurement.lock
flock -x 9
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?GPU visibility absent}" \
    --query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$report_dir/gpu-before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$report_dir/processes-before.csv"
python3 - "$report_dir" <<'PY'
import csv
import sys
from pathlib import Path
folder = Path(sys.argv[1])
gpus = list(csv.DictReader((folder / 'gpu-before.csv').open()))
uuids = {row[' uuid'].strip() for row in gpus}
apps = list(csv.DictReader((folder / 'processes-before.csv').open()))
if any(row['gpu_uuid'].strip() in uuids for row in apps):
    raise SystemExit('Allocated GPU has an existing compute process; timing refused.')
if any(int(row[' utilization.gpu [%]'].strip().split()[0]) > 0 for row in gpus):
    raise SystemExit('Allocated GPU is busy; timing refused.')
PY
printf '%s DenSNet performance start %s\n' "$(date -Is)" "$project_root" \
    >> "/tmp/agents-ressource-usage-${SLURM_JOB_ID}.txt"
export DENSNET_REPORT_DIR=$report_dir
set +e
bash "$command_file" > "$report_dir/run.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$report_dir/exit-code.txt"
date -Is > "$report_dir/end.txt"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,utilization.gpu,memory.used \
    --format=csv > "$report_dir/gpu-after.csv"
printf '%s DenSNet performance complete exit=%s\n' "$(date -Is)" "$result" \
    >> "/tmp/agents-ressource-usage-${SLURM_JOB_ID}.txt"
exit "$result"
