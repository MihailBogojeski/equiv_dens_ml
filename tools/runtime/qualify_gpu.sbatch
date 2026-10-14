#!/usr/bin/env bash
# Short capability checks, not performance measurements or production runs.
#SBATCH --job-name=densnet-qualify
#SBATCH --partition=gpu-9m
#SBATCH --time=00:09:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
if [[ $# != 4 ]]; then
    echo 'Usage: sbatch qualify_gpu.sbatch PROJECT_ROOT torch|jax IMAGE REPORT_DIR' >&2
    exit 2
fi
project_root=$1
stack=$2
source_image=$3
report_dir=$4
cd "$project_root"
mkdir -p "$report_dir" "/temp/$(id -un)"
local_image="/temp/$(id -un)/densnet-qualify-${SLURM_JOB_ID}.sif"
trap 'rm -f "$local_image"' EXIT
cp "$source_image" "$local_image"
export DENSNET_TORCH_IMAGE=$local_image DENSNET_JAX_IMAGE=$local_image
runner="$project_root/tools/runtime/run.sh"
printf '%s qualification %s start %s\n' "$(date -Is)" "$stack" "$project_root" \
    >> "/tmp/agents-ressource-usage-${SLURM_JOB_ID}.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
    > "$report_dir/$stack-host-gpus.csv"
"$runner" "$stack" python tools/runtime/audit_environment.py > "$report_dir/$stack-environment.json"
"$runner" "$stack" cat /opt/densnet-runtime/installed.txt > "$report_dir/$stack-installed.txt"
"$runner" "$stack" python -m pip check > "$report_dir/$stack-pip-check.txt"
failed=0
if [[ $stack == torch ]]; then
    CUDA_VISIBLE_DEVICES='' "$runner" torch python -c \
        'import os, torch; assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""; assert not torch.cuda.is_available(); print("Explicit empty GPU visibility preserved: passed")' \
        > "$report_dir/torch-empty-gpu-visibility.txt" 2>&1 || failed=1
    "$runner" torch python -m pytest tests/ -q -ra \
        --junitxml="$report_dir/torch-pytest.xml" > "$report_dir/torch-pytest.txt" 2>&1 || failed=1
    "$runner" torch python tools/runtime/smoke_torch.py --device cpu \
        > "$report_dir/torch-cpu.json" 2> "$report_dir/torch-cpu.stderr.txt" || failed=1
    "$runner" torch python tools/runtime/smoke_torch.py --device cuda \
        > "$report_dir/torch-cuda.json" 2> "$report_dir/torch-cuda.stderr.txt" || failed=1
    "$runner" torch python tools/runtime/smoke_gxtb.py \
        > "$report_dir/gxtb.txt" 2>&1 || failed=1
    "$runner" torch nvcc --version > "$report_dir/nvcc.txt"
    "$runner" torch nvcc -O2 tools/runtime/cuda_smoke.cu \
        -o "/temp/$(id -un)/densnet-cuda-${SLURM_JOB_ID}" \
        > "$report_dir/nvcc-build.txt" 2>&1 || failed=1
    "$runner" torch "/temp/$(id -un)/densnet-cuda-${SLURM_JOB_ID}" \
        > "$report_dir/nvcc-run.txt" 2>&1 || failed=1
    rm -f "/temp/$(id -un)/densnet-cuda-${SLURM_JOB_ID}"
elif [[ $stack == jax ]]; then
    "$runner" jax python tools/runtime/smoke_jax.py --device cpu \
        --output "$report_dir/jax-cpu.json" > "$report_dir/jax-cpu.stdout.txt" 2> "$report_dir/jax-cpu.stderr.txt" || failed=1
    "$runner" jax python tools/runtime/smoke_jax.py --device cuda \
        --output "$report_dir/jax-cuda.json" > "$report_dir/jax-cuda.stdout.txt" 2> "$report_dir/jax-cuda.stderr.txt" || failed=1
else
    echo 'Runtime must be torch or jax' >&2
    exit 2
fi
printf 'Qualification exit: %s\n' "$failed" > "$report_dir/$stack-exit.txt"
printf '%s qualification %s complete exit=%s\n' "$(date -Is)" "$stack" "$failed" \
    >> "/tmp/agents-ressource-usage-${SLURM_JOB_ID}.txt"
exit "$failed"
