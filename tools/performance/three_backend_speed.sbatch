#!/usr/bin/env bash
#SBATCH --job-name=densnet-3backend
#SBATCH --partition=gpu-5h
#SBATCH --time=04:30:00
#SBATCH --gpus=40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo 'Usage: sbatch three_backend_speed.sbatch REPORT_DIR [--rounds N --cases CASE ... --modes MODE ...]' >&2
    exit 2
fi
project_root=${EQUIV_DENS_PROJECT_ROOT:-/home/johannes/equiv_dens_ml}
report_dir=$1
shift
if [[ -f "$report_dir/manifest.json" || -f "$report_dir/register.json" ]]; then
    echo 'Report directory already contains a run; use a fresh directory.' >&2
    exit 2
fi
cd "$project_root"
mkdir -p "$report_dir" "/temp/$(id -un)"
report_dir=$(realpath "$report_dir")
job_id=${SLURM_JOB_ID:-no-slurm-job-id}
user_tmp="/temp/$(id -un)"
torch_image_source=${DENSNET_TORCH_IMAGE:-/home/johannes/cont/densnet-torch-2026-10-11.sif}
jax_image_source=${DENSNET_JAX_IMAGE:-/home/johannes/cont/densnet-jax-pipeline-2026-10-11.sif}
torch_image="$user_tmp/densnet-3backend-torch-${job_id}.sif"
jax_image="$user_tmp/densnet-3backend-jax-${job_id}.sif"
result=1

finish() {
    local trapped=$?
    if [[ $result -eq 1 && $trapped -ne 0 ]]; then result=$trapped; fi
    date -Is > "$report_dir/end.txt"
    if [[ -f "$report_dir/source-files.sha256" ]]; then
        if ! sha256sum -c "$report_dir/source-files.sha256" > "$report_dir/source-verification-after.txt"; then
            result=1
            printf 'FAIL: source file hashes changed during allocation\n' > "$report_dir/source-verification-status.txt"
        else
            printf 'PASS\n' > "$report_dir/source-verification-status.txt"
        fi
    fi
    if [[ -f "$report_dir/data-files.sha256" ]]; then
        if ! sha256sum -c "$report_dir/data-files.sha256" > "$report_dir/data-verification-after.txt"; then
            result=1
            printf 'FAIL: input data hashes changed during allocation\n' > "$report_dir/data-verification-status.txt"
        else
            printf 'PASS\n' > "$report_dir/data-verification-status.txt"
        fi
    fi
    if [[ -f "$torch_image" && -f "$jax_image" ]]; then
        sha256sum "$torch_image" "$jax_image" > "$report_dir/images-after.sha256" || true
    fi
    if command -v nvidia-smi >/dev/null 2>&1 && [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,utilization.gpu,memory.used,memory.total \
            --format=csv > "$report_dir/gpu-after.csv" 2>&1 || true
    fi
    printf '%s %s three-backend benchmark exit=%s\n' "$(date -Is)" "$job_id" "$result" \
        >> "/tmp/agents-ressource-usage-${job_id}.txt" 2>/dev/null || true
    rm -f "$torch_image" "$jax_image"
    printf '%s\n' "$result" > "$report_dir/exit-code.txt"
    trap - EXIT
    exit "$result"
}
trap finish EXIT

printf '%s\n' "$job_id" > "$report_dir/job-id.txt"
hostname > "$report_dir/node.txt"
date -Is > "$report_dir/start.txt"
git rev-parse HEAD > "$report_dir/source-head.txt"
git -C ../research-infra rev-parse HEAD > "$report_dir/shared-head.txt"
printf '%s\n' "$project_root" > "$report_dir/source-root.txt"
printf '%s\n' "${SLURM_JOB_PARTITION:-unknown}" > "$report_dir/partition.txt"
printf '%s\n' "${SLURM_JOB_NODELIST:-unknown}" > "$report_dir/nodelist.txt"
printf '%s\n' "${SLURM_CPUS_PER_TASK:-unknown}" > "$report_dir/cpus.txt"
printf '%s\n' "${SLURM_MEM_PER_NODE:-unknown}" > "$report_dir/memory-mb.txt"
scontrol show job "$job_id" > "$report_dir/slurm-job.txt" 2>&1 || true

# Snapshot all tracked and newly prepared instrument/fixture source that can
# affect the result; hash the supplied geometry and auxiliary basis inputs too.
find src benchmarks tools/runtime tools/performance -type f \
    \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' \) -print0 2>/dev/null \
    | sort -z | xargs -0 -r sha256sum > "$report_dir/source-files.sha256"
data_inputs=(datasets/ethanol_train_10.xyz datasets/ethanol_train_10.npy \
    datasets/augccpvqzjkfit_orbital_basis_df.npy \
    datasets/augccpvqzjkfit_radial_coeffs_df.npy \
    datasets/augccpvqzjkfit_orbital_basis_libcint_df.npy \
    datasets/augccpvqzjkfit_radial_coeffs_libcint_df.npy \
    datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy)
present_inputs=()
for input_file in "${data_inputs[@]}"; do
    [[ ! -f "$input_file" ]] || present_inputs+=("$input_file")
done
if ((${#present_inputs[@]})); then sha256sum "${present_inputs[@]}" > "$report_dir/data-files.sha256"; fi

cp "$torch_image_source" "$torch_image"
cp "$jax_image_source" "$jax_image"
export DENSNET_TORCH_IMAGE=$torch_image DENSNET_JAX_IMAGE=$jax_image
export EQUIV_DENS_PROJECT_ROOT=$project_root DENSNET_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
sha256sum "$torch_image" "$jax_image" > "$report_dir/images-before.sha256"
printf '%s\n' "${CUDA_VISIBLE_DEVICES:-}" > "$report_dir/allocated-visible-devices.txt"

# Default measurements own the node-wide lock exclusively. Explicitly authorized
# parallel workers on distinct allocated GPUs share that lock and each own their
# GPU UUID lock exclusively; older node-wide exclusive workers still wait.
exec 9>/tmp/johannes_measurement.lock
lock_scope=${DENSNET_MEASUREMENT_LOCK_SCOPE:-node}
case "$lock_scope" in
    node) flock -x 9 ;;
    allocated-gpu)
        flock -s 9
        allocated_uuid=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?Missing allocated GPU}" \
            --query-gpu=uuid --format=csv,noheader)
        if [[ ! "$allocated_uuid" =~ ^GPU-[a-fA-F0-9-]+$ ]]; then
            echo 'Per-GPU measurement requires exactly one visible allocated UUID.' >&2
            exit 1
        fi
        exec 8>"/tmp/johannes_measurement-${allocated_uuid}.lock"
        flock -x 8
        ;;
    *) echo 'Unknown measurement lock scope.' >&2; exit 2 ;;
esac
printf '%s\n' "$lock_scope" > "$report_dir/measurement-lock-scope.txt"
printf '%s %s three-backend benchmark on %s\n' "$(date -Is)" "$job_id" "$project_root" \
    >> "/tmp/agents-ressource-usage-${job_id}.txt"
printf '%s %s three-backend benchmark\n' "$(date -Is)" "$job_id" \
    >> /tmp/johannes_measurement.lock.holders
nvidia-smi -i "${CUDA_VISIBLE_DEVICES:?Slurm did not expose an allocated GPU}" \
    --query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,memory.total \
    --format=csv > "$report_dir/gpu-before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv > "$report_dir/processes-before.csv"
python3 - "$report_dir" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

folder = Path(sys.argv[1])
def rows(path):
    return [{key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(path.open())]

gpus = rows(folder / "gpu-before.csv")
apps = rows(folder / "processes-before.csv")
uuids = {row["uuid"] for row in gpus}
allowed = {int(value) for value in os.environ.get("DENSNET_ALLOWED_PAUSED_GPU_PIDS", "").split(",") if value}
paused = []
for pid in allowed:
    proc = Path(f"/proc/{pid}")
    state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
    argv = (proc / "cmdline").read_bytes().split(b"\0")
    training_script = b"/home/johannes/atomistic-deq-ff/src/atomistic_deq_ff/scripts/train_deq_modular.py"
    if state not in ("T", "t") or proc.stat().st_uid != os.getuid() or training_script not in argv:
        raise SystemExit("Allowed resident process is not the verified stopped training process.")
    paused.append({"pid": pid, "state": state, "retains_device_memory": True})
if not gpus or any(row["gpu_uuid"] in uuids and int(row["pid"]) not in allowed for row in apps):
    raise SystemExit("Allocated GPU has an existing compute process; timing refused.")
for row in gpus:
    utilization = row.get("utilization.gpu [%]", "0 %").split()[0]
    if int(utilization) > 0:
        raise SystemExit("Allocated GPU is busy; timing refused.")
    if allowed and int(row["memory.total [MiB]"].split()[0]) - int(row["memory.used [MiB]"].split()[0]) < 10240:
        raise SystemExit("Paused-training GPU has less than 10 GiB free.")
(folder / "paused-resident-processes.json").write_text(json.dumps({
    "allowed_paused_processes": paused,
    "scope": "Resident training allocations remain pinned; reported backend allocator memory belongs to its own process.",
}, indent=2) + "\n")
PY

# Record each image's actual library/runtime identity and the allocated device.
tools/runtime/run.sh torch python -c 'import json,torch; print(json.dumps({"torch":torch.__version__,"cuda":torch.version.cuda,"device_count":torch.cuda.device_count()}))' \
    > "$report_dir/torch-runtime.json" 2> "$report_dir/torch-runtime.log"
tools/runtime/run.sh jax python -c 'import json,jax; print(json.dumps({"jax":jax.__version__,"devices":[str(d) for d in jax.devices()]}))' \
    > "$report_dir/jax-runtime.json" 2> "$report_dir/jax-runtime.log"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" -q > "$report_dir/hardware.txt"

set +e
python3 -u tools/performance/three_backend_speed.py --output "$report_dir" "$@" \
    > "$report_dir/run.log" 2>&1
result=$?
set -e
exit "$result"
