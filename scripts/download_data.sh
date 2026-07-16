#!/usr/bin/env bash
# Download large datasets moved from Git LFS to Zenodo/Figshare.
# See DATA.md for DOI references and manual download options.
# Usage: ./scripts/download_data.sh [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Placeholder Zenodo record IDs - replace after uploading
# Update these in DATA.md when DOIs are available
ZENODO_POLYTHIOPHENE_FULL_MD=""      # e.g. 1234567
ZENODO_POLYTHIOPHENE_WITH_VELOCITY="" # e.g. 1234568
ZENODO_THIOPHENE_MD_ZIP=""           # e.g. 1234569
ZENODO_1PS_MID_ZIP=""                # e.g. 1234570

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

download_url() {
    local url="$1"
    local dest="$2"
    if [[ -z "${url}" ]] || [[ "${url}" == "https://zenodo.org/record//files/"* ]]; then
        echo "Skipping (DOI not set): ${dest}"
        return 0
    fi
    echo "Downloading: ${url} -> ${dest}"
    if ! ${DRY_RUN}; then
        mkdir -p "$(dirname "${dest}")"
        if command -v wget &>/dev/null; then
            wget -q --show-progress -O "${dest}" "${url}" || true
        elif command -v curl &>/dev/null; then
            curl -L -o "${dest}" "${url}"
        else
            echo "Error: wget or curl required. Install one, or download manually (see DATA.md)."
            exit 1
        fi
    fi
}

download_zenodo_files() {
    local record_id="$1"
    local out_dir="$2"
    shift 2
    local files=("$@")
    [[ -z "${record_id}" ]] && { echo "Skipping (record ID not set): ${out_dir}"; return 0; }
    local base="https://zenodo.org/record/${record_id}/files"
    for f in "${files[@]}"; do
        download_url "${base}/${f}" "${out_dir}/${f}"
    done
}

# Polythiophene full MD (12 npy files)
FILES_FULL_MD=(
    "8mer/simulation_local_all_8mer_md_001_0.npy"
    "8mer/simulation_local_all_8mer_md_001_1.npy"
    "8mer/simulation_local_all_8mer_md_001_2.npy"
    "8mer/simulation_local_all_8mer_md_001_3.npy"
    "10mer/simulation_local_all_10mer_md_001_0.npy"
    "10mer/simulation_local_all_10mer_md_001_1.npy"
    "10mer/simulation_local_all_10mer_md_001_2.npy"
    "10mer/simulation_local_all_10mer_md_001_3.npy"
    "12mer/simulation_local_all_12mer_md_001_0.npy"
    "12mer/simulation_local_all_12mer_md_001_1.npy"
    "12mer/simulation_local_all_12mer_md_001_2.npy"
    "12mer/simulation_local_all_12mer_md_001_3.npy"
)
download_zenodo_files "${ZENODO_POLYTHIOPHENE_FULL_MD}" "paper/trajectories/polythiophene_full_md" "${FILES_FULL_MD[@]}"

# Polythiophene with velocity (12 npy files)
FILES_WITH_VEL=(
    "8mer/simulation_local_all_8mer_md_001_vel_0.npy"
    "8mer/simulation_local_all_8mer_md_001_vel_1.npy"
    "8mer/simulation_local_all_8mer_md_001_vel_2.npy"
    "8mer/simulation_local_all_8mer_md_001_vel_3.npy"
    "10mer/simulation_local_all_10mer_md_001_vel_0.npy"
    "10mer/simulation_local_all_10mer_md_001_vel_1.npy"
    "10mer/simulation_local_all_10mer_md_001_vel_2.npy"
    "10mer/simulation_local_all_10mer_md_001_vel_3.npy"
    "12mer/simulation_local_all_12mer_md_001_vel_0.npy"
    "12mer/simulation_local_all_12mer_md_001_vel_1.npy"
    "12mer/simulation_local_all_12mer_md_001_vel_2.npy"
    "12mer/simulation_local_all_12mer_md_001_vel_3.npy"
)
download_zenodo_files "${ZENODO_POLYTHIOPHENE_WITH_VELOCITY}" "paper/trajectories/polythiophene_full_with_velocity" "${FILES_WITH_VEL[@]}"

# thiophene_md_trajectories.zip
if [[ -n "${ZENODO_THIOPHENE_MD_ZIP}" ]]; then
    download_url "https://zenodo.org/record/${ZENODO_THIOPHENE_MD_ZIP}/files/thiophene_md_trajectories.zip" "to_organize/thiophene_md_trajectories.zip"
fi

# polythiophene_1ps_mid_trajectories.zip
if [[ -n "${ZENODO_1PS_MID_ZIP}" ]]; then
    mkdir -p paper/trajectories/archives
    download_url "https://zenodo.org/record/${ZENODO_1PS_MID_ZIP}/files/polythiophene_1ps_mid_trajectories.zip" "paper/trajectories/archives/polythiophene_1ps_mid_trajectories.zip"
fi

echo "Done. If DOIs are not yet set, upload to Zenodo and update ZENODO_* in this script and DATA.md."
