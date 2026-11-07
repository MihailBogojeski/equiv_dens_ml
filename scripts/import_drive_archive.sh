#!/usr/bin/env bash
# Import EquivDens Drive Archive — download phase.
# Uses gdown (not Drive MCP) for multi-GB files.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="${REPO_ROOT}/paper/_staging/download"
ARCHIVES="${REPO_ROOT}/paper/archives"
GDOWN="${REPO_ROOT}/.venv/bin/gdown"

mkdir -p "${STAGING}" "${ARCHIVES}"

log() { echo "[$(date -Iseconds)] $*"; }

download_if_missing() {
    local id="$1"
    local dest="$2"
    local name
    name="$(basename "${dest}")"
    if [[ -f "${dest}" ]]; then
        log "SKIP (exists): ${name}"
        return 0
    fi
    log "DOWNLOAD: ${name} (${id})"
    "${GDOWN}" "https://drive.google.com/uc?id=${id}" -O "${dest}"
}

# 1. Archive folder
FOLDER_ID="1_Y2Zvk-ZdZZcmGFHjpM3-t9akL-7Fj8L"
FOLDER_DEST="${STAGING}/EquivDens_Project_Archive"
if [[ ! -d "${FOLDER_DEST}" ]] || [[ -z "$(ls -A "${FOLDER_DEST}" 2>/dev/null)" ]]; then
    log "DOWNLOAD FOLDER: EquivDens Project Archive"
    "${GDOWN}" --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" \
        -O "${FOLDER_DEST}" --continue
else
    log "SKIP FOLDER (exists): ${FOLDER_DEST}"
fi

# 2. Leftover standalone Drive files
declare -A STANDALONE=(
    ["16vtZsd7ra8DFC7ICIBZAWJioXUZcLOL7"]="new_energy_models.zip"
    ["1bLz2lafxPV88HwcocZ_Ec-pnWVF41X21"]="thiophene_traj_with_velocities.zip"
    ["1HCqz68guLckGOsDhcphiyruqcdIlm-uE"]="simulation_local_all_8mer_md_001_0.xyz"
    ["1h54CKkXMp07tZD4hjdh-9fXd64T2DVNI"]="simulation_resorcinol_cluster_all_001_compressed_2.npy"
    ["1QHVRXpZyhdOs6SrKHoPCrqbJ8WGtQGEl"]="simulation_resorcinol_cluster_all_001_compressed_3.npy"
)

for id in "${!STANDALONE[@]}"; do
    download_if_missing "${id}" "${STAGING}/${STANDALONE[$id]}"
done

# 3. Copy all zips from archive folder + standalone to paper/archives/
log "Copying zips to paper/archives/"
find "${STAGING}" -name '*.zip' -type f | while read -r z; do
    base="$(basename "${z}")"
    if [[ ! -f "${ARCHIVES}/${base}" ]]; then
        cp -n "${z}" "${ARCHIVES}/${base}"
        log "ARCHIVED: ${base}"
    fi
done

# Copy standalone non-zip files to staging root for organize step
for f in simulation_local_all_8mer_md_001_0.xyz \
         simulation_resorcinol_cluster_all_001_compressed_2.npy \
         simulation_resorcinol_cluster_all_001_compressed_3.npy; do
    if [[ -f "${STAGING}/${f}" ]] && [[ ! -f "${ARCHIVES}/${f}" ]]; then
        cp -n "${STAGING}/${f}" "${ARCHIVES}/${f}" 2>/dev/null || true
    fi
done

log "Download phase complete."
du -sh "${STAGING}" "${ARCHIVES}" 2>/dev/null || true
