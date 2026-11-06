#!/usr/bin/env bash
# Import EquivDens Drive Archive — organize/unzip phase.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="${REPO_ROOT}/paper/_staging/download"
ARCHIVES="${REPO_ROOT}/paper/archives"
EXTRACT="${REPO_ROOT}/paper/_staging/extract"
MODELS="${REPO_ROOT}/paper/models"
TRAJ="${REPO_ROOT}/paper/trajectories"
NOTEBOOKS="${REPO_ROOT}/paper/notebooks"
MANUSCRIPTS="${REPO_ROOT}/docs/manuscripts"

mkdir -p "${ARCHIVES}" "${EXTRACT}" "${TRAJ}" "${NOTEBOOKS}" \
    "${MANUSCRIPTS}/drive_linked" \
    "${MANUSCRIPTS}/gmail/2024" "${MANUSCRIPTS}/gmail/2025" "${MANUSCRIPTS}/gmail/2026" \
    "${MANUSCRIPTS}/gmail_manual" "${MANUSCRIPTS}/source_configs"

log() { echo "[$(date -Iseconds)] $*"; }

# Collect all zips from staging (archive folder tree + standalone)
collect_zips() {
    find "${STAGING}" -name '*.zip' -type f 2>/dev/null | sort -u
}

archive_all_zips() {
    log "Archiving zips to ${ARCHIVES}/"
    while IFS= read -r z; do
        base="$(basename "${z}")"
        if [[ ! -f "${ARCHIVES}/${base}" ]]; then
            cp -n "${z}" "${ARCHIVES}/${base}"
            log "  archived: ${base}"
        fi
    done < <(collect_zips)
}

zipinfo_record() {
    local z="$1"
    local out="${REPO_ROOT}/paper/_staging/zipinfo/$(basename "${z}").txt"
    mkdir -p "$(dirname "${out}")"
    zipinfo -1 "${z}" > "${out}" 2>/dev/null || unzip -l "${z}" > "${out}" 2>/dev/null || true
}

extract_zip() {
    local z="$1"
    local dest="$2"
    mkdir -p "${dest}"
    log "  extract $(basename "${z}") -> ${dest}"
    unzip -o -q "${z}" -d "${dest}" 2>/dev/null || unzip -o "${z}" -d "${dest}"
}

# rsync checkpoints without overwriting args.txt or same-size .pth/.pt
rsync_models() {
    local src="$1"
    local dest="$2"
    mkdir -p "${dest}"
    rsync -av --ignore-existing \
        --exclude='args.txt' \
        "${src}/" "${dest}/" 2>/dev/null || true
    # For .pth/.pt: skip if same size exists
    find "${src}" \( -name '*.pth' -o -name '*.pt' \) -type f | while read -r f; do
        rel="${f#${src}/}"
        target="${dest}/${rel}"
        if [[ -f "${target}" ]]; then
            src_size=$(stat -c%s "${f}")
            tgt_size=$(stat -c%s "${target}")
            if [[ "${src_size}" -eq "${tgt_size}" ]]; then
                continue
            fi
        fi
        mkdir -p "$(dirname "${target}")"
        cp -n "${f}" "${target}" 2>/dev/null || cp "${f}" "${target}"
    done
}

map_zip_to_dest() {
    local base="$1"
    case "${base}" in
        thiophene_new_model.zip)
            echo "model:polythiophene"
            ;;
        new_ethanethiol_model.zip|ethanethiol_models_for_stability.zip)
            echo "model:ethanethiol"
            ;;
        new_resorcinol_model.zip)
            echo "model:resorcinol"
            ;;
        2024-03-22_96w7KyGG.zip)
            echo "model:ethanol/2024-03-22_96w7KyGG"
            ;;
        qm7x250_model.zip)
            echo "model:qm7x/2024-04-22_bjOUNzrR"
            ;;
        thiophene_md_trajectories.zip)
            echo "traj:polythiophene"
            ;;
        thiophene_traj_with_velocities.zip)
            echo "traj:polythiophene/with_velocity"
            ;;
        tomihail.zip|new_energy_models.zip|new_dens_model.zip|old_md_model.zip|delta_energy_models.zip|model_data.zip)
            echo "unmapped"
            ;;
        *)
            echo "inspect"
            ;;
    esac
}

process_zips() {
    archive_all_zips
    mkdir -p "${REPO_ROOT}/paper/_staging/zipinfo"
    while IFS= read -r z; do
        local base dest_key staging_dir
        base="$(basename "${z}")"
        zipinfo_record "${z}"
        dest_key="$(map_zip_to_dest "${base}")"
        staging_dir="${EXTRACT}/${base%.zip}"
        extract_zip "${z}" "${staging_dir}"

        case "${dest_key}" in
            model:*)
                local mol="${dest_key#model:}"
                if [[ "${mol}" == *"/"* ]]; then
                    rsync_models "${staging_dir}" "${MODELS}/${mol}"
                else
                    # Find run_id subdirs or rsync top-level
                    if find "${staging_dir}" -mindepth 1 -maxdepth 1 -type d | grep -q .; then
                        find "${staging_dir}" -mindepth 1 -maxdepth 1 -type d | while read -r d; do
                            rsync_models "${d}" "${MODELS}/${mol}/$(basename "${d}")"
                        done
                    else
                        rsync_models "${staging_dir}" "${MODELS}/${mol}/_from_${base%.zip}"
                    fi
                fi
                ;;
            traj:*)
                local tpath="${dest_key#traj:}"
                mkdir -p "${TRAJ}/${tpath}"
                rsync -av "${staging_dir}/" "${TRAJ}/${tpath}/"
                ;;
            unmapped)
                mkdir -p "${MODELS}/_unmapped/${base%.zip}"
                rsync -av "${staging_dir}/" "${MODELS}/_unmapped/${base%.zip}/"
                ;;
            inspect)
                log "  INSPECT NEEDED: ${base} -> ${MODELS}/_unmapped/${base%.zip}/"
                mkdir -p "${MODELS}/_unmapped/${base%.zip}"
                rsync -av "${staging_dir}/" "${MODELS}/_unmapped/${base%.zip}/"
                ;;
        esac
    done < <(collect_zips)
}

place_standalone_files() {
    # 8mer xyz
    local xyz="${STAGING}/simulation_local_all_8mer_md_001_0.xyz"
    if [[ -f "${xyz}" ]]; then
        mkdir -p "${TRAJ}/polythiophene/8mer"
        cp -n "${xyz}" "${TRAJ}/polythiophene/8mer/" 2>/dev/null || cp "${xyz}" "${TRAJ}/polythiophene/8mer/"
        cp -n "${xyz}" "${ARCHIVES}/" 2>/dev/null || true
    fi
    # Also search archive folder
    find "${STAGING}" -name 'simulation_local_all_8mer_md_001_0.xyz' -type f | while read -r f; do
        mkdir -p "${TRAJ}/polythiophene/8mer"
        cp -n "${f}" "${TRAJ}/polythiophene/8mer/" 2>/dev/null || true
    done

    # Resorcinol npy
    for i in 0 1 2 3; do
        local npy="simulation_resorcinol_cluster_all_001_compressed_${i}.npy"
        find "${STAGING}" -name "${npy}" -type f | while read -r f; do
            mkdir -p "${TRAJ}/resorcinol"
            cp -n "${f}" "${TRAJ}/resorcinol/" 2>/dev/null || cp "${f}" "${TRAJ}/resorcinol/"
            cp -n "${f}" "${ARCHIVES}/" 2>/dev/null || true
        done
    done

    # Ethanethiol cluster npy from archive
    find "${STAGING}" -name 'simulation_-ethanethiol_cluster_all_001_compressed_0.npy' -o \
         -name 'simulation_ethanethiol_cluster_all_001_compressed_0.npy' 2>/dev/null | while read -r f; do
        mkdir -p "${TRAJ}/ethanethiol"
        cp -n "${f}" "${TRAJ}/ethanethiol/" 2>/dev/null || true
    done

    # loss_delta.csv from archive
    find "${STAGING}" -name 'loss_delta.csv' -type f | while read -r f; do
        mkdir -p "${TRAJ}/ethanol"
        cp -n "${f}" "${TRAJ}/ethanol/" 2>/dev/null || true
    done

    # thiophene_append_clean.ipynb
    find "${STAGING}" -name 'thiophene_append_clean.ipynb' -type f | while read -r f; do
        cp -n "${f}" "${NOTEBOOKS}/" 2>/dev/null || cp "${f}" "${NOTEBOOKS}/"
    done

    # 4mer/6mer npy duplicates
    for npy in thiophene4mer_test_d4.npy thiophene6mer_test_d4.npy; do
        find "${STAGING}" -name "${npy}" -type f | while read -r f; do
            mkdir -p "${TRAJ}/polythiophene"
            cp -n "${f}" "${TRAJ}/polythiophene/" 2>/dev/null || true
        done
    done

    # Gmail-recovered files (if present in staging/gmail/)
    local gmail_staging="${REPO_ROOT}/paper/_staging/gmail"
    if [[ -d "${gmail_staging}" ]]; then
        # ethanol cluster npy + csvs
        for f in simulation_cluster_all_001_compressed_0.npy loss_base.csv loss_delta.csv; do
            if [[ -f "${gmail_staging}/${f}" ]]; then
                mkdir -p "${TRAJ}/ethanol"
                cp -n "${gmail_staging}/${f}" "${TRAJ}/ethanol/" 2>/dev/null || \
                    cp "${gmail_staging}/${f}" "${TRAJ}/ethanol/"
            fi
        done
        # dipole zip
        if [[ -f "${gmail_staging}/traj_dipole_moment_results.zip" ]]; then
            extract_zip "${gmail_staging}/traj_dipole_moment_results.zip" "${EXTRACT}/traj_dipole_moment_results"
            rsync -av "${EXTRACT}/traj_dipole_moment_results/" "${TRAJ}/polythiophene/dipole/"
            cp -n "${gmail_staging}/traj_dipole_moment_results.zip" "${ARCHIVES}/" 2>/dev/null || true
        fi
        # 1ps mid trajectories
        if [[ -f "${gmail_staging}/polythiophene_1ps_mid_trajectories.zip" ]]; then
            extract_zip "${gmail_staging}/polythiophene_1ps_mid_trajectories.zip" "${EXTRACT}/polythiophene_1ps_mid"
            rsync -av "${EXTRACT}/polythiophene_1ps_mid/" "${TRAJ}/polythiophene/1ps_mid/"
            cp -n "${gmail_staging}/polythiophene_1ps_mid_trajectories.zip" "${ARCHIVES}/" 2>/dev/null || true
        fi
        # thiophene config compare
        if [[ -f "${gmail_staging}/thiophene_poly_all_001_coreless_dpm.txt" ]]; then
            cp "${gmail_staging}/thiophene_poly_all_001_coreless_dpm.txt" \
                "${MANUSCRIPTS}/source_configs/thiophene_poly_all_001_coreless_dpm.gmail.txt"
        fi
        # manuscripts
        find "${gmail_staging}/manuscripts" -type f 2>/dev/null | while read -r f; do
            cp -n "${f}" "${MANUSCRIPTS}/gmail/$(dirname "${f#${gmail_staging}/manuscripts/}")/" 2>/dev/null || true
        done
    fi
}

place_manuscripts() {
    local archive="${STAGING}/EquivDens_Project_Archive"
    [[ -d "${archive}" ]] || return 0
    log "Copying manuscripts from archive..."
    # Drive-linked PDFs
    find "${archive}/01_Drive_Linked_Files" -name '*.pdf' -o -name '*.pptx' 2>/dev/null | while read -r f; do
        cp -n "${f}" "${MANUSCRIPTS}/drive_linked/" 2>/dev/null || true
    done
    # Gmail attachments by year
    for year in 2024 2025 2026; do
        src="${archive}/02_Gmail_Attachments/${year}"
        [[ -d "${src}" ]] || continue
        mkdir -p "${MANUSCRIPTS}/gmail/${year}"
        find "${src}" -type f | while read -r f; do
            cp -n "${f}" "${MANUSCRIPTS}/gmail/${year}/" 2>/dev/null || true
        done
    done
    # loss_delta.csv at 2025 root
    [[ -f "${archive}/02_Gmail_Attachments/2025/loss_delta.csv" ]] && \
        mkdir -p "${TRAJ}/ethanol" && \
        cp -n "${archive}/02_Gmail_Attachments/2025/loss_delta.csv" "${TRAJ}/ethanol/" 2>/dev/null || true
    # Source configs from archive
    for cfg in ethanethiol_all_004_31_coreless.txt thiophene_poly_all_001_coreless_dpm.txt; do
        find "${archive}" -name "${cfg}" -type f | while read -r f; do
            cp -n "${f}" "${MANUSCRIPTS}/source_configs/${cfg%.txt}.drive_archive.txt" 2>/dev/null || true
        done
    done
}

log "=== Organize phase start ==="
process_zips
place_standalone_files
place_manuscripts
log "=== Organize phase complete ==="
du -sh "${ARCHIVES}" "${MODELS}" "${TRAJ}" "${MANUSCRIPTS}" 2>/dev/null || true
