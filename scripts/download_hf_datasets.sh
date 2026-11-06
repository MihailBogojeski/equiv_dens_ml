#!/usr/bin/env bash
# Download the equiv_dens_ml datasets from Hugging Face Hub and wire them into
# datasets/ via symlinks. See DATA.md for the directory map and what is
# intentionally NOT hosted here (OMol25 raw CSH dump, OMol25 train_4M index).
#
# Usage:
#   scripts/download_hf_datasets.sh [--subset water|qm7x|paper|all] [--root DIR]
#
# Env overrides:
#   EQUIV_DENS_DATA_ROOT   Local download target (default: /scratch/$USER/data/equiv_dens_ml,
#                          falling back to $HOME/data/equiv_dens_ml if /scratch is absent)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ID="muhammadhasyim/equiv_dens_ml"

SUBSET="all"
ROOT_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --subset) SUBSET="$2"; shift 2 ;;
        --root) ROOT_OVERRIDE="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "${ROOT_OVERRIDE}" ]]; then
    DATA_ROOT="${ROOT_OVERRIDE}"
elif [[ -n "${EQUIV_DENS_DATA_ROOT:-}" ]]; then
    DATA_ROOT="${EQUIV_DENS_DATA_ROOT}"
elif [[ -d "/scratch/${USER}" ]]; then
    DATA_ROOT="/scratch/${USER}/data/equiv_dens_ml"
else
    DATA_ROOT="${HOME}/data/equiv_dens_ml"
fi

mkdir -p "${DATA_ROOT}"

INCLUDE_ARGS=()
case "${SUBSET}" in
    all) ;;
    water) INCLUDE_ARGS=(--include "revision/water_*") ;;
    qm7x) INCLUDE_ARGS=(--include "revision/qm7x/*") ;;
    paper) INCLUDE_ARGS=(--include "*.npy") ;;
    *) echo "Unknown --subset '${SUBSET}' (expected water|qm7x|paper|all)" >&2; exit 1 ;;
esac

echo "Downloading ${REPO_ID} (subset: ${SUBSET}) -> ${DATA_ROOT}"
hf download "${REPO_ID}" --repo-type dataset --local-dir "${DATA_ROOT}" "${INCLUDE_ARGS[@]}"

echo "Linking ${DATA_ROOT} into ${REPO_ROOT}/datasets"
python3 "${SCRIPT_DIR}/link_hf_datasets.py" --data-root "${DATA_ROOT}" --repo-root "${REPO_ROOT}"

cat <<EOF

Done. datasets/ now resolves to ${DATA_ROOT} via symlinks (tracked xyz/json
geometry files in git are left untouched).

Not hosted on this Hub repo (re-download from upstream if needed):
  - OMol25 CSH raw HDF5 (Fock/density matrices, ~318 GB):
    https://huggingface.co/facebook/OMol25
  - OMol25 train_4M index mirror (~20 GB):
    https://huggingface.co/facebook/OMol25
EOF
