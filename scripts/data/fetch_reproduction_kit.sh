#!/usr/bin/env bash
# Download the figure-reproduction kit from Hugging Face and link it into this checkout.
#
# The kit (private dataset repo muhammadhasyim/equiv_dens_ml_dataset, built by
# scripts/data/build_reproduction_kit.py) holds the label arrays, basis/prior tables,
# minimal checkpoints and score outputs behind the JACS-revision figures, laid out as
# datasets/, results/revision/ and paper/models/. Access must have been granted to your
# Hugging Face account; log in first with `hf auth login`.
#
# Usage:
#   scripts/data/fetch_reproduction_kit.sh [--repo muhammadhasyim/equiv_dens_ml_dataset] [--root DIR]
#                                          [--include PATTERN ...] [--verify] [--dry-run]
#
# Env overrides:
#   EQUIV_DENS_KIT_ROOT   Local download target (default: /scratch/$USER/data/equiv_dens_ml_dataset,
#                         falling back to $HOME/data/equiv_dens_ml_dataset if /scratch is absent)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ID="muhammadhasyim/equiv_dens_ml_dataset"
ROOT_OVERRIDE=""
INCLUDE_ARGS=()
VERIFY=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) REPO_ID="$2"; shift 2 ;;
        --root) ROOT_OVERRIDE="$2"; shift 2 ;;
        --include) INCLUDE_ARGS+=(--include "$2"); shift 2 ;;
        --verify) VERIFY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "${ROOT_OVERRIDE}" ]]; then
    KIT_ROOT="${ROOT_OVERRIDE}"
elif [[ -n "${EQUIV_DENS_KIT_ROOT:-}" ]]; then
    KIT_ROOT="${EQUIV_DENS_KIT_ROOT}"
elif [[ -d "/scratch/${USER}" ]]; then
    KIT_ROOT="/scratch/${USER}/data/equiv_dens_ml_dataset"
else
    KIT_ROOT="${HOME}/data/equiv_dens_ml_dataset"
fi

PYTHON="${PYTHON:-python3}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
fi
HF="${HF:-hf}"
if ! command -v "${HF}" >/dev/null 2>&1 && [[ -x "${REPO_ROOT}/.venv/bin/hf" ]]; then
    HF="${REPO_ROOT}/.venv/bin/hf"
fi

echo "Downloading ${REPO_ID} -> ${KIT_ROOT}"
if [[ ${DRY_RUN} -eq 0 ]]; then
    mkdir -p "${KIT_ROOT}"
    "${HF}" download "${REPO_ID}" --repo-type dataset --local-dir "${KIT_ROOT}" "${INCLUDE_ARGS[@]}"
fi

if [[ ${VERIFY} -eq 1 && ${DRY_RUN} -eq 0 ]]; then
    "${PYTHON}" "${SCRIPT_DIR}/build_reproduction_kit.py" --verify "${KIT_ROOT}" --manifest "${KIT_ROOT}/manifest.json"
fi

echo "Linking ${KIT_ROOT} into ${REPO_ROOT}"
"${PYTHON}" "${SCRIPT_DIR}/link_reproduction_kit.py" --kit-root "${KIT_ROOT}" --repo-root "${REPO_ROOT}" \
    $([[ ${DRY_RUN} -eq 1 ]] && echo --dry-run)

cat <<EOF

Done. The figure scripts under experiments/jacs_revision/analysis now find their inputs
through symlinks into ${KIT_ROOT}; see ${KIT_ROOT}/README.md for the command behind each figure.
EOF
