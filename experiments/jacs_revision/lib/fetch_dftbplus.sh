#!/usr/bin/env bash
# Install DFTB+ 24.1 Linux binary and 3ob-3-1 Slater–Koster files under tools/dftbplus/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PREFIX="${ROOT}/tools/dftbplus"
BIN="${PREFIX}/bin/dftb+"
SK="${PREFIX}/skfiles/3ob-3-1"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

DFTB_URL="https://github.com/dftbplus/dftbplus/releases/download/24.1/dftbplus-24.1.x86_64-linux.tar.xz"
SK_REPO="https://github.com/dftbparams/3ob/archive/refs/heads/main.tar.gz"

if [[ -x "${BIN}" ]]; then
  echo "DFTB+ binary already present: ${BIN}"
else
  mkdir -p "${PREFIX}"
  echo "Downloading DFTB+ 24.1 binary"
  curl -fsSL "${DFTB_URL}" -o "${TMP}/dftbplus.tar.xz"
  tar -xJf "${TMP}/dftbplus.tar.xz" -C "${TMP}"
  EXTRACTED="$(find "${TMP}" -maxdepth 1 -type d -name 'dftbplus-*' | head -1)"
  if [[ -z "${EXTRACTED}" ]]; then
    echo "ERROR: could not find extracted dftbplus directory" >&2
    exit 1
  fi
  mkdir -p "${PREFIX}/bin"
  cp "${EXTRACTED}/bin/dftb+" "${BIN}"
  chmod +x "${BIN}"
  # Ship any bundled libs next to the binary.
  if [[ -d "${EXTRACTED}/lib" ]]; then
    cp -a "${EXTRACTED}/lib" "${PREFIX}/"
  fi
  echo "Installed ${BIN}"
fi

if [[ -f "${SK}/C-H.skf" || -f "${SK}/H-C.skf" ]]; then
  echo "3ob SK files already present under ${SK}"
else
  echo "Downloading 3ob parameter set"
  curl -fsSL "${SK_REPO}" -o "${TMP}/3ob.tar.gz"
  tar -xzf "${TMP}/3ob.tar.gz" -C "${TMP}"
  REPO_DIR="$(find "${TMP}" -maxdepth 1 -type d -name '3ob-*' | head -1)"
  mkdir -p "${PREFIX}/skfiles"
  if [[ -d "${REPO_DIR}/3ob-3-1" ]]; then
    cp -a "${REPO_DIR}/3ob-3-1" "${PREFIX}/skfiles/"
  elif [[ -d "${REPO_DIR}/skfiles" ]]; then
    cp -a "${REPO_DIR}/skfiles/." "${PREFIX}/skfiles/"
    mkdir -p "${SK}"
    if [[ -d "${PREFIX}/skfiles/3ob-3-1" ]]; then
      true
    elif compgen -G "${PREFIX}/skfiles/*.skf" >/dev/null; then
      mkdir -p "${SK}"
      cp "${PREFIX}/skfiles/"*.skf "${SK}/" 2>/dev/null || true
    fi
  else
    echo "ERROR: unexpected 3ob archive layout under ${REPO_DIR}" >&2
    ls -la "${REPO_DIR}" >&2
    exit 1
  fi
  # Flatten if sk files live one level deeper.
  if [[ ! -f "${SK}/C-H.skf" && ! -f "${SK}/H-C.skf" ]]; then
    FOUND="$(find "${PREFIX}/skfiles" -name 'C-H.skf' -o -name 'H-C.skf' | head -1)"
    if [[ -n "${FOUND}" ]]; then
      SK_PARENT="$(dirname "${FOUND}")"
      mkdir -p "${SK}"
      cp "${SK_PARENT}/"*.skf "${SK}/"
    fi
  fi
  echo "Installed SK files under ${SK}"
fi

"${BIN}" --version || true
ls "${SK}/"*.skf 2>/dev/null | head -5 || ls "${SK}/" | head -10
