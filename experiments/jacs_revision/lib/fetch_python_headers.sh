#!/usr/bin/env bash
# Fetch CPython 3.12.9 Include/ for Triton gcc compiles (no system devel RPM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DEST="${ROOT}/tools/python312-headers"
VERSION="3.12.9"
URL="https://www.python.org/ftp/python/${VERSION}/Python-${VERSION}.tgz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ -f "${DEST}/Python.h" ]]; then
  echo "Python headers already present at ${DEST}/Python.h"
  exit 0
fi

mkdir -p "${DEST}"
echo "Downloading ${URL}"
curl -fsSL "${URL}" -o "${TMP}/Python-${VERSION}.tgz"
tar -xzf "${TMP}/Python-${VERSION}.tgz" -C "${TMP}"
cp -a "${TMP}/Python-${VERSION}/Include/." "${DEST}/"

# pyconfig.h: prefer the active venv's generated header when available.
if [[ -f "${ROOT}/.venv/bin/python" ]]; then
  PYCONFIG="$("${ROOT}/.venv/bin/python" - <<'PY'
import sysconfig
from pathlib import Path
p = Path(sysconfig.get_config_h_filename())
print(p if p.is_file() else "")
PY
)"
  if [[ -n "${PYCONFIG}" && -f "${PYCONFIG}" ]]; then
    cp "${PYCONFIG}" "${DEST}/pyconfig.h"
    echo "Copied pyconfig.h from venv: ${PYCONFIG}"
  fi
fi

if [[ ! -f "${DEST}/pyconfig.h" ]]; then
  echo "WARNING: no pyconfig.h in venv; copying pyconfig.h.in from CPython source"
  cp "${TMP}/Python-${VERSION}/pyconfig.h.in" "${DEST}/pyconfig.h"
fi

echo "Installed headers under ${DEST}"
ls -la "${DEST}/Python.h"
