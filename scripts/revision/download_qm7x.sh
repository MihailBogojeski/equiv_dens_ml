#!/usr/bin/env bash
# Download official QM7-X from Zenodo 4288677 (HDF5 property archive, no densities).
# Default: all eight xz shards plus README / DupMols / createDB.py.
# Use QM7X_FILES="8000.xz README.txt" for a small catalog smoke.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${QM7X_DEST:-${ROOT}/datasets/revision/qm7x/raw}"
EXTRACT="${QM7X_EXTRACT:-1}"
mkdir -p "$DEST"

FILES="${QM7X_FILES:-1000.xz 2000.xz 3000.xz 4000.xz 5000.xz 6000.xz 7000.xz 8000.xz README.txt DupMols.dat createDB.py}"

fetch() {
  local name="$1"
  local url="https://zenodo.org/records/4288677/files/${name}?download=1"
  local out="${DEST}/${name}"
  echo "fetching ${url} -> ${out}"
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$out" "$url"
  else
    curl -L -C - -o "$out" "$url"
  fi
}

for name in $FILES; do
  fetch "$name"
  if [[ "$EXTRACT" == "1" && "$name" == *.xz ]]; then
    hdf5="${DEST}/${name%.xz}.hdf5"
    if [[ ! -s "$hdf5" ]]; then
      echo "extracting ${DEST}/${name} -> ${hdf5}"
      xz -dc "${DEST}/${name}" > "$hdf5"
    fi
  fi
done

echo "QM7-X files in ${DEST}:"
ls -lh "$DEST"
