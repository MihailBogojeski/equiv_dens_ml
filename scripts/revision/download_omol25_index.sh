#!/usr/bin/env bash
# Download the public OMol25 train_4M ASE-DB index (19 GB). No HF token required.
# Densities still need Globus/MDF approval — do not pull the 500 TB dump.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${ROOT}/datasets/revision/omol25"
mkdir -p "$DEST"
URL="https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train_4M.tar.gz"
OUT="${DEST}/train_4M.tar.gz"
echo "fetching ${URL} -> ${OUT}"
if command -v wget >/dev/null 2>&1; then
  wget -c -O "$OUT" "$URL"
else
  curl -L -C - -o "$OUT" "$URL"
fi
echo "wrote ${OUT} ($(du -h "$OUT" | awk '{print $1}'))"
echo "extract with: tar -C ${DEST} -xzf ${OUT}"
