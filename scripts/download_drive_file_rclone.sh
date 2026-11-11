#!/usr/bin/env bash
# Download a private Drive file by ID using rclone (requires one-time OAuth).
#
# First-time setup (run once in an interactive terminal):
#   module load rclone/1.72.1
#   rclone config
#     n -> name: gdrive -> drive -> leave client id/secret blank
#     scope: 1 (full access) or drive.readonly
#     auto config: n   # headless: open URL in browser, paste token
#
# Then download:
#   bash scripts/download_drive_file_rclone.sh 16vtZsd7ra8DFC7ICIBZAWJioXUZcLOL7 paper/_staging/download/new_energy_models.zip
set -euo pipefail

FILE_ID="${1:?usage: $0 FILE_ID OUTPUT_PATH}"
OUTPUT="${2:?usage: $0 FILE_ID OUTPUT_PATH}"

module load rclone/1.72.1

if ! rclone listremotes | grep -q '^gdrive:'; then
  echo "ERROR: No rclone remote named 'gdrive'. Run: module load rclone/1.72.1 && rclone config"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

echo "Downloading file ID ${FILE_ID} ..."
rclone backend copyid gdrive: "${FILE_ID}" "${OUTPUT}"
echo "Saved: ${OUTPUT} ($(du -h "${OUTPUT}" | cut -f1))"
