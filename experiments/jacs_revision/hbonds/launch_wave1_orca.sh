#!/bin/bash
# Submit wave-1 dual-theory ORCA labels for all new-system campaigns.
#
# Order: MeOH-water -> AcAc -> paracetamol -> NMA -> PGG -> crambin fragments.
# Run geometry generators first:
#   ./.venv/bin/python scripts/revision/hbonds/generate_all.py

set -euo pipefail
cd "$(dirname "$0")/../../.."

SYSTEMS=(meoh_water acac paracetamol nma_dimer pgg crambin_frags)
THEORIES=(wb97mv_def2tzvpd pbe_d4_avdz)
MODE=${1:---build-only}

for theory in "${THEORIES[@]}"; do
  for system in "${SYSTEMS[@]}"; do
    echo "=== ${system} @ ${theory} ==="
    SYSTEM="$system" THEORY="$theory" scripts/revision/hbonds/launch_hbond_campaign.sh "$MODE"
  done
done

echo "Wave 1 shard build complete. Submit with: $0 --submit"
