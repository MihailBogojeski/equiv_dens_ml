#!/bin/bash
# Re-submit timed-out ood-geom shards. Sibling trees only; no 3k tr-* jobs.
#
# The 12-mer is energy-only and often dies at the wall. The large-tier worker
# now keeps the GBW on the shard and MOREADs it, so the next task continues
# instead of starting SCF over. RETRY_ERRORS reopens shards that finished
# with only AO-map rejects (those energies were valid).
#
# Usage:
#   sbatch scripts/revision/submit_ood_geometry_watch.sbatch
#   scripts/revision/watch_ood_geometry_dft.sh          # one-shot from a login node

set -euo pipefail
cd "$(dirname "$0")/../../.."

SHARD_ROOT=${SHARD_ROOT:-datasets/revision/shards_ood_geom}
OUT_ROOT=${OUT_ROOT:-results/revision/ood_geom_orca}
INTERVAL=${INTERVAL:-1800}
ONESHOT=${ONESHOT:-1}
JOB_PREFIX=${JOB_PREFIX:-og}
CONCURRENT_SMALL=${CONCURRENT_SMALL:-20}
CONCURRENT_LARGE=${CONCURRENT_LARGE:-60}
# cs only. cpu_short also contains gl (L40S) and gr (RTX 6000).
PARTITION="${PARTITION:-cs}"

watch_split() {
  local split=$1 sbatch_file=$2 concurrent=$3 no_df=$4 retry=$5 partition=$6
  local theory
  for theory in pbe_d4_avdz wb97mv_def2tzvpd; do
    local shard_dir="${SHARD_ROOT}/${split}/${theory}"
    local outdir="${OUT_ROOT}/${split}/${theory}"
    [[ -d "$shard_dir" ]] || continue
    echo "=== ${split} ${theory} ==="
    ONESHOT="$ONESHOT" INTERVAL="$INTERVAL" JOB_PREFIX="$JOB_PREFIX" \
      SBATCH_FILE="$sbatch_file" CONCURRENT="$concurrent" \
      PARTITION="$partition" \
      NO_DF="$no_df" RETRY_ERRORS="$retry" KEEP_ORCA=1 \
      scripts/revision/water/watch_and_resubmit.sh "$shard_dir" "$outdir" "$theory"
  done
}

watch_split ood_modes_water_n6 scripts/revision/submit_water_orca.sbatch \
  "$CONCURRENT_SMALL" 0 1 "$PARTITION"
watch_split ood_modes_resorcinol scripts/revision/submit_water_orca.sbatch \
  "$CONCURRENT_SMALL" 0 1 "$PARTITION"
watch_split ood_modes_malonaldehyde scripts/revision/submit_water_orca.sbatch \
  "$CONCURRENT_SMALL" 0 1 "$PARTITION"
watch_split ood_modes_thiophene_2mer scripts/revision/submit_water_orca.sbatch \
  "$CONCURRENT_SMALL" 0 1 "$PARTITION"
watch_split ood_modes_thiophene_12mer scripts/revision/submit_water_orca_large.sbatch \
  "$CONCURRENT_LARGE" 1 1 "${PARTITION_LARGE:-cs}"
