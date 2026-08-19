#!/bin/bash
# One watchdog for the whole water/OOD labelling campaign.
#
# The campaign is 15 arrays across 10 splits and 2 levels of theory, spread over
# a 4 h small tier and a 24 h large tier. Running 15 separate watchdogs would
# mean 15 processes to keep alive; instead this walks every shard directory that
# exists, re-submits only the shards still outstanding in each, and sleeps.
#
# Splits are discovered from the shard tree rather than listed here, so a split
# added later by launch_water_campaign.sh is picked up with no edit.
#
# Usage:  scripts/revision/watch_water_campaign.sh [INTERVAL_S]

set -euo pipefail
cd "$(dirname "$0")/../.."

INTERVAL=${1:-1800}
SHARD_ROOT=${SHARD_ROOT:-datasets/revision/shards}
OUT_ROOT=${OUT_ROOT:-results/revision/water_orca}
MAX_ROUNDS=${MAX_ROUNDS:-500}

# The large tier is the set of splits whose frames need 16 cores and a 24 h
# window; everything else fits the small tier. Keep in sync with
# launch_water_campaign.sh, which does the same split when it first submits.
is_large() {
  case "$1" in
    water_ood_size_large | water_ood_order | water_ood_density) return 0 ;;
    *) return 1 ;;
  esac
}

for ((round = 1; round <= MAX_ROUNDS; round++)); do
  echo "=========== [$(date -Is)] campaign round ${round} ==========="
  ./.venv/bin/python scripts/revision/campaign_status.py \
    --root "$SHARD_ROOT" --out-root "$OUT_ROOT" || true

  for shard_dir in "$SHARD_ROOT"/*/*/; do
    shard_dir=${shard_dir%/}
    compgen -G "${shard_dir}/shard_*.json" > /dev/null || continue
    theory=$(basename "$shard_dir")
    split=$(basename "$(dirname "$shard_dir")")
    outdir="${OUT_ROOT}/${split}/${theory}"

    if is_large "$split"; then
      sbatch_file=scripts/revision/submit_water_orca_large.sbatch
      concurrent=${CONCURRENT_LARGE:-60}
    else
      sbatch_file=scripts/revision/submit_water_orca.sbatch
      concurrent=${CONCURRENT_SMALL:-200}
    fi

    ONESHOT=1 SBATCH_FILE="$sbatch_file" CONCURRENT="$concurrent" \
      scripts/revision/watch_and_resubmit.sh "$shard_dir" "$outdir" "$theory" || true
  done

  # Training is launched from here rather than by --dependency on the labelling
  # arrays: the arrays alive at submission time are not the ones that finish the
  # campaign, so a dependency on them fires while most shards are still
  # unlabelled. The marker makes this fire exactly once per level of theory.
  for theory_dir in "$SHARD_ROOT"/*/*/; do
    theory=$(basename "${theory_dir%/}")
    marker="${OUT_ROOT}/.train_submitted_${theory}"
    [[ -e "$marker" ]] && continue
    outstanding=$(./.venv/bin/python scripts/revision/campaign_status.py \
      --root "$SHARD_ROOT" --out-root "$OUT_ROOT" --json 2>/dev/null |
      ./.venv/bin/python -c "
import json, sys
rows = json.load(sys.stdin)
print(sum(len(r['outstanding']) for r in rows if r['theory'] == '${theory}'))
" 2>/dev/null || echo 1)
    if [[ "$outstanding" == "0" ]]; then
      echo "[$(date -Is)] ${theory} labels complete; submitting training"
      mkdir -p "$OUT_ROOT"
      THEORY="$theory" sbatch --job-name="train-${theory}" \
        scripts/revision/submit_water_train.sbatch && touch "$marker"
    fi
  done

  if [[ "$round" -eq "$MAX_ROUNDS" ]]; then
    break
  fi
  sleep "$INTERVAL"
done
