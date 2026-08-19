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

# How much memory a task in this split actually needs.
#
# Worth getting right because memory, not cores, is what caps the campaign: the
# cpu48 QOS allows 3000 cores but only 6000 GB, so at the original flat 48 GB the
# account could hold 125 tasks (1000 cores) and no more, while the CPU limit
# would have allowed 375. Most of the campaign is small: 1250 of the 2967 frames
# are n=2-6 clusters, and the calibration measured 6.8 GB peak RSS for the
# largest of those at wB97M-V, with malonaldehyde far below that. Sizing per
# split rather than for the worst frame in the campaign roughly triples how many
# of them fit.
#
# The values keep a wide margin over the measured peaks; an underestimate costs
# an OOM kill, which the per-frame resume recovers from but which wastes the
# frame in flight.
mem_for() {
  case "$1" in
    malonaldehyde_*) echo 12G ;;                     # 9 atoms, ~250 AO
    water_train_small | water_val_small | water_id_test_small) echo 16G ;;  # n=2-6, 6.8 GB measured
    water_ood_size_small) echo 32G ;;                # n=8,10
    *) echo "" ;;                                    # large tier keeps the script's own 96G
  esac
}

# The order splits are offered to the scheduler.
#
# This is not cosmetic. The QOS caps the account at 6000 GB, which the campaign
# reaches long before its 3000-core limit, so at any moment only some splits fit
# and the rest wait. Slurm breaks ties by job age, so whichever split is
# submitted first in a round is the one that gets the memory.
#
# Iterating the shard tree alphabetically put `water_train_small` ninth of ten,
# behind every OOD tier. The result was the campaign's one indispensable split --
# nothing can be trained without it, and every OOD number is an error of a model
# that does not otherwise exist -- sitting at 0 of 65 shards while 75 tasks of
# the tier the plan itself calls the weakest held 3600 GB.
#
# So: train and validation first, then the malonaldehyde pair and its scan
# (nine atoms, nearly free, and the physically interpretable axis of the headline
# figure), then the in-distribution control, then the OOD tiers cheapest first.
split_rank() {
  case "$1" in
    water_train_small) echo 0 ;;
    water_val_small) echo 1 ;;
    malonaldehyde_train) echo 2 ;;
    malonaldehyde_val) echo 3 ;;
    malonaldehyde_ood_proton_transfer) echo 4 ;;
    water_id_test_small) echo 5 ;;
    water_ood_size_small) echo 6 ;;
    water_ood_size_large) echo 7 ;;
    water_ood_order) echo 8 ;;
    water_ood_density) echo 9 ;;
    *) echo 99 ;;
  esac
}

for ((round = 1; round <= MAX_ROUNDS; round++)); do
  echo "=========== [$(date -Is)] campaign round ${round} ==========="
  ./.venv/bin/python scripts/revision/campaign_status.py \
    --root "$SHARD_ROOT" --out-root "$OUT_ROOT" || true

  # Sorted by rank rather than taken in glob order; see split_rank above for why
  # the order decides which splits get the account's memory.
  ordered=()
  for shard_dir in "$SHARD_ROOT"/*/*/; do
    shard_dir=${shard_dir%/}
    compgen -G "${shard_dir}/shard_*.json" > /dev/null || continue
    ordered+=("$(split_rank "$(basename "$(dirname "$shard_dir")")") ${shard_dir}")
  done

  while read -r _rank shard_dir; do
    [[ -n "$shard_dir" ]] || continue
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

    mem=$(mem_for "$split")
    ONESHOT=1 SBATCH_FILE="$sbatch_file" CONCURRENT="$concurrent" \
      SBATCH_EXTRA="${mem:+--mem=$mem}" \
      scripts/revision/watch_and_resubmit.sh "$shard_dir" "$outdir" "$theory" || true
  done < <(printf '%s\n' "${ordered[@]+"${ordered[@]}"}" | sort -n -k1,1)

  # Training is launched from here rather than by --dependency on the labelling
  # arrays: the arrays alive at submission time are not the ones that finish the
  # campaign, so a dependency on them fires while most shards are still
  # unlabelled. Training is then kept alive the same way the labelling is -- a
  # run whose job hit the walltime has no .training_done marker and is
  # re-submitted, resuming from its checkpoint.
  # Shards of `theory` still unlabelled across the given splits, or a positive
  # number if that cannot be determined.
  #
  # Counts `n_shards - n_done`, not the `outstanding` list. Those are different
  # questions: `outstanding` is "what needs submitting", so a shard a live worker
  # is part-way through counts as zero, and gating on it launches training on
  # labels that are still being computed. Only `--min-complete` stood between
  # that and a model trained on a dataset missing whatever had not landed yet.
  #
  # Per-run rather than per-theory, because a run does not read the whole
  # campaign. Training consumes train, validation and the in-distribution test
  # set; the OOD tiers exist to be evaluated afterwards by csh_evaluate.py, and
  # the ice and droplet tiers are the slowest arrays in the campaign by a wide
  # margin -- 24 h tasks on the large tier. Waiting for them before starting
  # training would idle the GPU for days while the data it actually optimises
  # against sat finished on disk, and would push the headline figure out by the
  # same amount for no benefit.
  unlabelled_for() {
    local theory=$1
    shift
    ./.venv/bin/python scripts/revision/campaign_status.py \
      --root "$SHARD_ROOT" --out-root "$OUT_ROOT" --json 2>/dev/null |
      SPLITS="$*" THEORY="$theory" ./.venv/bin/python -c "
import json, os, sys
splits = set(os.environ['SPLITS'].split())
theory = os.environ['THEORY']
rows = [r for r in json.load(sys.stdin)
        if r['theory'] == theory and r['split'] in splits]
# No rows means the shard tree for this run does not exist yet, which is not
# the same as being finished.
print(sum(r['n_shards'] - r['n_done'] for r in rows) if len(rows) == len(splits) else 1)
" 2>/dev/null || echo 1
  }

  #: What each training run actually reads, as a list of shard-tree splits.
  WATER_INPUTS="water_train_small water_val_small water_id_test_small"
  MALON_INPUTS="malonaldehyde_train malonaldehyde_val"

  theories=$(for d in "$SHARD_ROOT"/*/*/; do basename "${d%/}"; done | sort -u)
  for theory in $theories; do
    runs=(water malonaldehyde)
    if [[ "$theory" == "pbe_d4_avdz" ]]; then
      runs+=(cutoff_4 cutoff_5 cutoff_6 cutoff_8)
    fi
    for run in "${runs[@]}"; do
      case "$run" in
        water) name="water_${theory}"; inputs="$WATER_INPUTS" ;;
        malonaldehyde) name="malonaldehyde_${theory}"; inputs="$MALON_INPUTS" ;;
        cutoff_*) name="water_pbe_orca_cutoff_${run#cutoff_}"; inputs="$WATER_INPUTS" ;;
      esac
      [[ -e "results/revision/${name}/.training_done" ]] && continue
      left=$(unlabelled_for "$theory" $inputs)
      if [[ "$left" != "0" ]]; then
        echo "[$(date -Is)] ${name}: ${left} shards of its inputs still unlabelled"
        continue
      fi
      job="tr-${run}-${theory}"
      queued=$(squeue -h -u "$USER" -n "$job" -o "%i" 2>/dev/null | wc -l)
      [[ "$queued" -gt 0 ]] && continue
      echo "[$(date -Is)] submitting ${job}"
      THEORY="$theory" RUN="$run" sbatch --job-name="$job" \
        scripts/revision/submit_water_train.sbatch || true
    done

    # The analysis is cheap and incremental -- it skips whatever it has already
    # produced -- so it runs on every pass rather than waiting for a signal that
    # everything is finished. That means the error-versus-distance curve appears
    # as soon as the first model has a checkpoint and refreshes as the rest
    # arrive, instead of depending on someone remembering to run it at the end.
    if [[ "${RUN_ANALYSIS:-1}" -ne 0 ]]; then
      scripts/revision/run_ood_analysis.sh "$theory" \
        >> "logs/ood_analysis_${theory}.log" 2>&1 || true
    fi
  done

  if [[ "$round" -eq "$MAX_ROUNDS" ]]; then
    break
  fi
  sleep "$INTERVAL"
done
