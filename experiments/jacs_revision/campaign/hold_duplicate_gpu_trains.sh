#!/bin/bash
# Hold pending duplicate gpu48 water trains so the next freed GPU can go to
# short terthiophene MH instead of another n50/cutoff clone.
#
# This cluster returns "Unspecified error" for `scontrol hold` and
# `scontrol update Nice=` on the user's own jobs (tried 2026-09-11).
# QOSMaxGRESPerUser is the running GPU count; hold would not free a GPU
# anyway. Re-run this after a site-policy change.
set -uo pipefail
HOLD_IDS=(
  17298427 17299832 17299824 17320572
  17329893 17329877 17329857 17329824
  17327345 17327344 17329633 17329624
  17327347 17327346 17329675 17329674
  17327349 17327348
  17327355 17327354 17332758 17332786
  17327358 17327357 17329734 17329733 17332849 17332805
  17299837 17303939 17303924 17305649 17329988
)
ok=0
fail=0
for j in "${HOLD_IDS[@]}"; do
  if scontrol hold "$j" 2>/dev/null; then
    echo "held $j"
    ok=$((ok+1))
  else
    echo "cannot hold $j (no user hold/Nice on this QOS)"
    fail=$((fail+1))
  fi
done
echo "held=$ok failed=$fail"
echo "Do not hold running ensemble 17330001/015/022 or PBE size-curve runners."
