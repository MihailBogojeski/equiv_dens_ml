#!/usr/bin/env bash
# Evaluate PBE cutoff-sweep checkpoints on ID test + size OOD (R3.5).
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=./.venv/bin/python
LABELS=datasets/revision/water_labels
OUT=results/revision/eval/cutoff_sweep
mkdir -p "$OUT"

eval_one() {
  local cut=$1
  local run_dir
  run_dir=$(scripts/revision/lib/latest_run_dir.sh "results/revision/water_pbe_orca_cutoff_${cut}")
  if [[ -z "$run_dir" ]]; then
    echo "cutoff ${cut} Å: no trained run yet"
    return 1
  fi
  echo "=== cutoff ${cut} Å (${run_dir}) ==="
  for split in water_id_test water_ood_size; do
    local out="${OUT}/cutoff_${cut}_${split}.json"
    $PY scripts/revision/csh/csh_evaluate.py \
      --args-file "${run_dir}/args.txt" \
      --run-dir "$run_dir" \
      --np-dataset "${LABELS}/${split}_pbe_d4_avdz_base.npy" \
      --dens-dataset "${LABELS}/${split}_pbe_d4_avdz.npy" \
      --elements 1,8 \
      --label "cutoff_${cut}_${split}" \
      --out "$out"
  done
}

for cut in 4 5 6 8; do
  eval_one "$cut" || true
done

$PY - <<'PY'
import json
from pathlib import Path

root = Path("results/revision/eval/cutoff_sweep")
rows = []
for cut in (4, 5, 6, 8):
    row = {"cutoff_angstrom": cut, "splits": {}}
    for split in ("water_id_test", "water_ood_size"):
        p = root / f"cutoff_{cut}_{split}.json"
        if p.is_file():
            s = json.loads(p.read_text()).get("summary", {})
            row["splits"][split] = {
                "afe_mean": s.get("afe_mean"),
                "afe_median": s.get("afe_median"),
                "n_structures": s.get("n_structures"),
            }
    rows.append(row)
out = Path("results/revision/cutoff_sweep_summary.json")
out.write_text(json.dumps({"cutoffs": rows}, indent=2) + "\n")
print(f"wrote {out}")
PY
