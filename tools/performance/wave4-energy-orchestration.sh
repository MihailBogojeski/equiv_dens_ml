#!/usr/bin/env bash
set -euo pipefail
report=${DENSNET_REPORT_DIR:?Report directory missing}
reference=/home/johannes/equiv_dens_ml-paper-reference
candidate=/home/johannes/equiv_dens_ml-performance-precision
runtime=/home/johannes/equiv_dens_ml/tools/runtime/run.sh
cd "$candidate"
export EQUIV_DENS_PROJECT_ROOT=$candidate
git diff --binary -- src > "$report/frozen-candidate.patch"
rg --files -0 src benchmarks | sort -z | xargs -0 sha256sum \
    > "$report/frozen-candidate-and-instruments.sha256"
"$runtime" torch python -m benchmarks.pipeline --source-root "$reference" \
    generate --dtype float64 --modules 2 --energy-normalize 2 \
    --output "$report/normalized-fp64.pt"
"$runtime" torch python -m benchmarks.energy_orchestration --source-root "$reference" \
    --fixture "$report/normalized-fp64.pt" --output "$report/reference-cuda-fp64.pt" \
    --device cuda --finite-differences --fd-steps 1e-3 3e-4 1e-4 3e-5 1e-5 3e-6 1e-6
"$runtime" torch python -m benchmarks.energy_orchestration --source-root "$candidate" \
    --fixture "$report/normalized-fp64.pt" --backend device \
    --output "$report/device-cuda-fp64.pt" --device cuda --finite-differences \
    --fd-steps 1e-3 3e-4 1e-4 3e-5 1e-5 3e-6 1e-6
"$runtime" torch python - "$report" <<'PY'
import json
import sys
from pathlib import Path
import torch
from benchmarks.pipeline import compare_snapshots
folder = Path(sys.argv[1])
reference = torch.load(folder/'reference-cuda-fp64.pt', weights_only=False)
candidate = torch.load(folder/'device-cuda-fp64.pt', weights_only=False)
result = compare_snapshots(reference, candidate, 'float64')
(folder/'fp64-comparison.json').write_text(json.dumps(result, indent=2)+'\n')
assert result['passed'], result['failures']
PY
for case in tiny wide; do
    for normalize in 2 0; do
        options=(--modules 2)
        if [[ $case == wide ]]; then options=(--features 128 --order 3 --modules 3); fi
        "$runtime" torch python -m benchmarks.pipeline --source-root "$reference" \
            generate --case "$case" --dtype float32 "${options[@]}" \
            --energy-normalize "$normalize" --output "$report/$case-norm$normalize-fp32.pt"
        for mode in ef force_training; do
            "$runtime" torch python -m benchmarks.paired_pipeline \
                --reference-root "$candidate" --candidate-root "$candidate" \
                --candidate-energy-backend device --device cuda --mode "$mode" \
                --fixture "$report/$case-norm$normalize-fp32.pt" \
                --rounds 5 --warmup 5 --iterations 20 \
                --output "$report/$case-norm$normalize-$mode-cuda.json"
        done
    done
done
