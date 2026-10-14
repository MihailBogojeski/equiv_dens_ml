#!/usr/bin/env bash
set -euo pipefail
report=${DENSNET_REPORT_DIR:?Report directory missing}
reference=/home/johannes/equiv_dens_ml-paper-reference
candidate=/home/johannes/equiv_dens_ml
measured=/home/johannes/equiv_dens_ml-performance-precision
wide_fixture=$candidate/docs/analysis/densnet-performance/r4-energy-orchestration/wide-norm2-fp32.pt
runtime=$candidate/tools/runtime/run.sh
cd "$candidate"
export EQUIV_DENS_PROJECT_ROOT=$candidate
git diff --binary -- src > "$report/final-candidate.patch"
rg --files -0 src benchmarks tests | sort -z | xargs -0 sha256sum \
    > "$report/final-candidate-and-instruments.sha256"
"$runtime" torch python -m pytest tests -q -ra \
    --junitxml="$report/full-contracts.xml" > "$report/full-contracts.log" 2>&1
"$runtime" torch python -m benchmarks.pipeline --source-root "$reference" \
    generate --output "$report/tiny-fp64.pt"
"$runtime" torch python -m benchmarks.pipeline --source-root "$reference" --device cuda \
    snapshot --fixture "$report/tiny-fp64.pt" --output "$report/reference-cuda.pt"
"$runtime" torch python -m benchmarks.pipeline --source-root "$candidate" --device cuda \
    compare --fixture "$report/tiny-fp64.pt" --reference "$report/reference-cuda.pt" \
    --output "$report/default-cuda.json"
for arm in paper measured-legacy; do
    selected=$reference
    if [[ $arm == measured-legacy ]]; then selected=$measured; fi
    for repeat in 1 2; do
        "$runtime" torch python - "$selected" "$wide_fixture" "$report/$arm-repeat-$repeat.pt" <<'PY'
import sys
from pathlib import Path
import torch
from benchmarks.fixtures import select_source, load_fixture
root = select_source(sys.argv[1])
from benchmarks import pipeline
from benchmarks.paired_pipeline import diagnostic_envelope
from benchmarks.precision import use_full_precision
use_full_precision()
torch.set_num_threads(1)
fixture = load_fixture(sys.argv[2])
contract = pipeline.evaluate_contract(fixture, 'force_training', True, 'cuda')
diagnostic = diagnostic_envelope(fixture, 'force_training', contract)
diagnostic['provenance'] = pipeline.provenance(root, fixture)
torch.save(diagnostic, Path(sys.argv[3]))
PY
    done
done
"$runtime" torch python - "$report" <<'PY'
import json
import sys
from pathlib import Path
import torch
from benchmarks.pipeline import compare_snapshots
folder = Path(sys.argv[1])
results = {}
for arm in ('paper', 'measured-legacy'):
    first = torch.load(folder/f'{arm}-repeat-1.pt', weights_only=False)
    second = torch.load(folder/f'{arm}-repeat-2.pt', weights_only=False)
    results[arm] = compare_snapshots(first, second, 'float32')
(folder/'reference-repeat-comparison.json').write_text(json.dumps(results, indent=2)+'\n')
# These are registered pass/fail diagnostic outcomes, not a prerequisite that
# silently rejects a negative reference-repeat result or relaxes its threshold.
PY
sha256sum -c "$report/final-candidate-and-instruments.sha256" \
    > "$report/final-source-unchanged.log"
