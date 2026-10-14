#!/usr/bin/env bash
set -euo pipefail
report=${DENSNET_REPORT_DIR:?Report directory missing}
reference=/home/johannes/equiv_dens_ml-paper-reference
candidate=/home/johannes/equiv_dens_ml-performance-snapshot
runtime=/home/johannes/equiv_dens_ml/tools/runtime/run.sh
# Freeze benchmark instruments as well as scientific source before dispatch.
cd "$candidate"
export EQUIV_DENS_PROJECT_ROOT=$candidate
git diff --binary -- src > "$report/frozen-candidate.patch"
rg --files -0 src benchmarks tests/performance_contracts | sort -z | xargs -0 sha256sum \
    > "$report/frozen-candidate-and-instruments.sha256"
"$runtime" torch python -m pytest tests/performance_contracts/test_kernels.py \
    tests/performance_contracts/test_paired_pipeline.py -q -ra \
    --junitxml="$report/instrument-contracts.xml" > "$report/instrument-contracts.log" 2>&1
"$runtime" torch python -m benchmarks.pipeline --source-root "$reference" \
    generate --dtype float32 --output "$report/tiny-fp32.pt"
"$runtime" torch python -m benchmarks.pipeline --source-root "$reference" \
    generate --case wide --dtype float32 --features 32 --order 2 --modules 2 \
    --output "$report/wider-fp32.pt"
"$runtime" torch python -m benchmarks.calculator --fixture "$report/tiny-fp32.pt" \
    --reference-root "$reference" --allocation "$SLURM_JOB_ID" --device cuda \
    --dtype float32 --rounds 5 --iterations 20 --warmup 5 \
    --container-sha256 9ae7deabc4766fc81897a93a8de4d8aa1cfff385d32fc4cd78ffbea1a5d6b9e4 \
    --output "$report/calculator-cuda.json"
for case in tiny wider; do
    for mode in coefficients density ef force_training; do
        "$runtime" torch python -m benchmarks.paired_pipeline \
            --reference-root "$reference" --candidate-root "$candidate" \
            --fixture "$report/$case-fp32.pt" --mode "$mode" --device cuda \
            --rounds 5 --iterations 10 --warmup 3 \
            --output "$report/$case-$mode-cuda.json"
    done
done
