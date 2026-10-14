#!/usr/bin/env bash
set -euo pipefail
report=${DENSNET_REPORT_DIR:?Report directory missing}
reference=/home/johannes/equiv_dens_ml-paper-reference
candidate=/home/johannes/equiv_dens_ml-performance-precision
runtime=/home/johannes/equiv_dens_ml/tools/runtime/run.sh
cd "$candidate"
export EQUIV_DENS_PROJECT_ROOT=$candidate
git diff --binary -- src > "$report/frozen-candidate.patch"
rg --files -0 src benchmarks tests | sort -z | xargs -0 sha256sum \
    > "$report/frozen-candidate-and-instruments.sha256"
"$runtime" torch python -m pytest tests -q -ra \
    --junitxml="$report/full-contracts.xml" > "$report/full-contracts.log" 2>&1
"$runtime" torch python -m benchmarks.pipeline --source-root "$reference" \
    generate --output "$report/tiny-fp64.pt"
"$runtime" torch python -m benchmarks.pipeline --source-root "$reference" --device cuda \
    snapshot --fixture "$report/tiny-fp64.pt" --output "$report/reference-cuda.pt"
"$runtime" torch python -m benchmarks.pipeline --source-root "$candidate" --device cuda \
    --density-backend contracted compare --fixture "$report/tiny-fp64.pt" \
    --reference "$report/reference-cuda.pt" --output "$report/contracted-cuda.json"
for grid in 1000 30000 100000; do
    for mode in rho rho-grad backward mixed density-training; do
        "$runtime" torch python -m benchmarks.density_evaluation \
            --reference-root "$reference" --device cuda --dtype float32 \
            --grid-points "$grid" --mode "$mode" --rounds 5 --warmup 3 --iterations 10 \
            --output "$report/grid-$grid-$mode-cuda.json"
    done
done
for mode in rho-grad backward; do
    "$runtime" torch python -m benchmarks.density_evaluation \
        --reference-root "$reference" --device cuda --dtype float32 --precision-policy tf32 \
        --grid-points 30000 --mode "$mode" --rounds 5 --warmup 3 --iterations 10 \
        --output "$report/grid-30000-$mode-tf32-cuda.json"
done
for grid in 1000 30000; do
    "$runtime" torch python -m benchmarks.pipeline --source-root "$reference" \
        generate --dtype float32 --grid-points "$grid" --output "$report/grid-$grid-fp32.pt"
    for mode in density density_training; do
        "$runtime" torch python -m benchmarks.paired_pipeline \
            --reference-root "$candidate" --candidate-root "$candidate" \
            --candidate-density-backend contracted --device cuda --mode "$mode" \
            --fixture "$report/grid-$grid-fp32.pt" --rounds 5 --warmup 3 --iterations 10 \
            --output "$report/adam-grid-$grid-$mode-cuda.json"
    done
done
