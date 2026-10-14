#!/usr/bin/env bash
set -euo pipefail
report=${DENSNET_REPORT_DIR:?Report directory missing}
reference=/home/johannes/equiv_dens_ml-paper-reference
candidate=/home/johannes/equiv_dens_ml
runtime=tools/runtime/run.sh
"$runtime" torch python -m pytest tests/performance_contracts/test_geometry.py \
    tests/performance_contracts/test_packing.py tests/performance_contracts/test_kernels.py \
    -q -ra --junitxml="$report/contracts.xml" > "$report/contracts.log" 2>&1
"$runtime" torch python -m benchmarks.pipeline --source-root "$reference" \
    generate --output "$report/tiny-fp64.pt"
"$runtime" torch python -m benchmarks.pipeline --source-root "$reference" --device cuda \
    snapshot --fixture "$report/tiny-fp64.pt" --output "$report/reference-cuda.pt"
"$runtime" torch python -m benchmarks.pipeline --source-root "$candidate" --device cuda \
    compare --fixture "$report/tiny-fp64.pt" --reference "$report/reference-cuda.pt" \
    --output "$report/candidate-broadcast-cuda.json"
"$runtime" torch python -m benchmarks.pipeline --source-root "$candidate" --device cuda \
    --kernel-backend matmul compare --fixture "$report/tiny-fp64.pt" \
    --reference "$report/reference-cuda.pt" --output "$report/candidate-matmul-cuda.json"
for device in cpu cuda; do
    "$runtime" torch python benchmarks/geometry.py --reference-root "$reference" \
        --allocation "$SLURM_JOB_ID" --device "$device" --iterations 50 \
        --container-sha256 9ae7deabc4766fc81897a93a8de4d8aa1cfff385d32fc4cd78ffbea1a5d6b9e4 \
        --output "$report/geometry-$device.json"
done
for mode in codec head-inference head-backward head-mixed; do
    "$runtime" torch python -m benchmarks.packing --reference-root "$reference" \
        --device cuda --mode "$mode" --iterations 30 --output "$report/packing-$mode-cuda.json"
done
for mode in forward backward mixed; do
    "$runtime" torch python -m benchmarks.kernels --device cuda --kind pair \
        --mode "$mode" --order 3 --width 128 --rows 32 --iterations 30 \
        --output "$report/kernels-pair-$mode-cuda.json"
done
"$runtime" torch python -m benchmarks.kernels --device cuda --kind self \
    --mode forward --order 3 --width 128 --rows 32 --iterations 30 \
    --output "$report/kernels-self-forward-cuda.json"
