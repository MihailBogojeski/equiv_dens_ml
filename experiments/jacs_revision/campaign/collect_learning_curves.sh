#!/bin/bash
# Merge per-eval JSON into the two files the response-letter figures read.
#
# Usage:  bash scripts/revision/collect_learning_curves.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

PY=./.venv/bin/python
mkdir -p results/revision/eval

$PY scripts/revision/analysis/score_learning_curve_ckpts.py collect \
  --kind ntrain \
  --glob "results/revision/eval/learning_curves/ntrain_*.json" \
  --out results/revision/eval/learning_curves_ntrain.json

$PY scripts/revision/analysis/score_learning_curve_ckpts.py collect \
  --kind step \
  --glob "results/revision/eval/learning_curves/step_*.json" \
  --out results/revision/eval/learning_curves_step.json
