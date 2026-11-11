#!/bin/bash
# Local Phase A / Phase B geometry expansion (GFN2 Langevin, not DFT).
# Sets one OpenMP thread per worker so tblite does not oversubscribe.
#
# Usage:
#   scripts/revision/water/run_geometry_expansion.sh phase_a
#   scripts/revision/water/run_geometry_expansion.sh phase_b_parents
#   scripts/revision/water/run_geometry_expansion.sh phase_b_md

set -euo pipefail
cd "$(dirname "$0")/../../.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p logs/revision

STAGE=${1:-phase_a}
WORKERS=${WORKERS:-4}

case "$STAGE" in
  phase_a)
    exec ./.venv/bin/python -u scripts/revision/water/thicken_with_md.py \
      --calculator auto --ice-mode rigid --dedup parent --workers "$WORKERS"
    ;;
  phase_b_parents)
    exec ./.venv/bin/python -u scripts/revision/water/generate_new_motif_parents.py --water --ice --droplet --malon
    ;;
  phase_b_md)
    exec ./.venv/bin/python -u scripts/revision/water/thicken_with_md.py \
      --water-dir datasets/revision/water_clusters_newmotifs \
      --ood-dir datasets/revision/water_ood_newmotifs \
      --malon-dir datasets/revision/malonaldehyde_newmotifs \
      --water-out datasets/revision/water_clusters_newmotifs_md10x \
      --ood-out datasets/revision/water_ood_newmotifs_md10x \
      --malon-out datasets/revision/malonaldehyde_newmotifs_md10x \
      --calculator auto --ice-mode rigid --dedup parent --workers "$WORKERS"
    ;;
  phase_md40x)
    exec ./.venv/bin/python -u scripts/revision/water/thicken_with_md.py \
      --n-keep 40 --stride 80 --workers "$WORKERS" \
      --calculator auto --ice-mode rigid --dedup parent \
      --water-out datasets/revision/water_clusters_md40x \
      --ood-out datasets/revision/water_ood_md40x \
      --malon-out datasets/revision/malonaldehyde_md40x
    ;;
  phase_b_md40x)
    exec ./.venv/bin/python -u scripts/revision/water/thicken_with_md.py \
      --n-keep 40 --stride 80 --workers "$WORKERS" \
      --water-dir datasets/revision/water_clusters_newmotifs \
      --ood-dir datasets/revision/water_ood_newmotifs \
      --malon-dir datasets/revision/malonaldehyde_newmotifs \
      --water-out datasets/revision/water_clusters_newmotifs_md40x \
      --ood-out datasets/revision/water_ood_newmotifs_md40x \
      --malon-out datasets/revision/malonaldehyde_newmotifs_md40x \
      --calculator auto --ice-mode rigid --dedup global
    ;;
  *)
    echo "unknown stage: ${STAGE} (phase_a|phase_b_parents|phase_b_md|phase_md40x|phase_b_md40x)" >&2
    exit 1
    ;;
esac
