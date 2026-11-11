#!/usr/bin/env bash
# Open-source PyMOL for headless structure renders (not the incentive binary).
set +x
set -euo pipefail
PREFIX=/scratch/mh7373/projects/equiv_dens_ml/.conda/pymol-open-source
if [ -x "${PREFIX}/bin/pymol" ]; then
  echo "already installed: ${PREFIX}/bin/pymol"
  exit 0
fi
if [ -x /share/apps/anaconda3/bin/conda ]; then
  /share/apps/anaconda3/bin/conda create -y -p "${PREFIX}" -c conda-forge pymol-open-source
elif command -v pixi >/dev/null; then
  mkdir -p "${PREFIX}"
  ( cd /scratch/mh7373/projects/equiv_dens_ml/.conda
    pixi init pymol-os --format pixi
    cd pymol-os
    pixi add pymol-open-source
  )
else
  echo "no conda or pixi" >&2
  exit 1
fi
ls -l "${PREFIX}/bin/pymol" /scratch/mh7373/projects/equiv_dens_ml/.conda/pymol-os/.pixi/envs/default/bin/pymol 2>/dev/null || true
