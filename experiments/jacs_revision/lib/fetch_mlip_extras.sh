#!/usr/bin/env bash
# Install the optional Reviewer 3 MLIPs without allowing pip to replace torch.
set -euo pipefail

cd "$(dirname "$0")/../../.."
PYTHON=${PYTHON:-./.venv/bin/python}
PIP=("$PYTHON" -m pip)

torch_before=$("$PYTHON" -c 'import torch; print(torch.__version__)')

# MACE-POLAR requires graph_longrange. MACE itself is installed without
# dependency resolution so the campaign's working torch build cannot move.
"${PIP[@]}" install --no-deps "mace-torch>=0.3.16"
"${PIP[@]}" install \
  "git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0"

# SO3LR pins JAX 0.5.3 and carries its remaining JAX ecosystem dependencies in
# its own pyproject. The CUDA extra supplies the matching CUDA 12 plugin.
"${PIP[@]}" install "jax[cuda12]==0.5.3"
"${PIP[@]}" install \
  "git+https://github.com/general-molecular-simulations/so3lr.git"
"${PIP[@]}" install memspectrum

torch_after=$("$PYTHON" -c 'import torch; print(torch.__version__)')
if [[ "$torch_after" != "$torch_before" ]]; then
  echo "ERROR: torch changed from ${torch_before} to ${torch_after}" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
from mace.calculators import mace_polar
from so3lr import So3lrCalculator

assert callable(mace_polar)
assert So3lrCalculator is not None
print("MACE-POLAR and SO3LR imports succeeded")
PY
