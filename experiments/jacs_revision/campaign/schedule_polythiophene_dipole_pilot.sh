#!/usr/bin/env bash
# Schedule polythiophene 10-mer dipole pilot array.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ACCOUNT=${SLURM_ACCOUNT:-torch_pr_283_chemistry}
echo "Submitting polythiophene dipole pilot (account=${ACCOUNT})"
sbatch --account="$ACCOUNT" scripts/revision/submit_polythiophene_dipole_pilot.sbatch
