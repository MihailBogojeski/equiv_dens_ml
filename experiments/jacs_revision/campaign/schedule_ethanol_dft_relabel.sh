#!/usr/bin/env bash
# Schedule ethanol DFT relabel array when a Slurm account is available.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ACCOUNT=${SLURM_ACCOUNT:-torch_pr_283_chemistry}
echo "Submitting ethanol DFT relabel array (account=${ACCOUNT})"
sbatch --account="$ACCOUNT" scripts/revision/submit_ethanol_dft_relabel.sbatch
