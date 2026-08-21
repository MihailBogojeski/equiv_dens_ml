#!/bin/bash
# Re-run every number that depends on the published ethanol energy head (R1.2,
# R2.4, R2.11), now that config/training/eval_ethanol_paper.txt agrees with the
# 2024-03-22_96w7KyGG checkpoint.
#
# These were all reported with a "not physical" caveat because the loader had
# been silently building an energy head the checkpoint did not fit: L0_start
# added untrained radial_L0_map layers, append_atom_density doubled the L=0
# input width, and order_en / num_en_basis_functions disagreed. The checkpoint
# now loads with zero missing and zero unexpected tensors, so the caveat should
# come off -- or, if these numbers stay wrong, the cause is something else and
# we want to see that clearly rather than infer it.
#SBATCH --job-name=energy-head-evals
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=gpu48
#SBATCH --partition=a100_chemistry,h200_public,l40s_public
#SBATCH --output=logs/energy_head_evals_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE=offline

PY=./.venv/bin/python
ARGS=config/training/eval_ethanol_paper.txt
MODEL=paper/models/ethanol/2024-03-22_96w7KyGG
mkdir -p results/revision logs

echo "=== R2.11 geometry optimisation vs DFT ==="
$PY scripts/revision/optimize_densnet.py \
  --structure datasets/ethanol_train_10.xyz \
  --model "$MODEL" --args-file "$ARGS" \
  --dft-xc pbe --use-gpu \
  --out results/revision/geoopt_ethanol_densnet.json

echo
echo "=== R1.2 ethanol OOD energies and forces ==="
$PY scripts/training/eval_model_npy.py "$ARGS" \
  datasets/revision/ood/ethanol_ood_pyscf_augccpvdz_pbe.npy \
  --batch_size 1 2>&1 | tail -40

echo
echo "=== R2.4 NVE energy conservation ==="
$PY scripts/revision/nve_energy_drift.py \
  --hdf5 results/revision/mlip_gpu_densnet/densnet_nve.jsonl \
  --n-atoms 9 \
  --out results/revision/nve_drift_gpu.json

echo
echo "finished $(date -Is)"
