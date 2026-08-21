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
#
# CPU, not GPU. Ethanol is nine atoms: the published CPU benchmark is 1.62 s per
# single point, so the 190-frame OOD set is minutes and the BFGS relaxation is
# less, while the NVE drift is post-processing of a trajectory already on disk
# and touches no model at all. Asking for a GPU put this behind every training
# run in the account under a shared QOSGrpGRES cap, waiting hours to do a few
# minutes of arithmetic.
#SBATCH --job-name=energy-head-evals
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_283_chemistry
#SBATCH --qos=cpu48
#SBATCH --output=logs/energy_head_evals_%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# The login shell exports a CUDA preload absent from the CPU partition, and
# every child prints a loader error before doing anything.
unset LD_PRELOAD
export CUDA_VISIBLE_DEVICES=""
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
  --dft-xc pbe \
  --out results/revision/geoopt_ethanol_densnet.json

echo
echo "=== R1.2 ethanol OOD energies and forces ==="
# score_ood_forces.py, not eval_model_npy.py: the latter sets energy_weight and
# forces_weight to zero, so it never builds the energy head and cannot report a
# force error at all. Its --args-file default is the MD config, which now
# matches the checkpoint.
$PY scripts/revision/score_ood_forces.py \
  --dens datasets/revision/ood/ethanol_ood_pyscf_augccpvdz_pbe.npy \
  --model "$MODEL" --args-file config/md/nn/ethanol_500ps.txt \
  --out results/revision/eval_ethanol_ood_forces.json 2>&1 | tail -25

echo
echo "=== R2.4 NVE energy conservation ==="
$PY scripts/revision/nve_energy_drift.py \
  --hdf5 results/revision/mlip_gpu_densnet/densnet_nve.jsonl \
  --n-atoms 9 \
  --out results/revision/nve_drift_gpu.json

echo
echo "finished $(date -Is)"
