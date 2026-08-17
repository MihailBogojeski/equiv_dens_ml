# Molecular Dynamics and Dipole Computation

MD workflows: NN-based MD, g-xtb MD, AIMD, dipole recomputation, and IR spectra.

## NN-based MD (trained models)

Uses a trained model for energy and forces. Configs in `config/md/nn/`.

```bash
python run.py md @config/md/nn/polythiophene_2mer.txt
python run.py md @config/md/nn/ethanethiol.txt
```

## g-xtb MD (no model)

Uses g-xtb as calculator. No trained model required. Configs in `config/md/gxtb/`.

Requires g-xtb binary and `$GXTBHOME`. See [g-xtb/README.md](../g-xtb/README.md).

```bash
python run.py gxtb-md @config/md/gxtb/gxtb_polythiophene_2mer.txt
```

## AIMD (ab initio MD)

PySCF/gpu4pyscf with PBE/aug-cc-pVDZ and D4 dispersion. GPU acceleration when gpu4pyscf is installed.

```bash
python scripts/md/aimd_gpu4pyscf.py --structure path/to/init.xyz --output aimd.traj --steps 1000 --temperature 300 --ensemble nvt
```

## MACE-OFF MD (off-the-shelf organic MLIP)

MACE-OFF23 is a transferable organic force field (JACS 2024) covering H, C, N, O, P, S, F, Cl, Br, I — suitable for polythiophene and related molecules. Requires `pip install mace-torch`.

Single run:

```bash
python scripts/md/mace_off_md_run.py --structure datasets/thiophene2mer_init.npy --output mace_off.traj --steps 2000 --temperature 300 --model medium --device cuda
```

Scaling benchmark (n = 1–6, 4 replicas each, 1 ps per run):

```bash
python scripts/md/run_thiophene_scaling_mace.py
```

Output: `scratch/md_logs/mace_off/thiophene{n}mer_rep{r}.traj`

## Dipole moment computation

Single trajectory:

```bash
python run.py infer --model path/to/model --trajectory traj.npy --dpm-intor --batch-size 10
```

Parallel (SLURM):

```bash
cd paper
./submit_all_trajectories.sh 50
./check_status.sh
./combine_results.sh
```

## IR spectrum from dipole trajectories

MESA/MaxEnt-based IR spectra from ML dipole moment trajectories:

```bash
python scripts/analysis/compute_ir_spectrum.py results/dipole_parallel --output_dir ir_spectra
```
