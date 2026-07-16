# equiv_dens_ml

SE3-equivariant neural network for predicting molecular electron densities. The model learns density coefficients in an atom-centered basis and supports training on densities, energies, and forces.

Any publication that uses this code or its outputs should cite the associated manuscript (see [Citing this work](#citing-this-work)).

## Requirements

- Python >= 3.10
- PyTorch >= 2.6
- SchNetPack >= 2.1.1 (for MD)
- NumPy >= 2.0

A compatibility patch for SchNetPack with PyTorch 2.5+ is applied automatically on import.

## Installation

```bash
conda create -n equiv_dens_ml python=3.10 pip
conda activate equiv_dens_ml
pip install -e . -r requirements.txt
```

**Optional GPU-accelerated DFT:** For dataset generation and AIMD scripts, install gpu4pyscf: `pip install equiv-dens[gpu-cuda12]` (or `gpu-cuda11` / `gpu-cuda13` for other CUDA versions). See [docs/SETUP.md](docs/SETUP.md).

**Note:** Notebooks may require `weights_only=False` in `torch.load()` when using PyTorch 2.6+; the main package scripts have been updated.

## Usage

### Quick reference

| Task | Command |
|------|---------|
| Train (density only) | `python run.py train --mode density @config/training/h2o_small_all_001.txt` |
| Train (joint) | `python run.py train --mode joint @config/...` |
| Train (multi-phase) | `python run.py train --mode multiphase @config/...` |
| Train (energy only) | `python run.py train --mode energy @config/...` |
| Evaluate on dataset | `python run.py test --checkpoint path/to/model` |
| Inference on trajectory | `python run.py infer --model path/to/model --trajectory traj.npy` |
| Run MD | `python run.py md @config/md/nn/ethanethiol.txt` or `@config/md/nn/polythiophene_2mer.txt` |
| g-xtb MD | `python run.py gxtb-md @config/md/gxtb/gxtb_polythiophene_2mer.txt` |
| Dipole (parallel) | `python run.py dipole --trajectory X --model Y` |
| AIMD (DFT) | `python scripts/md/aimd_gpu4pyscf.py --structure X --steps N` |
| MACE-OFF MD | `python scripts/md/mace_off_md_run.py --structure X --steps N` (requires `pip install mace-torch`) |

### Training and testing

Configurations use argument files in `config/training/` (or `args/` for backward compatibility).

Unified entry point:

```bash
python run.py train @config/training/h2o_small_all_001.txt
```

**Quick tests** (outputs go to `scratch/test_runs`): add `@config/training/test_overrides.txt` or pass `--save_dir=scratch/test_runs --keep_checkpoints=0 --max_steps=20`.

```bash
python run.py train @config/training/h2o_small_all_001.txt @config/training/test_overrides.txt
python run.py test --checkpoint path/to/model
python run.py dipole --trajectory path/to/traj.npy --model path/to/model
```

Scripts are under `scripts/`:

- **scripts/training/train_all.py** – Multi-phase training: density first, then energy/forces with frozen density head
- **scripts/training/train.py** – Joint training of all properties
- **tests/test_eval.py** – Evaluate a trained model on a dataset (via `run.py test`)

Example:

```bash
python run.py train --mode multiphase @config/training/h2o_small_all_001.txt
```

### Dipole moment computation

Single trajectory (via run.py):

```bash
python run.py infer --model path/to/model --trajectory traj.npy --dpm-intor --batch-size 10
```

Or directly:

```bash
python scripts/training/eval_model_npy.py path/to/model/args.txt traj.npy --dpm_intor --batch_size=10
```

Use `--dpm_intor` for integral-based dipole calculation. Results are written to `scratch/dipole/` by default.

### Parallel dipole computation (SLURM)

For long trajectories, use the parallel workflow:

```bash
cd paper
./submit_all_trajectories.sh 50
./check_status.sh
./combine_results.sh
```

The first argument (50) is frames per job. Partial results are written to `results/dipole_parallel/` and merged by `combine_results.sh`.

**SLURM limits (NYU Greene):** Max array size 10,000 tasks; default GPU QOS 48 hours. Use `frames_per_job >= total_frames / 10000` to stay within limits. Limit concurrent jobs with `%N` (e.g. `%50`).

## Documentation

- [docs/quickstart.md](docs/quickstart.md) – Get running in a few minutes
- [docs/SETUP.md](docs/SETUP.md) – Full setup, GPU-accelerated DFT, missing datasets
- [docs/training.md](docs/training.md) – Training workflows and configs
- [docs/md.md](docs/md.md) – MD (NN, g-xtb, AIMD), dipole, IR spectra
- [docs/paper_reproduction.md](docs/paper_reproduction.md) – Paper models and one-command runs
- [docs/MODELS.md](docs/MODELS.md) – Model parameters and comparison

## Directory structure

```
equiv_dens_ml/
├── config/training/     Argument files for training
├── config/md/            MD configs per molecule (ethanethiol, polythiophene, etc.)
├── src/equiv_dens/      Model and training code
├── scripts/             Analysis and HPC scripts
├── paper/
│   ├── models/          Trained models (polythiophene, ethanol, ethanethiol, resorcinol, qm7x)
│   ├── trajectories/    MD trajectories (.npy, .xyz)
│   ├── results/         Output (dipole_parallel/, dipole_combined/)
│   └── archives/        Original model/trajectory archives
├── datasets/            Training data
├── notebooks/           Tutorials, experiments, paper notebooks
└── docs/                User guide (user_guide.tex)
```

## Additional features

### g-xtb MD

The repository supports MD with g-xtb as calculator (no trained model required), reusing SchNetPack infrastructure. Requires the g-xtb binary and `$GXTBHOME` (parameter directory). See [g-xtb/README.md](g-xtb/README.md).

```bash
python run.py gxtb-md @config/md/gxtb/gxtb_polythiophene_2mer.txt
```

Configs: `config/md/gxtb/*.txt`. See [config/README.md](config/README.md).

### Polythiophene generator

The polythiophene generator builds n-mer oligomers from SMILES (2–5 alpha linkage), embeds 3D coordinates with RDKit, and optionally optimizes with g-xtb.

```bash
python scripts/data/polythiophene_generator.py --help
```

### AIMD (ab initio MD)

Ab initio MD with PySCF/gpu4pyscf uses PBE/aug-cc-pVDZ and D4 dispersion. GPU acceleration is used when gpu4pyscf is installed. Suitable for reference trajectories or dataset generation without a trained model.

```bash
python scripts/md/aimd_gpu4pyscf.py --structure path/to/struct.xyz --steps 100
```

### MACE-OFF MD (off-the-shelf organic MLIP)

MACE-OFF23 is a transferable organic force field (JACS 2024) covering H, C, N, O, P, S, F, Cl, Br, I — suitable for polythiophene and related molecules. Requires `pip install mace-torch`. See [docs/md.md](docs/md.md).

```bash
python scripts/md/mace_off_md_run.py --structure datasets/thiophene2mer_init.npy --output mace_off.traj --steps 2000
python scripts/md/run_thiophene_scaling_mace.py  # n=1–6, 4 replicas, scaling benchmark
```

### Polythiophene dataset generation

The dataset generator runs PBE/aug-cc-pVDZ DFT with D4 on each frame of a trajectory and produces the format expected by AtomsDensityData. Accepts XYZ, single NPY, or per-frame NPY patterns.

```bash
python scripts/data/generate_polythiophene_dataset.py --help
```

### QM7x molecular dataset

QM7x is a molecular dataset for density and energy training. The workflow is two-phase: (1) density-only training until converged, then (2) energy-only training with the density head frozen.

**Setup:** Extract the archive and organize model and datasets:

```bash
./scripts/setup_qm7x.sh
```

The script looks for `qm7x250_model.zip` in `paper/archives/` or the parent directory. It extracts the converged density model to `paper/models/qm7x/2024-04-22_bjOUNzrR/` and base structures to `datasets/`.

**Prerequisites:** Energy training requires density/DFT result files (with energy and forces labels) and auxiliary basis files in `datasets/`:

- `qm7x_train_dft_augccpvdz.npy`, `qm7x_valid_dft_augccpvdz.npy`, `qm7x_test_dft_augccpvdz.npy`
- `augccpvqzjkfit_orbital_basis_libcint_df.npy`
- `augccpvqzjkfit_radial_coeffs_libcint_df.npy`
- `free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimzed.npy` (note: original typo preserved)

If these are not in the zip, they must be provided or generated separately.

**Training workflow:**

- **Phase 1 (density):** Already done. Model checkpoint: `paper/models/qm7x/2024-04-22_bjOUNzrR/best_bjOUNzrR.pth`
- **Phase 2 (energy):** Run energy-only training, loading the frozen density model:

```bash
python run.py train --mode energy @config/training/qm7x_energy_001.txt
```

**Optional overrides:** `--save_dir`, `--max_steps`, `--load_from`, etc. Example dry run:

```bash
python run.py train --mode energy @config/training/qm7x_energy_001.txt --max_steps=10 --save_dir=scratch/qm7x_test
```

### IR spectrum from dipole trajectories

IR spectra can be computed from ML dipole moment trajectories using MESA/MaxEnt.

```bash
python scripts/analysis/compute_ir_spectrum.py results/dipole_parallel --output_dir ir_spectra
```

## Paper reproduction

One-command MD and inference for each molecule (after `git lfs pull`):

| Molecule | MD | Inference |
|----------|-----|-----------|
| Ethanethiol | `python run.py md @config/md/nn/ethanethiol.txt` | `python run.py infer --model paper/models/ethanethiol/2024-02-29_NUmID4hT_ext4 --trajectory path/to/traj.npy --dpm-intor` |
| Polythiophene | `python run.py md @config/md/nn/polythiophene_2mer.txt` | `python run.py infer --model paper/models/polythiophene/2024-03-23_1XDL67zp_ext --trajectory datasets/thiophene2mer_md.npy --dpm-intor` |
| Ethanol | Blocked (missing datasets) | `python run.py infer --model paper/models/ethanol/2024-03-22_96w7KyGG --trajectory path/to/traj.npy --dpm-intor` |
| Resorcinol | Blocked (missing density files) | `python run.py infer --model paper/models/resorcinol/2024-03-18_Ozf6CkNF_ext1 --trajectory path/to/traj.npy --dpm-intor` |

See [docs/SETUP.md](docs/SETUP.md) for missing datasets, configs, and how to generate required data.

## Models and data (paper)

| Molecule | Model ID | Path |
|----------|----------|------|
| Polythiophene | 2024-03-23_1XDL67zp_ext | `paper/models/polythiophene/2024-03-23_1XDL67zp_ext` |
| Ethanethiol | 2024-02-29_NUmID4hT_ext4 | `paper/models/ethanethiol/2024-02-29_NUmID4hT_ext4` |
| Resorcinol | 2024-03-18_Ozf6CkNF_ext1 | `paper/models/resorcinol/2024-03-18_Ozf6CkNF_ext1` |
| Ethanol | 2024-03-22_96w7KyGG | `paper/models/ethanol/2024-03-22_96w7KyGG` |

**Polythiophene:** Delta-learning with SAD baseline, order [1,3,5], cutoff 5.0 Å. Checkpoint: `best_ext.pth`. The model `2024-03-23_1XDL67zp_ext` is **MD-capable** (joint density+energy training) — use for both inference/dipole recomputation and MD. Extracted from `thiophene_new_model.zip`.

**Args files:** `models/args/*.txt` are templates. `models/<molecule>/<id>/args.txt` records the configuration used to train that model.

**Key model parameters:** All paper models use `atom_dens_type=mo_coeffs`, `remove_atom_density=True` (SAD baseline), and `append_atom_density=True`. Polythiophene was trained for 600k steps; others for 300k. See [docs/MODELS.md](docs/MODELS.md) for full parameter explanations and model comparison.

**Trajectories:** `polythiophene_1ps_mid/` (short XYZ snapshots); `polythiophene_full_md/` (full .npy); `resorcinol/`, `ethanethiol/`, `ethanol/` for small molecules.

## Git LFS and large datasets

Large files (`*.npy`, `*.xyz`, `*.pth`, `*.pt`, `*.zip`) are stored with Git LFS. After cloning, run `git lfs pull` if files appear as pointers. See `.gitattributes` for patterns.

Some large trajectories (~9.7 GB) were moved off LFS to external hosting (Zenodo/Figshare). See [DATA.md](DATA.md) and run `./scripts/download_data.sh` to fetch them when needed.

If models are missing after clone, run `./scripts/setup_paper_models.sh` (requires zip archives in `paper/archives/` or set `ARCHIVES_DIR`). For QM7x, run `./scripts/setup_qm7x.sh` (zip may be in parent directory or `paper/archives/`).

## Citing this work

```bibtex
@article{manuscript_in_prep,
  title={Equivariant Neural Networks for Molecular Density Prediction},
  author={...},
  journal={In preparation},
  year={2025}
}
```

## Contributors

Mihail Bogojeski is the primary contributor and built the original code. Muhammad R. Hasyim refactored the code and added the polythiophene generator, PySCF AIMD, IR spectrum, and g-xtb integration. Leslie Vogt-Maranto provided the initial versions of the polythiophene generator scripts.

## Contact

For model access or missing files: Mihail Bogojeski. For code issues: open a GitHub issue.
