# Paper Reproduction Setup

This document describes the current status of molecule support for inference and MD, missing datasets/configs, and how to generate required data.

## Optional: GPU-accelerated DFT (gpu4pyscf)

For faster DFT calculations (dataset generation, AIMD), install gpu4pyscf. Choose the extra matching your CUDA version:

```bash
# CUDA 11.x
pip install equiv-dens[gpu-cuda11]

# CUDA 12.x
pip install equiv-dens[gpu-cuda12]

# CUDA 13.x
pip install equiv-dens[gpu-cuda13]
```

Check your CUDA version with `nvcc --version`. The gpu4pyscf packages require compute capability 7.0+ (Volta and later).

### Polythiophene dataset generation

From a trajectory (XYZ, NPY, or per-frame NPY), generate training datasets:

```bash
python scripts/data/generate_polythiophene_dataset.py --trajectory path/to/traj.xyz --output-prefix polythiophene --df
```

Output: `{prefix}_pyscf_augccpvdz_d4.npy` (density) and `{prefix}_npy.npy` (structures). Use `--df` for density-fitting coeffs (training). Use `--no-gpu` to force CPU.

### Ab initio MD with gpu4pyscf

Run AIMD with PBE/aug-cc-pVDZ + D4:

```bash
python scripts/md/aimd_gpu4pyscf.py --structure path/to/init.xyz --output aimd.traj --steps 1000 --temperature 300 --ensemble nvt
```

**Density dataset:** Not required for MD or inference. The model predicts energy and forces; the density dataset is only needed for training and for DFT-based evaluation (testing, error metrics).

## Upgrade Notes (PyTorch 2.6+ / SchNetPack 2.1.1)

The codebase has been upgraded for compatibility with:

- **PyTorch >= 2.6**: Checkpoint loading uses `weights_only=False` for trusted sources.
- **SchNetPack >= 2.1.1**: A runtime patch in `equiv_dens.compat` fixes the `T_co` import issue with PyTorch 2.5+.
- **NumPy >= 2.0**: Relaxed upper bounds for modern environments.

Verification commands after setup:

```bash
python run.py md @config/md/nn/ethanethiol.txt --md_steps=10
python run.py infer --model paper/models/polythiophene/2024-03-23_1XDL67zp_ext --trajectory datasets/thiophene2mer_md.npy --dpm-intor
```

## Current Status

| Molecule | Inference | MD | Model ID |
|----------|-----------|-----|----------|
| **Ethanethiol** | Yes | Yes | 2024-02-29_NUmID4hT_ext4 |
| **Polythiophene** | Yes | Yes (2mer) | 2024-03-23_1XDL67zp_ext |
| **Ethanol** | Yes | Yes | 2024-03-22_96w7KyGG |
| **Resorcinol** | Yes | Yes | 2024-03-18_Ozf6CkNF_ext1 |

## One-Command Examples

After `pip install -e .` and `git lfs pull`:

```bash
# Ethanethiol MD
python run.py md @config/md/nn/ethanethiol.txt

# Polythiophene (2mer) MD
python run.py md @config/md/nn/polythiophene_2mer.txt

# Resorcinol MD
python run.py md @config/md/nn/resorcinol.txt

# Ethanol MD (WARNING: uses datasets/ethanol_train_10.npy; geometry-only, no ethanol_dft_train.npy)
python run.py md @config/md/nn/ethanol.txt

# Inference on any trajectory (with dipole)
python run.py infer --model paper/models/polythiophene/2024-03-23_1XDL67zp_ext --trajectory datasets/thiophene2mer_md.npy --dpm-intor --batch-size 10
```

## Missing Datasets

### Resorcinol

**Available:** `resorcinol_combo_kmeansidx-1000_train.npy`, `resorcinol_augccpvdz_test.npy` (geometry).

**Blocking:** None for MD; geometry datasets exist. The model `paper/models/resorcinol/2024-03-18_Ozf6CkNF_ext1` is available. Run `python run.py md @config/md/nn/resorcinol.txt` once models are extracted.

## Missing Configs

- **Polythiophene 8mer/10mer/12mer MD:** No configs yet. Create by copying `config/md/nn/polythiophene_2mer.txt` and setting `--np_dataset_test` to oligomer-specific structure files (e.g. from `paper/trajectories/polythiophene_full_md/` or built via `notebooks/thiophene_append_clean.ipynb`).

## Path Requirements

All canonical configs use repo-relative paths (e.g. `datasets/`, `paper/models/`). Run from the `equiv_dens_ml` directory. The scripts apply path replacement for legacy `/home/ml-dft/equiv_dens/` when present.

## Initial Model Extraction

If models are not yet in the repo (e.g. after fresh clone without LFS pull), run:

```bash
./scripts/setup_paper_models.sh
```

Place zip archives in `paper/archives/` or set `ARCHIVES_DIR` to their location (e.g. `ARCHIVES_DIR=to_organize` if archives are in `to_organize/`).
