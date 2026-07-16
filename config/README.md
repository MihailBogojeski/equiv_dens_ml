# Config File Naming

Argument files (configs) use the `@file` syntax with `run.py` and training/MD scripts.

**Outputs**: Training, MD, dipole, and other run outputs default to `scratch/`. You can delete the entire `scratch/` folder to clean up.

## Directory Layout

| Directory | Purpose | Example |
|-----------|---------|---------|
| **config/training/** | Training configs | `run.py train @config/training/h2o_small_all_001.txt` |
| **config/md/nn/** | NN-based MD configs (reference trained model via `--restart`) | `run.py md @config/md/nn/polythiophene_2mer.txt` |
| **config/md/gxtb/** | g-xtb MD configs (no model) | `run.py gxtb-md @config/md/gxtb/gxtb_polythiophene_2mer.txt` |
| **config/paper/** | Paper-specific experiments | `thiophene_poly_all_001_coreless_dpm.txt` |

## Suffixes

| Suffix | Meaning |
|--------|---------|
| `_v001`, `_002` | Variant or run index |
| `_all` | Full dataset |
| `_coreless` | Coreless density model |
| `_hm_dm` | Hamiltonian + dipole moment |
| `_dpm` | Dipole moment |
| `_2mer`, `_8mer` | Oligomer size (polythiophene) |

## MD Configs

- **config/md/nn/** – NN-based MD configs. Each file points to a trained model via `--restart`.
- **config/md/gxtb/** – g-xtb MD configs (no trained model required).
- Example: `run.py md @config/md/nn/polythiophene_2mer.txt`

### NN-based MD (trained models)

| Config | Geometry dataset | Notes |
|--------|------------------|-------|
| `nn/ethanol.txt` | datasets/ethanol_train_10.npy | **WARNING:** Uses ethanol_train_10.npy (geometry-only); ethanol_dft_train.npy is not in repo |
| `nn/ethanethiol.txt` | paper/... | |
| `nn/polythiophene_2mer.txt` | datasets/thiophene2mer_md.npy | |
| `nn/resorcinol.txt` | datasets/resorcinol_augccpvdz_test.npy | |

### g-xtb MD (no model)

Configs in `config/md/gxtb/` run MD with g-xtb as the calculator.
Use any `.npy` (positions, atom_numbers) or `.xyz` structure file.

| Config | Structure | Notes |
|--------|-----------|-------|
| `gxtb/gxtb_polythiophene_2mer.txt` | datasets/thiophene2mer_md.npy | 2-mer, ~16 atoms |
| `gxtb/gxtb_ethanol.txt` | datasets/ethanol_train_10.npy | 9 atoms |
| `gxtb/gxtb_resorcinol.txt` | datasets/resorcinol_augccpvdz_test.npy | 14 atoms |
| `gxtb/gxtb_thiophene_8mer.txt` | datasets/thiophene8mer_quiet.xyz | 8-mer, 58 atoms (C32H18S8) |
| `gxtb/gxtb_thiophene_12mer.txt` | datasets/thiophene12mer_test.xyz | 12-mer, 86 atoms (C48H26S12) |

Example: `run.py gxtb-md @config/md/gxtb/gxtb_ethanol.txt`
