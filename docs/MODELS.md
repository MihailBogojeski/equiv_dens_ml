# Model Configurations and Key Parameters

This document describes the key training and inference parameters used by the paper models, and explains crucial differences between them. For a complete list of all configuration options, see [args_reference.md](args_reference.md).

## Key Parameters Explained

### max_steps and "600k steps"

- **max_steps**: Maximum number of training steps; training stops when this value is reached.
- **600k steps**: The polythiophene model (`thiophene_poly_all_001_1_coreless`) was trained for 600,000 steps. Other paper models (ethanol, ethanethiol, resorcinol) use 300,000 steps.
- **Why it matters**: Longer training can improve convergence for larger or denser systems; polythiophene benefits from the extra steps.

### atom_dens_type (mo_coeffs vs spline vs df_coeffs)

- **mo_coeffs**: Free-atom density expanded in molecular-orbital coefficients. Required for analytic dipole integrals (`--dpm_intor`); used by all paper models.
- **spline**: Spline representation. Older variant; some archived configs use it; not compatible with analytic dipole.
- **df_coeffs**: Density-fitting coefficients.
- **Rule of thumb**: Use `mo_coeffs` for inference, dipole computation, and MD; use `spline` only for legacy or non-dipole workflows.

### SAD baseline (remove_atom_density + append_atom_density)

- **remove_atom_density=True**: The model predicts the *delta* density (total minus free-atom density). This is the SAD (superposed atomic densities) baseline / delta-learning setup.
- **append_atom_density=True**: At inference, the predicted delta is added back to the free-atom density to obtain the total density. Required when `remove_atom_density=True`.
- **linear_out**: When `remove_atom_density=True`, the density output head uses a linear (no softmax) output.

## Crucial Differences Between Paper Models

| Parameter | Polythiophene | Ethanol | Ethanethiol | Resorcinol |
|-----------|---------------|---------|-------------|------------|
| **max_steps** | 600,000 | 300,000 | 300,000 | 300,000 |
| **atom_dens_type** | mo_coeffs | mo_coeffs | mo_coeffs | mo_coeffs |
| **remove_atom_density** | True | True | True | True |
| **cutoff** | 5.0 Å | default (~15 Bohr) | default | default |
| **num_en_basis_functions** | 128 | 128 | 32 | 32 |
| **density_subsamples** | 30,000 | 10,000 | 10,000 | 10,000 |
| **density_loss_comp** | mae | perc_mae | perc_mae | perc_mae |
| **electron_num_batching** | True | — | — | — |
| **train_batch_size** | 1550 | 15 | 10 | 10 |
| **normalize_en** | 0 | 2 | 0 | 2 |

### Implications

- **cutoff=5.0** (polythiophene): Shorter interaction radius; important for MD/inference consistency. Other models rely on the default (~15 Bohr).
- **num_en_basis_functions**: 128 for polythiophene/ethanol vs 32 for ethanethiol/resorcinol — affects energy-branch capacity.
- **density_loss_comp**: `mae` vs `perc_mae` — different density loss formulations.
- **electron_num_batching**: Polythiophene uses adaptive batching by electron count; others use fixed batch sizes.
- **normalize_en**: 0 vs 2 — different energy normalization schemes.

## When Each Choice Matters

- **Inference / dipole**: Use `atom_dens_type=mo_coeffs` and `remove_atom_density=True` with `append_atom_density=True` to match paper models. Use `--dpm_intor` for analytic dipole integrals.
- **MD**: Same as inference; ensure the model's `args.txt` has `cutoff` and `atom_dens_type` consistent with training.
- **Training new models**: Match the molecule's args template in `paper/models/args/` for reproducibility.

## Density Dataset

MD and inference run **without** the density dataset. The model predicts energy and forces; the density file is only needed for training and for DFT-based evaluation (testing, error metrics).
