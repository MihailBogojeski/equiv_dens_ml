# Configuration Reference (args.txt)

This document provides a complete reference for all configuration options used by the equivariant density and energy models. Training, inference, and molecular dynamics workflows read these options from configuration files (typically named `args.txt`) or from the command line.

## Configuration File Format

Configuration is specified using text files that are passed to the training or MD scripts with the `@` prefix. The parser treats each line as space-separated tokens; arguments are read as if they had been typed on the command line.

**Example: passing a config file**

```bash
cd equiv_dens_ml
python run.py train @config/training/h2o_small_all_001.txt
```

You can override or append options by passing additional arguments after the config file:

```bash
python run.py train @config/training/h2o_small_all_001.txt --max_steps=100 --save_dir=my_run
```

**Multi-value arguments** such as `--order` or `--order_en` accept one value per line. In a config file:

```
--order
1
3
5
```

Boolean arguments accept `True` or `False` (or `true`/`false`, `1`/`0`).

---

## Quick Reference

The following options are among the most frequently overridden when running training or experiments:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--max_steps` | int | — | Maximum number of training steps |
| `--save_dir` | str | `.` | Directory for checkpoints and logs |
| `--np_dataset` | str | — | Path to atomic structures dataset (.npy) |
| `--dens_dataset` | str | — | Path to density dataset (.npy) |
| `--learning_rate` | float | 1e-3 | Optimizer learning rate |
| `--train_batch_size` | int | 1 | Batch size for training |
| `--use_gpu` | bool | True | Use GPU for training if available |
| `--optimizer` | str | sgd | Optimizer: `adam`, `amsgrad`, or `sgd` |
| `--density_weight` | float | 1.0 | Weight of density loss |
| `--energy_weight` | float | 0.0 | Weight of energy loss |
| `--forces_weight` | float | 0.0 | Weight of forces loss |

---

## Restart and Checkpoint Loading

These options control how training resumes from a previous run or how a model is initialized from a checkpoint.

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--restart` | str | None | Path to folder from which to restart training. When set, all other arguments are ignored and loaded from the folder's `args.txt`. |
| `--load_from` | str | None | Path to a `.pth` file to initialize the model. Architecture hyperparameters from the checkpoint are used; other arguments may still apply. |
| `--no_restore` | bool | False | Do not restore optimizer state or training step from the checkpoint. |
| `--fix_arguments` | bool | False | After loading a checkpoint, do not change any arguments except hyperparameters. |
| `--fix_hyperparams` | bool | False | After loading a checkpoint, do not change any arguments at all. |
| `--args_file_name` | str | None | Base filename for saving the arguments file (e.g. `h2o_small_all_001`). Used for bookkeeping. |
| `--ignore_missing_keywords` | bool | False | Ignore missing keywords when loading the model from a checkpoint. |

---

## Neural Network Architecture Hyperparameters

These options define the equivariant neural network used for density and energy prediction.

### Activation and Angular Orders

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--activation` | str | swish | Activation function. Choices: `ssp` (shifted softplus), `swish`. |
| `--order` | int+ | [2] | Angular order(s) of the feature vectors for density prediction. One or more integers. |
| `--mixing_order` | int+ | None | Angular order(s) for interactions in the density branch. If None, derived from `order`. |
| `--order_en` | int+ | [2] | Angular order(s) of the feature vectors for energy prediction. |
| `--mixing_order_en` | int+ | None | Angular order(s) for interactions in the energy branch. If None, derived from `order_en`. |

### Feature and Basis Dimensions

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--num_features` | int | 32 | Dimensionality of the feature vectors. |
| `--num_energy_features` | int | None | Dimensionality of energy feature vectors. If None, uses `num_features`. |
| `--num_basis_functions` | int | 32 | Number of radial basis functions. |
| `--num_en_basis_functions` | int | None | Number of radial basis functions for energy prediction. If None, uses `num_basis_functions`. |
| `--num_radial_components` | int | 32 | Number of radial basis components for the density radial functions. |
| `--num_modules` | int | 3 | Number of interaction modules (iterations) in the network. |
| `--num_en_modules` | int | None | Number of modules for energy prediction. If None, uses `num_modules`. |
| `--num_neighbours` | int | 1 | Average number of neighbours (for message passing). |

### Residual Blocks

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--num_residual_pre_x` | int | 1 | Residual blocks for refining atomic features before interaction. |
| `--num_residual_post_x` | int | 1 | Residual blocks for refining atomic features after interaction. |
| `--num_residual_pre_vi` | int | 1 | Residual blocks for interaction features (channel i) before interaction. |
| `--num_residual_pre_vj` | int | 1 | Residual blocks for interaction features (channel j) before interaction. |
| `--num_residual_post_v` | int | 1 | Residual blocks for interaction features after interaction. |
| `--num_residual_output` | int | 1 | Residual blocks for refining output features. |
| `--num_energy_output` | int | 2 | Number of layers in the energy output network. |

### Basis Functions and Cutoff

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--basis_functions` | str | exp-bernstein | Radial basis type. Choices: `exp-bernstein`, `exp-gaussian`, `bernstein`, `gaussian`. |
| `--cutoff` | float | 7.94 | Cutoff radius for interactions in Bohr (default ≈ 15 Bohr). |
| `--orthonormal_basis` | bool | False | Use orthonormal basis (overlap matrix is identity). Requires compatible reference data. |

### Density and Integral Constraints

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--expansion_constraint` | str | None | Constraint type for ensuring density positivity. |
| `--integral_constraint` | str | None | Constrain density integral to number of electrons. Choices: `None`, `grid`, `coeffs`, `coeffs_in_coeffs_net`, `True`. |
| `--integral_scale` | bool | False | Scale density integral by a limited amount. |
| `--integral_min` | float | None | Minimum value for the density integral constraint. |
| `--positive_coeffs` | bool | True | Enforce non-negative order-0 coefficients. |

### Energy and Density Prediction

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--energy_model` | str | None | Use a neural network for energy prediction (e.g. `spherical`) instead of a density functional. |
| `--energy_offset` | bool | False | Use a constant offset to adjust energy levels for different functionals. |
| `--pred_radial_coeffs` | bool | True | Predict radial coefficients in addition to angular. |
| `--dummy_coeff_model` | bool | False | Optimize coefficients directly without a neural network. |
| `--compressed_extraction` | bool | False | Extract spherical harmonic coefficients in a more compressed way. |
| `--scale_sph_order` | bool | False | Rescale predicted density coefficients by spherical harmonic order. |

### Normalization and Equivariance

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--normalize` | int | 0 | Internal normalization in the density network. |
| `--normalize_en` | int | 0 | Internal normalization in the energy network. |
| `--parity_dens` | bool | False | Include parity equivariance for density prediction. |
| `--parity_en` | bool | False | Include parity equivariance for energy prediction. |

### Representation and Interaction

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--ml_width_min` | float | 0 | Minimum learned factor for initial width. |
| `--ml_width_max` | float | 2 | Maximum learned factor for initial width. |
| `--nonmixing_interaction` | bool | False | Use a non-mixing interaction as final representation layer. |
| `--nonmixing_interaction_residual` | bool | True | Whether the final nonmixing layer is residual. |
| `--density_coeffs` | bool | True | Use density coefficients as part of the representation. |
| `--append_atom_density` | bool | False | Append atomic density coefficients when using coreless densities for energy. |
| `--L0_start` | bool | True | Start energy prediction with only L0 features. |

---

## Training Hyperparameters

### Datasets

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--np_dataset` | str | — | Path to training atomic structures (.npy). |
| `--dens_dataset` | str | — | Path to training density data (.npy). Use `None` for energy-only training. |
| `--np_dataset_valid` | str | — | Path to validation atomic structures. |
| `--dens_dataset_valid` | str | — | Path to validation density data. |
| `--np_dataset_test` | str | — | Path to test atomic structures. |
| `--dens_dataset_test` | str | — | Path to test density data. |
| `--num_train` | int | — | Number of training samples. |
| `--num_valid` | int | — | Number of validation samples. |
| `--num_test` | int | None | Number of test samples. |
| `--ignore_split_indices` | bool | False | Ignore pre-defined split indices in the dataset. |

### Basis and Potentials

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--pseudo_pot_path` | str | — | Path to pseudopotential files. |
| `--orbitals_file` | str | — | Path to orbital basis (.npy). |
| `--calc_basis_file` | str | — | Path to calculation basis. |
| `--radial_coeffs_file` | str | None | Path to initial radial coefficients. |
| `--L0_coeffs_file` | str | None | Path to initial s-orbital coefficients. |
| `--atom_dens_path` | str | None | Path to free-atom densities file. |
| `--atom_dens_type` | str | spline | Type of free-atom density expansion. Choices: `spline`, `df_coeffs`, `mo_coeffs`. Use `mo_coeffs` for analytic dipole (`--dpm_intor`) and MD; `spline` is an older variant not compatible with analytic dipole. See [MODELS.md](MODELS.md). |

### Batching and Workers

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--train_batch_size` | int | 1 | Batch size for training. |
| `--valid_batch_size` | int | 1 | Batch size for validation. |
| `--test_batch_size` | int | 1 | Batch size for testing. |
| `--electron_num_batching` | bool | False | Use adaptive batching based on number of electrons. |
| `--batch_efficiency` | float | 0.7 | Target fraction of non-padded entries when using electron-based batching. |
| `--num_workers` | int | 0 | Number of worker threads for data loading. |
| `--split_seed` | int | 42 | Random seed for train/valid/test splitting. |

### Optimizer

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--optimizer` | str | sgd | Optimizer. Choices: `adam`, `amsgrad`, `sgd`. |
| `--learning_rate` | float | 1e-3 | Learning rate. |
| `--decay_factor` | float | 0.5 | Factor by which learning rate is reduced when validation does not improve. |
| `--decay_patience` | int | 10 | Number of validation intervals without improvement before decay. |
| `--stop_at_learning_rate` | float | 1e-1 | Stop training when learning rate falls below this value. |
| `--stop_at_learning_rate_patience` | float | 0 | Patience for stop-at-learning-rate criterion. |
| `--lookahead_k` | int | 5 | Lookahead steps; use -1 to disable. |
| `--epsilon` | float | 1e-8 | Epsilon for Adam/AMSGrad. |
| `--beta1` | float | 0.9 | Beta1 for Adam/AMSGrad. |
| `--beta2` | float | 0.999 | Beta2 for Adam/AMSGrad. |
| `--momentum` | float | 0.0 | Momentum for SGD. |
| `--weight_decay` | float | 0.0 | L2 regularization for weights. |
| `--en_weight_decay` | float | 0.0 | L2 regularization for energy weights. |
| `--clip_norm` | float | 0.0 | Gradient clipping norm (when gradient clipping is enabled). |

### Loss Weights

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--density_weight` | float | 1.0 | Weight of density loss. |
| `--density_grad_weight` | float | 0.0 | Weight of density gradient loss. |
| `--df_weight` | float | 0.0 | Weight of density-fitting coefficient loss. |
| `--dipole_moment_weight` | float | 0.0 | Weight of dipole moment loss. |
| `--energy_weight` | float | 0.0 | Weight of energy loss. |
| `--forces_weight` | float | 0.0 | Weight of forces loss. |
| `--energy_min_weight` | float | 0.0 | Weight of energy minimization loss. |

Each of the above has corresponding `_min`, `_decay` variants (e.g. `--density_weight_min`, `--density_weight_decay`) that control minimum weight and decay over training.

### Loss Composition

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--density_loss_comp` | str+ | [mae] | Density loss components. Choices: `mae`, `mse`, `rmse`, `lda_mae`, `lda_rmse`, `hartree_mae`, `hartree_rmse`, `coulomb`, `perc_mae`, `perc_rmse`, `mixed_dist_err`, `perc_mixed_dist_err`, `kl_loss`, `dpm_loss`, `dpm_abs_loss`. |
| `--density_loss_comp_weights` | float+ | [1.0] | Weights for each density loss component. |
| `--density_grad_loss_comp` | str+ | [norm_int] | Density gradient loss components. Choices: `norm_int`, `perc_norm_int`, `kinetic_vw`. |
| `--density_grad_loss_comp_weights` | float+ | [1.0] | Weights for density gradient loss components. |
| `--energy_loss_comp` | str+ | [mae] | Energy loss. Choices: `mae`, `rmse`. |
| `--energy_loss_comp_weights` | float+ | [1.0] | Weights for energy loss components. |
| `--forces_loss_comp` | str+ | [mae] | Forces loss. Choices: `mae`, `rmse`. |
| `--forces_loss_comp_weights` | float+ | [1.0] | Weights for forces loss components. |
| `--df_loss_comp` | str+ | [mae] | Density-fitting loss. Choices: `mae`, `rmse`. |
| `--df_loss_comp_weights` | float+ | [1.0] | Weights for density-fitting loss. |
| `--dipole_moment_loss_comp` | str+ | [mae] | Dipole moment loss. Choices: `mae`, `rmse`. |
| `--dipole_moment_loss_comp_weights` | float+ | [1.0] | Weights for dipole moment loss. |

### Stability and Normalization

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--max_energy_error` | float | 0.1 | Clamp energy MAE above this value for early-training stability. |
| `--max_forces_error` | float | 0.1 | Clamp forces MAE above this value for early-training stability. |
| `--center_energy` | bool | True | Subtract mean from energy targets. |
| `--atomic_energies` | str | None | File with atomic energies for total-energy normalization. |
| `--df_loss_weights` | bool | False | Weight density-fitting loss by radial coefficients. |

### Parameter Averaging and GPU

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--use_parameter_averaging` | bool | True | Keep exponential moving average of parameters. |
| `--ema_decay` | float | 0.999 | Decay rate for parameter EMA. |
| `--ema_start_epoch` | int | 0 | Step at which EMA begins. |
| `--use_gpu` | bool | True | Use GPU for training if available. |
| `--multiple_gpus` | bool | False | Use multiple GPUs. |

### Density Grid

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--density_subsamples` | int | 10000 | Number of grid points for density evaluation. |
| `--density_grad` | bool | False | Include density gradient in the loss. |
| `--pyscf_grid` | bool | True | Use PySCF for grid generation. |
| `--cube_grid` | bool | False | Use cubical grid for training. |
| `--cube_grid_valid` | bool | False | Use cubical grid for validation. |
| `--cube_size` | int | 50 | Linear size of cubical grid. |
| `--cube_extent` | float | 4.1483 | Extent of cubical grid. |
| `--cube_origin` | float | -2.0318 | Origin of cubical grid. |
| `--spherical_grid_level` | int | 2 | Level of spherical grid. |
| `--coord_weights` | bool | True | Weight grid points by density. |
| `--weights_balance` | float | 1.0 | Balance factor for coordinate weights. |
| `--rotate_grid` | bool | True | Rotate density grid during training. |
| `--cutoff_dens_coords` | bool | False | Cut off grid coordinates per atom by grid extent. |

### Grid Scaling and Density Labels

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--softmax_norm` | bool | True | Normalize coefficients with softmax. |
| `--percentage_error` | bool | True | Report density error as percentage of integral. |
| `--grid_scaling_factor` | bool | False | Scale density to correct integral on grid. |
| `--grid_scaling_annealing` | float | 1.0 | Annealing factor for grid scaling. |
| `--grid_scaling_start` | int | 10000 | Step at which grid scaling starts. |
| `--projected_density` | bool | False | Use density-fitting basis for labels. |

### Units

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--energy_unit_in` | str | kcal/mol | Energy unit of input data. Choices: `kcal/mol`, `hartree`, `eV`, `kelvin`, `millihartree`. |
| `--energy_unit_out` | str | kcal/mol | Energy unit for output. Same choices. |
| `--distance_unit_in` | str | angstrom | Distance unit of input. Choices: `angstrom`, `bohr`. |
| `--distance_unit_out` | str | angstrom | Distance unit for output. Same choices. |
| `--output_scaling` | bool | False | Scale output forces to unit variance. |

### Density Fitting and Fine-Tuning

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--fast_df` | bool | True | Use shorter density-fitting procedure. |
| `--core_density_basis` | float | 0.0 | Fraction of s orbitals for core density basis; if > 0, perform fitting. |
| `--density_fine_tuning` | bool | False | Fine-tune density with a different loss. |
| `--density_loss_comp_ft` | str+ | [mae] | Density loss for fine-tuning. Same choices as `density_loss_comp`. |
| `--density_loss_comp_ft_weights` | float+ | [1.0] | Weights for fine-tuning density loss. |
| `--fine_tuning_lr_factor` | float | 0.1 | Learning rate reduction for fine-tuning. |
| `--fine_tuning_stop_lr_factor` | float | 0.1 | Stop-at-LR factor for fine-tuning. |

### Coreless and Free-Atom Density

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--remove_atom_density` | bool | False | Subtract free-atom density from total density (SAD baseline / delta-learning). When True, the model predicts the delta density; use `--append_atom_density=True` at inference to recover total density. See [MODELS.md](MODELS.md). |
| `--density_from_df` | bool | False | Use density-fitting coefficients instead of predicting. |
| `--density_from_free_atoms` | bool | False | Use free-atom coefficients instead of predicting. |

### Logging and Experiment Tracking

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--wandb_mode` | str | offline | Weights & Biases mode. Choices: `online`, `offline`, `disabled`. |
| `--verbose` | int | 0 | Verbosity level. |
| `--timing` | bool | False | Print timing statistics. |
| `--memory` | bool | False | Print memory statistics. |
| `--compile` | bool | False | Compile model for higher efficiency (PyTorch 2.0+). |

---

## Simulation Hyperparameters (Molecular Dynamics)

These options apply when running ML-enhanced molecular dynamics via the `md` subcommand.

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--temperature` | int | 300 | Temperature in Kelvin. |
| `--new_run` | bool | True | Start a new simulation; if False, continue a previous one. |
| `--log_dir` | str | `.` | Directory for simulation logs. |
| `--log_suffix` | str | `` | Suffix for log filenames. |
| `--md_steps` | int | 100 | Number of MD steps. |
| `--langevin` | bool | True | Use Langevin dynamics; if False, use velocity Verlet. |
| `--warm_up` | bool | True | Use Langevin for first 5% of steps, then switch to Verlet. |
| `--simulation_type` | str | md | Type of simulation. Choices: `md`, `opt`. |
| `--port_num` | int | 50007 | Port for client-server communication. |
| `--force_conversion` | str | kcal/mol/Ang | Unit for forces. |
| `--position_conversion` | str | Ang | Unit for positions. |
| `--energy_conversion` | str | kcal/mol | Unit for energy. |
| `--start_idx` | int+ | [-1] | Start indices for simulation; -1 means auto. |

---

## Logging and Checkpoints

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--save_dir` | str | `.` | Directory for checkpoints and logs. |
| `--write_parameter_summaries` | bool | False | Write parameter summaries. |
| `--validation_interval` | int | 1 | Validate every N steps. |
| `--summary_interval` | int | 1 | Log summaries every N steps. |
| `--checkpoint_interval` | int | 1 | Save checkpoints every N steps. |
| `--keep_checkpoints` | int | 0 | Keep N older checkpoints; -1 keeps all. |

---

## Miscellaneous

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| `--dtype` | str | torch.float32 | Floating-point type. Choices: `torch.float16`, `torch.float32`, `torch.float64`. |
| `--legacy` | bool | False | Use legacy density network implementation. |
| `--test_save` | bool | False | Save test output to a file. |
| `--test_save_name` | str | test_save_results.pt | Filename for saved test output. |
| `--test_eval_all` | bool | False | Evaluate errors on training and validation data during testing. |
| `--no_compare` | bool | False | Skip accuracy comparison; only compute predictions. |
| `--dpm_intor` | bool | False | Compute dipole moments using analytic integrals. |

---

## See Also

- [run.py](../run.py) — Unified entry point for training, inference, MD, and dipole workflows.
- [config/training/](../config/training/) — Example configuration files for training and MD.
