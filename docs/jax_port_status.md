# JAX port status (`equiv_dens_jax`)

`equiv_dens_jax` (CLI: `run_jax.py`) was written against the original PyTorch DenSNet, before the
speedup work (#24) changed the Torch core. It was deliberately left unchanged in the merge
(see `docs/merge_densnet_performance.md`, "Decisions"), so Torch and JAX now use different internal
formats. The JAX tests pass only because the Torch side is switched back to the format JAX expects.
The goal is to update the JAX model to work directly with the current default Torch architecture
and drop these compatibility paths.

## Format mismatch

| | Torch core (current default) | `equiv_dens_jax` |
|---|---|---|
| Spherical features | one packed tensor `[atoms, (L+1)², F]` | per-L lists of `[..., 2L+1, F]` |
| Density coefficients | `PackedLayout` order, built per batch as index tensors | L-based dicts (`sph_dict` / `L_dict`, `(Z, L)` keys) and the legacy per-atom layout |
| CG products | fused Triton kernel; weights per path from the nonzero CG triples | dense `cg_matrix` / `cg_masked` from `ClebschGordanMatrix` |
| Free-atom (SAD) channels in the energy head | `atom_df_*_table` buffers, per-atom element lookup | per-element dicts `atom_df_coeffs` / `atom_df_widths` / `atom_df_scales` |

## Compatibility paths kept for JAX

These exist only so that the exporter and the JAX tests keep working. The Torch forward does not use them.

- **Exporter attributes** (commit "Keep per-element tables and clebsch_gordan attributes that the JAX exporter reads"):
  - `SelfMixing`, `PairMixing` and `PairInteraction` (`nn/modules/spherical_harmonic_layers.py`) hold `self.clebsch_gordan` again, so `equiv_dens_jax.convert.export_module` can export `cg_matrix` / `cg_masked`.
  - `SphericalHarmonicsEnergyNetwork` (`nn/property_output/energy.py`) builds the per-element dicts `atom_df_coeffs` / `atom_df_widths` / `atom_df_scales` next to the table buffers the Torch path uses.
- **Reference generators** (`tests/jax_pipeline/generate_layer_fixtures.py`, `generate_energy_reference.py`, `generate_density_reference.py`): pack the per-L inputs for the Torch layers (`to_packed`) and split the outputs back to per-L (`to_per_L`, `save_case(..., packed=True)`), so the saved references keep the format the JAX tests read.
- **Fixture models** (`benchmarks/fixtures.py`, `build_model`): `density_backend="legacy"`, `energy_backend="legacy"`, `normalize=0`, `L0_start=False`. Most energy references also use `L0_start=False` (the Torch default is `True`; only `spherical_L0` covers it).
- **Initial radial coefficients:** `PackedLayout` rounds `width0` / `scale0` to float32, as the original models stored them. JAX keeps float64.
- **Data:** legacy pickled `.npy` datasets load only with `--trusted-legacy` / `trusted_legacy=True`.

## Test status

Measured on `merge-densnet-performance` 232ce1f (CPU, references regenerated from the current Torch code):
**199 passed, 3 failed** (`tests/jax_pipeline`). Directly after the merge, before the generators were adapted: 105 passed, 20 failed, 48 errors.

Remaining failures:
- `test_density.py::test_torch_density_head_reference_bundles`: Torch rounds the initial radial coefficients to float32 and JAX keeps float64. The difference is ~3e-8 in `radial_width` / `radial_scale`, amplified to 3.6e-6 in the integral-constraint variant.
- `test_parameter_usage.py::test_complete_pipeline_matches_all_torch_none_gradients` and
  `test_pipeline.py::test_weight_decay_skips_torch_none_gradients_and_keeps_optimizer_state`:
  the tests expect Torch to return `None` gradients for structurally unused parameters (156 in the pipeline fixture). The packed Torch core returns zero tensors instead. The gradient values agree.

Environment caveats (from a run in the Hydra sandbox, where the suite gave 60 passed, 30 failed, 76 errors, identically before and after rebasing `dev`):
- `tests/performance_contracts` uses the b15a0ff paper reference, created automatically under `scratch/equiv_dens_ml-paper-reference` (or set `DENSNET_REFERENCE_ROOT`). The JAX tests read reference bundles under `scratch/jax_pipeline/...` (env vars `DENSNET_JAX_*_REFERENCE`) that must be generated in the Torch runtime.
- The JAX suite needs the JAX image (`containers/jax-pipeline.def`, run via `tools/runtime/run.sh jax`). The Hydra sandbox image is Python 3.10 and `jax==0.9.2` needs Python >= 3.11.
- Some tests need a GPU or reference data that isn't in the repo.
- `tests/jax_pipeline/test_pipeline.py` and `tests/performance_contracts/test_pipeline.py` share a basename, so `pytest tests` stops at collection. Run the two directories separately. CI (`.github/workflows/tests.yml`) runs only `tests/test_*.py` and does not run the JAX suite.

## Porting to the current architecture

1. Switch `equiv_dens_jax` to the packed `(L+1)²` feature layout and the `PackedLayout` coefficient order, so features and coefficients map one to one onto the Torch core.
2. Build the CG products from the per-path nonzero triples the Torch core uses, instead of the exported dense CG matrices.
3. Read the SAD channels from the `atom_df_*_table` buffers instead of the per-element dicts.
4. Regenerate the references from the default Torch architecture (`L0_start=True`, packed inputs and outputs), remove `to_per_L` / `packed=True` from the generators and the legacy fixture backends.
5. Remove the exporter-only attributes from the Torch modules.
6. Decide the float32 rounding of `width0` / `scale0` on one side, and update the gradient tests to treat zero gradients of unused parameters like `None`.
7. Fix the test environment (reference-checkout path, duplicate `test_pipeline.py` basename) so `tests/jax_pipeline` runs in a plain checkout, then add it to CI.
