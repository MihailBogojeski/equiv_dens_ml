# Merge of `optimization` and `densnet-performance`

## Branches

- `optimization` (061a873): speedups of the existing DenSNet architecture. Outputs match the original model to ~1e-13 on the reference configurations, and old checkpoints load.
- `merge-densnet-performance` (pushed): `optimization` merged with `densnet-performance` (abe41ea, its latest commit), with the fixed-topology compiled route removed, plus the follow-up work listed under "Changes after the merge". All further work lands on this branch.

## Changes in `optimization`

**Model core**
- Spherical features are one packed tensor `[atoms, (L+1)², F]` instead of a list per L.
- The channel-wise CG tensor products (SelfMixing / PairMixing) use one fused Triton kernel for forward, backward and double backward. The math is the same, but no per-triple intermediates are stored.
- Dense CG matrices are no longer stored per layer (checkpoints 3 GB → 33 MB).

**Density head**
- `PackedLayout` gathers the coefficients of all atoms at once.
- The grid expansion is fused and chunked, and the density gradient comes from autodiff.

**Compiled route**
- `--compile` / `DFTNetwork.enable_compile()` compiles with dynamic shapes and computes forces through `torch.func`.
- Compiled and eager gradients agree to 3e-14.

**PySCF and data**
- gpu4pyscf data loading: 127 → 16 ms per batch.
- Closed-form dipole integrals: 0.8 s → 0.009 s.
- Dataset generation: (H2O)8 341 → 56 s.
- Exact s-Gaussian free-atom density fits, sampled in torch (per-element counts since the merge, see below).

**Speed vs the original code** (`benchmarks/bench_model.py`, float32)

| Case | Original (eager) | Compiled | Speedup |
|---|---|---|---|
| h2o_small training step | 392 ms | 109 ms | 3.6x |
| water32 training step | 1724 ms | 324 ms | 5.3x |
| Forces inference, h2o_small | 160 ms | 24 ms | 6.7x |

- Peak memory at water32 drops from 6.5 to 3.6 GB.
- Eager-only speedups are 1.2–1.3x on small molecules and 3.5x at water32.

## Benchmark of the compiled routes

`torch_compile_speed.py` from `densnet-performance`, ethanol, float64, RTX 4070 Laptop (8 GB). Each cell is median ms per call, then peak GPU memory.

- **fixed-topology compile**: `FunctionalDenSNet` from `densnet-performance`
- **dynamic compile**: `DFTNetwork.enable_compile()` from `optimization`

| Mode | b1: fixed-topology compile | b1: dynamic compile | b1: JAX | b8: fixed-topology compile | b8: dynamic compile | b8: JAX |
|---|---|---|---|---|---|---|
| Energy + forces | 33 ms, 286 MB | 38 ms, 285 MB | 39 ms, 846 MB | 132 ms, 557 MB | 121 ms, 641 MB | 149 ms, 1.87 GB |
| Calculator (MD step) | 34 ms, 286 MB | 38 ms, 285 MB | 41 ms, 846 MB | 135 ms, 562 MB | 129 ms, 643 MB | 150 ms, 1.87 GB |
| Density inference | 91 ms, 342 MB | 18 ms, 106 MB | 38 ms, 335 MB | 1160 ms, 970 MB | 80 ms, 668 MB | 251 ms, 1.87 GB |
| Density training step | 140 ms, 611 MB | 53 ms, 303 MB | 144 ms, 603 MB | 6192 ms, 3.65 GB | 384 ms, 2.26 GB | out of memory (8 GB) |
| Energy/forces training step | 56 ms, 322 MB | 66 ms, 349 MB | 113 ms, 846 MB | 234 ms, 717 MB | 228 ms, 1.07 GB | 325 ms, 1.87 GB |

b1 = batch 1 with 1,000 grid points; b8 = batch 8 with 5,000 grid points.

Notes:
- **Torch memory:** the peak of PyTorch's own GPU memory counter.
- **JAX memory:** comes from XLA's allocator, which reserves memory in large blocks, so these figures aren't directly comparable to the Torch ones.
- **Timing statistic:** each cell is the median of the best of three warm rounds. Some rounds overlapped with other GPU jobs, so the best round is the cleanest measurement.
- **Numerical agreement:** against the `densnet-performance` eager outputs, both compiled routes, the optimization eager route and JAX energy/forces agree to fp64 round-off (relative ≤ 4e-12). JAX density differs by ~1e-10 relative, and its density gradient by up to ~7e-8 relative.
- **JAX density training at b8:** this ran out of memory on a single 3.18 GiB allocation, also with preallocation off, command buffers disabled and a 0.95 memory fraction.
- **JAX GPU support:** the JAX lock file only pulls `nvidia-cublas` and `nvidia-cuda-nvrtc`, so JAX fell back to CPU. Running on the GPU required `pip install "jax-cuda12-plugin[with-cuda]==0.9.2"`.
- **Precision:** float64 runs at 1/64 speed on this GPU, so at b8 every route is arithmetic-bound and compiling gains less than it does in float32.
- **Compile time:** cold start (setup plus first call, empty compile cache) is 8–25 min per mode for the fixed-topology route, 1–6 min for the dynamic route and 1.5–3 min for JAX.
- **Density training step at b8:** the dynamic route is faster in eager mode (326 ms) than compiled (384 ms).
- **Method:** the harness was run directly, without the Apptainer runner, with 3 quiet warm timing rounds per cell. The dynamic-compile columns used an adapter script that loads the same fixtures and weights and checks outputs against the eager results. On the merge branch, the `torch-compile` backend of the harness runs the dynamic compile route directly.

## Changes after the merge

**One compiled graph for any batch composition**
- The packed coefficient layout is built per batch outside the compiled region and passed in as index tensors.
- All batch-dependent sizes (molecules, atom columns, atoms, pairs, shells per L, coefficients) are unbacked symbolic sizes, so batch size 1, single atoms and missing L shells don't specialize.
- 18 mixed compositions (water, water clusters, ethanol, ethanethiol, resorcinol, random H/C/N/O/S mixes, single atoms; batch sizes 1–6, 1–53 atoms): 1 graph, 0 recompiles, 0 graph breaks, in fp32, fp32 with density gradient and fp64. Compiled vs eager in fp64: energy 3e-15, forces 2e-14, gradients 1e-12.
- Compiled training step speed is equal or better (water8 116 → 93 ms) and peak memory is unchanged.
- Uses two inductor settings and a patch of inductor's output lowering (strides of outputs with data-dependent sizes), which may need adjusting on a torch upgrade. The first compile of the full model takes ~8 min on this GPU.
- Test script: `benchmarks/compile_mixed.py --device cuda`.

**Fixed-shape compile mode for MD**
- `DFTNetwork.enable_compile(static=True)`, used by `DFTNetworkCalculator(static_compile=True)` and `--static_compile` in the MD script; off by default.
- Static shapes plus one manually captured CUDA graph per composition. Neighbor pairs are padded to a grow-only capacity with real out-of-cutoff pairs, which contribute exactly zero, so MD steps don't recompile.
- Inference and MD only, CUDA only.

| Case (ethanol, fp64, ms) | Static | Dynamic compile | Eager | Fixed-topology compile | JAX |
|---|---|---|---|---|---|
| b1 energy + forces | 29–30 | 35–36 | 86–111 | 33 | 39 |
| b1 calculator | 29–32 | 35–38 | 83–104 | 34 | 41 |
| b8 energy + forces | 115 | 120 | 145–158 | 132 | 149 |
| b8 calculator | 115 | 120–122 | 148–169 | 135 | 150 |

The fixed-topology and JAX columns are from the benchmark above. Reserved GPU memory: b1 328 MB (static) vs 390 MB (dynamic), b8 ~820 vs 804 MB. MD runs (two ethanols, up to 41 steps, three cutoffs) match eager to ~1e-14 with one graph and no recompiles. At b1 the step is now GPU-bound, about half of it in the fused CG kernel.

**Delta learning (`remove_atom_density` + `append_atom_density`)**
- `append_atom_density` (free-atom DF coefficients appended to the L=0 inputs of the energy head) failed on the packed core. It now uses per-element lookup tables on the packed path, in the old channel order, eager and compiled (0 recompiles on mixed batches).
- New reference configs `delta` and `delta_noappend` in `benchmarks/reference.py`, captured from the original code; both match.
- An existing ethanol delta checkpoint loads strictly and reproduces the original code's energies and forces to ~3e-14. The MD calculator and the static MD mode match eager on it.
- Behavior change for mixed-element batches: the original code wrote the free-atom channels of every element present at an atom slot into every molecule at that slot. Each atom now gets its own element's channels. Batches of a single composition are unchanged.

**Free-atom densities: per-element s-Gaussian fits**
- `scripts/data/fit_free_atom_sgauss.py` uses per-element numbers of s functions by default: H 18, C 36, N 36, O 36, F 40, S 44, Cl 44 (relative L1 error < 1e-4 and Coulomb error J < 1e-2 kcal/mol per element). `--n_s N` still forces a uniform count. Output: `datasets/free_atom_densities_augccpvdz_sgauss_pe.npy`.
- N and the nuclear attraction V are exact for these fits (analytic check: |dN| ~1e-15, |dV| < 1e-9 kcal/mol). For comparison, the jkfit l=0 fit used by the paper models has V errors of 13, 32, 38 and 145 kcal/mol for C, N, O and S.
- `sample_atom_density_sgauss` samples each element with its own number of functions: 64 waters, 20k grid points on CPU, 1.38 s (uniform 60, padded) → 0.64 s. Identical to the padded version to ~1e-18.
- All older free-atom density files stay available and selectable with `--atom_dens_path`.

**Fixes**
- `DFTNetwork.coeffs_head` was registered as a second submodule, which stored the coefficient head weights twice in the state dict and broke strict loading of older checkpoints.
- `layer_norm_orders` broadcast the per-order statistics along the wrong axis for a leading dimension > 1 (the model always uses 1, so outputs were not affected).

**For the JAX exporter (not used by the Torch forward)**
- The CG layers hold the shared `ClebschGordanMatrix` again (`cg_matrix` / `cg_masked`, non-persistent), sized to the model's maximum order instead of L=10.
- The energy head keeps the per-element dicts `atom_df_coeffs` / `atom_df_widths` / `atom_df_scales`; the Torch path uses `atom_df_*_table` buffers.

**JAX reference generators**
- `tests/jax_pipeline/generate_layer_fixtures.py`, `generate_energy_reference.py` and `generate_density_reference.py` pack their inputs for the Torch layers and split the outputs back to per-L, so the saved references keep the format the JAX tests read.

## Decisions

- There is a single PyTorch compiled route, the dynamic compile. It handles changing batch shapes and is faster on density and at batch 8. The fixed-topology route requires fixed inputs, which mixed-composition training can't provide. `compiled.py` and `regional.py` are removed.
- `equiv_dens_jax` is kept unchanged. It is to be ported to the final Torch version.

## Merge table

"Original" is master at b15a0ff, where both branches started. "Unchanged" means that branch did not modify the code. "Kept" is the state on `merge-densnet-performance`.

### Model and compiled code

| Area | `optimization` | `densnet-performance` | Kept |
|---|---|---|---|
| Spherical features | One packed tensor `[atoms, (L+1)², F]` | Unchanged (list per L) | optimization |
| CG tensor products | Fused Triton kernel for forward, backward and double backward. No dense CG matrices per layer (checkpoints 3 GB → 33 MB) | Unchanged. An experimental kernel was benchmarked but rejected | optimization |
| Density coefficient extraction | `PackedLayout`: one gather per L | Per-atom loop with a per-shell `index_copy` | optimization (the only real conflict) |
| Density on the grid | Fused, chunked expansion with the density gradient from autodiff | Unchanged in the model. The fixed-topology route has its own copy | optimization |
| Energy head | Rewritten so `torch.compile` can trace it. `abs` is computed as `x·sign(x).detach()` | Unchanged in the model. The fixed-topology route has its own copy | optimization |
| Spherical harmonic constants | `float(...)` | Same change | Identical on both branches |
| Compiled route | Dynamic compile: dynamic shapes, forces via `torch.func` | Fixed-topology compile: `FunctionalDenSNet`, `RegionalDensityDenSNet` | Dynamic compile as the single route. `compiled.py` and `regional.py` and their tests are removed |
| Benchmark harness | `benchmarks/bench_model.py`, `reference.py`, `profile_step.py` | `torch_compile_speed.py`, the performance-contract tests, `tools/performance/` | Both. The `torch-compile` backend now runs the dynamic compile route |
| MD calculator input prep | Small gpu4pyscf change | Optional `geometry_backend="native"` (`TorchAtomBatchLayout`) | Both. Tested with the new model core, eager and compiled |
| Neighbor list | Unchanged | Optional `backend="native"` with cached GPU pair candidates, plus a fix for the single-pair case | densnet-performance |
| JAX | n/a | `equiv_dens_jax`: training, inference, MD | densnet-performance, unchanged |

### PySCF and data

| Area | `optimization` | `densnet-performance` | Kept |
|---|---|---|---|
| Data loading | gpu4pyscf path (`utils/gpu_pyscf.py`, `--use_gpu4pyscf`). Removed a `gc.collect()` per batch. 127 → 16 ms per batch | Unchanged | optimization |
| Dipole integrals | Closed-form one-center `int1e_r` and a dipole table. 0.8 s → 0.009 s | Unchanged in `equiv_dens`. `equiv_dens_jax` has its own CPU PySCF version | optimization in the Torch code; the densnet-performance version in JAX |
| Free-atom densities | Exact s-Gaussian fits (per-element counts) plus a fast torch sampler | Unchanged | optimization |
| Coefficient → PySCF vector (`coeffs_dict_to_vector`) | Unchanged. The main path exports through `PackedLayout.to_df_vector` | One concatenation at the end and fewer host syncs | densnet-performance |
| `pyscf_gto_factor` | Unchanged | Python `float` | densnet-performance |
| Dataset generation | Contracted DF solve, gpu4pyscf gradients plus D4. (H2O)8 341 → 56 s | Unchanged. New CPU-PySCF label export and audit scripts | Both |
| gpu4pyscf | Used in loading, generation, dipole and SAPT0. Includes a fix for the torch/gpu4pyscf cusolver library clash | Installed in the container (CUDA 12) and checked in a smoke test; no new code uses it | optimization (the env uses CUDA 13) |

### Infrastructure and other

| Area | `optimization` | `densnet-performance` | Kept |
|---|---|---|---|
| Runtimes | Plain conda env (torch 2.14.1, CUDA 13) | Locked Apptainer Torch and JAX images, qualification scripts | Both |
| g-xTB calculator | Unchanged | Parser fix with regression tests | densnet-performance |
| Configs and data | Even-tempered basis recipes, water clusters | Ethanol/ethanethiol reference configs and splits | Both |
| `.gitignore` | Agent notes and the benchmark reference data folder | Ignores `/docs/`, `AGENTS.md` and `/tools/agent/` | Both. New files under `/docs/` are ignored |

## Notes for the JAX port

- **Feature layout:** packed `(L+1)²` m-components per atom.
- **CG products:** the same channel-wise math. Weights are per path, from the nonzero CG triples of each (l1, l2, L) path. The exported model contains `cg_matrix` / `cg_masked` again.
- **State dict:** parameter names are unchanged; CG buffers are non-persistent and not saved.
- **Coefficients:** `PackedLayout` order. The `df_coeffs` export is unchanged. `PackedLayout` rounds the initial radial coefficients (`width0`, `scale0`) to float32, as the original models had them.
- **Energy head:** `|x|` is computed as `x·sign(x).detach()`, which has the same values and derivatives.
- **Gradients of unused parameters:** the Torch core returns zero tensors instead of `None` for the 156 structurally unused parameters of the pipeline fixture.
- **Free-atom densities:** the per-element s-Gaussian file is a drop-in replacement with `--atom_dens_type=df_coeffs`.
- **PySCF:** `utils/gpu_pyscf.py` and the closed-form dipole code can replace the CPU PySCF paths in `equiv_dens_jax`.
- **Environment:** the containers use CUDA 12 builds of gpu4pyscf and cupy; the development env uses CUDA 13. The JAX GPU plugin needs `jax-cuda12-plugin[with-cuda]`.

## Verification status (`merge-densnet-performance`, 232ce1f)

- `benchmarks/reference.py compare`: ALL PASS on base, parity, dgrad, norm2, delta and delta_noappend (CPU, worst relative error ~1e-13) and water8 (CUDA).
- Torch tests (`tests/` without `jax_pipeline`): 164 passed, 29 skipped, 0 failed. The skips need the `paper/models` checkpoints or CUDA.
- `tests/performance_contracts` (CPU): 77 passed, 11 skipped, after removing 24 contract test functions that exercise the old per-L feature format, the dense CG buffers or the removed fixed-topology route.
- `run.py train` smoke runs (h2o_small, 20 steps), eager and `--compile=True`, with default flags and with delta learning: exit 0.
- `DFTNetworkCalculator`: eager, compiled and static, with `geometry_backend="native"` and `"reference"`: energies agree to ~1e-14.

## JAX status (`equiv_dens_jax` unchanged)

CPU run with references regenerated from the current Torch code: **199 passed, 3 failed, 0 errors** (directly after the merge: 105 passed, 20 failed, 48 errors). The remaining failures:
- `test_density.py::test_torch_density_head_reference_bundles`: numerical, a port item. Torch rounds the initial radial coefficients to float32 and JAX keeps float64: ~3e-8 in `radial_width` / `radial_scale`, amplified to 3.6e-6 in the integral-constraint variant.
- `test_parameter_usage.py::test_complete_pipeline_matches_all_torch_none_gradients` and `test_pipeline.py::test_weight_decay_skips_torch_none_gradients_and_keeps_optimizer_state`: the tests expect Torch to report `None` gradients for structurally unused parameters, and the Torch core now returns zeros. The gradient values agree.

## Known issues, not changed

- `SphericalLinearEnergyNetwork` sums the energy over all atoms of the batch instead of per molecule (same in the original code).
- `PackedLayout.from_df_vector` detaches the DF coefficients, so no gradient flows back into input `df_coeffs`.
- `.gitignore` from `densnet-performance` ignores `/docs/`; files in `docs/` are added explicitly.
