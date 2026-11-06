# Upstream integration, November 2026

Record of how the JACS-revision work was merged onto `MihailBogojeski/equiv_dens_ml` `master`
(`f05a471b`, PR #38) without changing the packed/compiled fast path. It supersedes
`docs/merge_mihail_fast_model.md`, which documents the October merge onto `adc07332`.

## What changed relative to upstream master

| Area | Change | Upstream files touched |
|---|---|---|
| Layer normalisation | Upstream PR #35 (explicit `seg` argument, `batch_segments`) is used as is. The context-variable port of October (`structure_segments`, `_segment_layer_norm*`) is removed, `DFTNetwork.forward` no longer wraps the pass. | none (upstream version taken verbatim) |
| `nn/dft_network.py` | `_skip_grid_property` (ASE energy/force calls carry no integration grid); after `_core` the non-tensor inputs (free-atom basis of the dipole head, grid specs) are copied back for the heads that run afterwards. | `nn/dft_network.py` |
| Heads | `has_df_sad` guard in the packed energy head; zero-denominator guards in the packed charge constraint. | `nn/property_output/energy.py`, `density.py` |
| Loader | Old checkpoints carry precomputed `.clebsch_gordan.cg_` buffers that the packed model no longer has; they are ignored when a checkpoint is checked against the model. | `training/model_loader.py` |
| Training | Energy-shape fix in `ErrorDict`, variance-scaling recovery, fused AMSGrad passthrough, `args.txt` round-trip, phase-dependent argument overrides, prefetch-iterator fixes. | `training/{errors,utils,trainer,parse_command_line_arguments}.py`, `utils/{orbitals,scaling}.py`, `data/{density_dataset,custom_samplers}.py` |
| MD | Additive ASE calculator (`DenSNetCalculator`, `load_densnet_calculator`, grid/density-head helpers, stale dataset path remapping) and the g-xTB calculator; upstream's `--geometry_backend`, system-atom-order forces and continue-from-checkpoint are kept and ported into `commands/md.py`. | `md/dft_network_calculator.py`, `md/gxtb_calculator.py` |
| Layout | `equiv_dens.cli` + `commands/` + `training/workflows/` replace the logic that lived in `train_all.py` and the top-level scripts; `scripts/` and `notebooks/` entries become thin shims or symlinks to `experiments/legacy/`. | `cli.py`, `train_all.py` |
| CI | The 22 `datasets/*.npy` fixtures upstream CI smoke-trains on are tracked again and un-ignored. `tests/conftest.py` registers `local_data` and `gpu` markers and skips them on a bare clone. | `.gitignore` |
| Revision work | `experiments/jacs_revision/{analysis,campaign,config,slurm,scripts}`, `config/` overlays, revision tests, the Hugging Face kit scripts (`scripts/data/*reproduction_kit*`). | none |

Not merged upstream: `results/` (score and evaluation files ship in the Hugging Face kit), `experiments/internal/`,
the manuscript and letter repositories (previously submodules), and the revision-planning notes.

## The merge itself

1. `integrate/upstream-2026-11` was branched from `merge/mihail-fast-model` (October state), `main` merged
   (16 commits, no `src/`), then `mihail/master` (`f05a471b`).
2. Two conflicts: `nn/modules/network_blocks.py` (resolved by taking upstream; the two segment tests were
   rewritten against `layer_norm_orders(x, seg)`) and `scripts/md/schnetpack_md_run.py` (the shim stays;
   upstream's changes went into `commands/md.py`).
3. The verified tree was sliced by path into five stacked branches, `mh/ci-compat`, `mh/model-fixes`,
   `mh/md-calculator`, `mh/cli-layout`, `mh/jacs-revision`; the last branch equals the integration tree
   (`git diff --stat` empty).

## Verification (6 Nov 2026, CPU only; no GPU timing was run)

- **Per-molecule layer norm reproduces our checkpoints.** The PBE-D4 4 Å energy head at 100k steps
  (`best_tTowpZVb.pth`, SHA-256 `f8ea049d…`) was re-scored on CPU with the integration code on the first
  12 frames of the ID test and size hold-out tiers and compared frame by frame with the reference scores
  of the same checkpoint (`cutoff_100k_evaluation_20261101/scores/v1/cutoff_4_100k_best`, A100, float32,
  TF32 off):

  | tier | frames | max Δ predicted energy / kcal mol⁻¹ | max Δ force MAE / kcal mol⁻¹ Å⁻¹ |
  |---|---|---|---|
  | ID test | 12 | 0.00011 | 0.00013 |
  | size hold-out | 12 | 0.00024 | 0.0016 |

  Batched (4 structures) and single-structure energies agree to 2e-6 kcal mol⁻¹ (the scorer's
  `--check-single`). These are float32 CPU-vs-GPU differences; a change of normalisation semantics would
  show up at the 0.1–1 kcal mol⁻¹ level.
- **Layer-norm unit tests** (`tests/test_layer_norm_orders_segments.py`, `tests/test_layer_norm_orders_compile.py`):
  7 passed. Under `torch.compile` the CPU inductor backend of torch 2.13 fails to generate code for the
  per-molecule `index_add` scatter (`CppCSEVariable` assertion); the test therefore uses `aot_eager` on CPU
  and inductor on CUDA. What it checks, dynamo honouring the segment argument across batches, is independent
  of the backend.
- **Upstream CI recipe on the integration tree** (`pytest tests/test_*.py`, CPU): 514 passed, 15 failed
  before the `local_data` markers; the 15 were config-consistency tests reading files that only exist in a
  populated checkout (13) and the spawn-safety test whose 20 s limit the import of torch and schnetpack alone
  exceeds on a loaded node (2, raised to 60 s). The per-branch results are in the pull-request texts.
- **Not verified here:** GPU step timing of the compiled/static paths on the merged tree, and GPU scoring.
  The October record (`docs/merge_mihail_fast_model.md`) has the eager-path timings on an L40S for the
  pre-#35 tree.

## Hugging Face kit

`muhammadhasyim/equiv_dens_ml_dataset` (private): 1753 files, 16.33 GiB, built by
`scripts/data/build_reproduction_kit.py` and verified by a full download and SHA-256 comparison against
`manifest.json`. `scripts/data/fetch_reproduction_kit.sh` downloads and links it into a checkout; `DATA.md`
has the details.
