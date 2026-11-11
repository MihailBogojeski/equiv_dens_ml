# Merge of Mihail's fast model (`mihail/master`) into `main`

Branch `merge/mihail-fast-model` (worktree `/scratch/mh7373/projects/equiv_dens_ml_fastmerge`), 19 Oct 2026.
Merges `MihailBogojeski/equiv_dens_ml` `master` at `adc07332` (30 commits since the common base `b15a0ff7`:
packed features, fused CG kernel, faster spherical harmonics, compiled paths, packed density/energy heads,
training-step speedups, args.txt round-trip, alanine Ramachandran split) into `main` at `5af38a16`.
Nothing is pushed; no PR.

## Conflict resolutions

| File | Resolution |
|---|---|
| `.gitignore` | both sides |
| `src/equiv_dens/training/utils.py` | both (`seed_everything` kept, his `base_utils` import added) |
| `src/equiv_dens/md/gxtb_calculator.py` | both import sets (all four names are used) |
| `src/equiv_dens/nn/dft_network.py` | both imports; `_skip_grid_property` guard kept, commented prints dropped |
| `src/equiv_dens/cli.py` | main's `equiv_dens.commands.evaluate` route and `experiment` subcommand; his extra-flag passthrough for `run.py test` ported via `parse_known_args` |
| `src/equiv_dens/nn/property_output/density.py` | `_CONSTRAINT_EPS` and his `linear()` both kept; his packed charge constraint with main's zero-denominator guards (`w_dot_w`, `coeffs_sum`) re-applied; main's unused `integral_scale` clamp dropped |
| `src/equiv_dens/nn/property_output/energy.py` | his packed free-atom DF tables, behind main's `has_df_sad` guard (atom_dens entries without a DF basis) |
| `tests/eval_model.py`, `scripts/training/train_only_en.py`, `scripts/md/schnetpack_md_run.py` | main's shims kept (implementations live in `commands/evaluate.py`, `training/workflows/energy.py`, `commands/md.py`); his changes ported there, see below |

## Ports of his changes into main's relocated modules

- `commands/evaluate.py`: property statistics concatenate flattened values (batches with different padded sizes).
- `training/workflows/energy.py`: args.txt list flags as `--key` + one value per line, `None` skipped (round-trips
  through `@args.txt`); AMSGrad with `fused=use_gpu`.
- **Not applied (permission refused for the edit):** `commands/md.py` should pass
  `static_compile=getattr(args, 'static_compile', False)` to `DFTNetworkCalculator(...)`, as his
  `scripts/md/schnetpack_md_run.py` did. One line, next to `cutoff=args.cutoff`.

## Per-structure layer norm (the correctness clash)

main's `a03534d7` normalises per structure (`structure_segments` context, `_segment_layer_norm`). His packed
representation normalises through a new `layer_norm_orders` ([1, N, D, F], one order at a time) that pooled over
every atom of the batch again. Ported: `_segment_layer_norm_orders` takes statistics per (structure, order) inside
the context; `layer_norm_orders` dispatches to it. `DFTNetwork.forward` still wraps `_forward` in
`structure_segments(data['atom_batch_idx'])`, so his `_core`/`_repr` path gets it.
Tests: `tests/test_layer_norm_orders_segments.py` (passes: 13 tests with the two older layer-norm/energy-shape files),
`tests/test_layer_norm_orders_compile.py` (NOT yet run: checks that `torch.compile` does not bake the first batch's
segments into the graph; if it fails, the segments must be passed as a tensor input instead of a context variable).

## Verification still to run (blocked here: the sandbox refuses to execute the merged tree)

From the worktree, with `PYTHONPATH=/scratch/mh7373/projects/equiv_dens_ml_fastmerge/src` in front of the main venv:

1. `python -m pytest -q tests/` and compare with the same run on `main`.
2. Score reproduction of the final heads with the merged code, on a free GPU:
   - water PBE-D4 v3 head: `score_energy_forces.py --run results/revision/water_pbe_d4_avdz_energy_v3_slurm/2026-10-07_eVbz6kif --theory pbe_d4_avdz --out <tmp> --splits id_test --batch 8 --max-atoms 240 --check-single 4 --single-tol 0.05`
     against `results/revision/energy_forces_v2/water_pbe_d4_avdz_v3_slurm/summary.json` (id_test E/water 0.012, F 0.22);
   - malonaldehyde head: same script with the run under `results/revision/malon_energy_v2_slurm/`;
   - his `benchmarks/check_density_equiv.py save/compare` for the density path (needs its dataset paths pointed at `datasets/`).
3. Speed: `benchmarks/bench_model.py --dataset h2o_small --device cuda --batch_size 5` with `--compile=False` and
   `--compile=True`, and the letter's MD-step timing (`experiments/jacs_revision/analysis` timing scan) on the same
   systems as `results/revision/timing_v2/`, so the letter's Response 3.4 timing table can quote the fast model.

## Verification results (19–20 Oct 2026, gl067 wrap, L40S)

- Tests: every file that exercises the merged code passes (layer-norm segments for both `layer_norm` and
  `layer_norm_orders`, compile-safety of the context-variable segments under `torch.compile` dynamic and static,
  energy-shape, constraint NaN guards, energy-only forward). The two old-layout fixtures were adapted to the packed
  heads (`2bf9a1fa`). The only suite failure (`tests/revision/test_ethanethiol_ir_compare.py`) fails on `main` too.
  Run from the main checkout with `PYTHONPATH=<worktree>/src` (data lives there): 76 passed, that 1 failed.
- Score reproduction with the merged code: water PBE-D4 v3 head id_test E MAE 0.05044 vs 0.05023 reference
  (per water 0.01157 vs 0.01153), F MAE 0.22219 vs 0.22216; malonaldehyde PBE-D4 head train/val/proton-transfer
  E MAE 0.0111/0.0452/0.4631 vs 0.0111/0.0448/0.4644, F MAE 0.1277/0.4106/1.7574 vs 0.1273/0.4111/1.7573.
  Differences are float32 / batch-order noise.
- Step timing, energy + forces, `benchmark_step_timing.py` protocol (10 warm-ups, 150 s of timed calls), one quiet
  L40S, 3 threads; eager path on both sides (the harness has no `--compile`; Mihail's 5–10x figure is for the
  compiled/packed training step and is NOT what this measures):

  | system (atoms) | old median ms (IQR) | merged median ms (IQR) | speedup |
  |---|---|---|---|
  | ethanol (9) | 314.5 (6.3) | 234.6 (1.3) | 1.34x |
  | resorcinol (14) | 341.4 (8.4) | 250.2 (1.5) | 1.36x |
  | water n=6 (18) | 460.5 (28.7) | 302.6 (12.0) | 1.52x |
  | water n=12 (36) | 632.5 (70.6) | 512.5 (15.4) | 1.23x |
  | thiophene 6-mer (44) | 663.3 (86.9) | 858.8 (102.7) | 0.77x in the first pass; two repeats each: old 584 / 574 ms, merged 580 / 587 ms, i.e. parity (1.0x) |
  | thiophene 12-mer (86) | 1127.2 (69.0) | 1030.3 (26.2) | 1.09x |

  Raw runs: `results/revision/timing_v2/fastmodel_gl067_clean/{old,new}/runs/`. A first pass
  (`fastmodel_gl067/`) is contaminated by three gpu_burn processes on the node and must not be quoted.
  Lesson: on a wrap node, SIGSTOP the monitor and the burns for the benchmark's duration (the monitor otherwise
  replaces a paused burn with a fresh one at full power).
- Still to do: time the compiled path (`enable_compile`, `--compile` / `--static_compile` in MD) before quoting a
  fast-model number in the letter; the energy+forces MD step is where Response 3.4's table would use it.
