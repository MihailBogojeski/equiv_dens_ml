# JACS revision campaign implementations

Commands and status live in [docs/jacs_revision_todo.md](../../docs/jacs_revision_todo.md).
Live status: [results/revision/STATUS.md](../../results/revision/STATUS.md).

Training and MD still run through `python run.py … @config/…`. This folder is
only the peer-review campaign (labelling, watchdogs, OOD analysis).

## Command interface

`python run.py experiment jacs list` lists explicit executable tasks.
`python run.py experiment jacs run water/campaign_status --root datasets/revision/shards`
runs the existing status command. `experiment jacs submit JOB [SBATCH_OPTIONS]`
submits a registered Slurm file; listing tasks does not execute them.
All prior `scripts/revision/` invocations remain available through compatibility
entry points. Slurm definitions live in `slurm/`, with byte-identical old-path aliases.
The current production checkout is unchanged until post-resubmission adoption.

## Layout

| Folder | What lives there |
| --- | --- |
| [`lib/`](lib/) | Shared theory levels, shard claims, ORCA/PySCF label workers, SAD helpers |
| [`water/`](water/) | Water/malonaldehyde geometries, CVs, H-bond eval, assemble/launch/watch |
| [`csh/`](csh/) | OMol25/CSH conversion and the permutation-probe suite |
| [`qm7x/`](qm7x/) | QM7-X extract / shard / assemble (worker is in `lib/`) |
| [`analysis/`](analysis/) | OOD reports, density metrics, IR/NVE/Figure 2, GPU/DFT campaign launchers |

Slurm entry points stay at `scripts/revision/submit_*.sbatch` so
`sbatch scripts/revision/submit_water_train.sbatch` is unchanged.

Thin wrappers at the old `scripts/revision/<name>` paths still work for the
watchdog and the todo’s “how to run” commands.

## Campaign commands

```bash
scripts/revision/launch_water_campaign.sh wb97mv_def2tzvpd
scripts/revision/launch_water_campaign.sh pbe_d4_avdz
sbatch scripts/revision/submit_water_watchdog.sbatch
python scripts/revision/campaign_status.py --root datasets/revision/shards
scripts/revision/run_ood_analysis.sh wb97mv_def2tzvpd
```

Those resolve to `water/` and `analysis/` via the wrappers. The same commands
can be run from the themed paths directly.

`launch_water_campaign.sh` reads `WATER`, `OOD`, `MALON`, `SHARD_ROOT`, and
`OUT_ROOT` from the environment so a wave can point at an expanded tree without
touching the finished 3k campaign. `launch_md_expansion_waves.sh` sets those
for Phase A / hybrid-10% / Phase B.

## Geometry expansion (MD 10×, then new motifs)

The labelled 3k set placed water frames by Gaussian jitter around a handful of
analytic minima. Reviewer-facing sampling is now a short equilibrated GFN2-xTB
Langevin trajectory per parent (the same engine already used to place
malonaldehyde and droplet frames), subsampled at 50–200 fs. Ice keeps
rigid-molecule translation/libration so the oxygen network stays crystalline.

```bash
# Phase A: 10× the current parents → datasets/revision/*_md10x/
scripts/revision/water/run_geometry_expansion.sh phase_a   # or:
python scripts/revision/water/thicken_with_md.py --calculator auto --ice-mode rigid --workers 4

# Phase B: new motif / pack / genice / ETKDG parents, then the same thickener
python scripts/revision/water/generate_new_motif_parents.py --water --ice --malon
python scripts/revision/water/thicken_with_md.py \
  --water-dir datasets/revision/water_clusters_newmotifs \
  --ood-dir datasets/revision/water_ood_newmotifs \
  --malon-dir datasets/revision/malonaldehyde_newmotifs \
  --water-out datasets/revision/water_clusters_newmotifs_md10x \
  --ood-out datasets/revision/water_ood_newmotifs_md10x \
  --malon-out datasets/revision/malonaldehyde_newmotifs_md10x

# DFT is waved, not one giant submit. Default is --build-only.
scripts/revision/water/launch_md_expansion_waves.sh wave1          # PBE-D4 on Phase A
scripts/revision/water/launch_md_expansion_waves.sh wave2          # ωB97M-V on 10%
scripts/revision/water/launch_md_expansion_waves.sh wave3          # PBE-D4 on Phase B

# md40x (~100k): do not change the 10× default. Budget first, then thicken, then waves.
./.venv/bin/python scripts/revision/water/plan_md40x_budget.py
scripts/revision/water/run_geometry_expansion.sh phase_md40x
scripts/revision/water/launch_md40x_waves.sh pbe                  # PBE-D4 on every kept frame
scripts/revision/water/launch_md40x_waves.sh hybrid               # ωB97M-V on the same XYZ
REQUIRE_DIPOLE=1 OUT_ROOT=results/revision/water_orca_md40x \
  SHARD_ROOT=datasets/revision/shards_md40x \
  LABELS=datasets/revision/water_labels_md40x \
  MALON_LABELS=datasets/revision/malonaldehyde_labels_md40x \
  scripts/revision/water/assemble_water_labels.sh pbe_d4_avdz
```

Full 100× × two theories is not a single watchdog cycle. New frames are first
an eval set; do not dump 125k water frames into one `run.py` train job.

## Shared modules (`lib/`)

| Script | Role |
| --- | --- |
| `theory_levels.py` | Named theory entries (ωB97M-V, PBE-D4, …) |
| `shard_claim.py` | Exclusive shard locks for CPU/GPU arrays |
| `qm7x_orca_worker.py` | One-shard ORCA worker (water and QM7-X) |
| `qm7x_orca_common.py` | ORCA input/output helpers |
| `pyscf_labeler.py` | PySCF / gpu4pyscf single-geometry labels |
| `generate_dft_labels.py` | Batch PySCF labels |
| `build_sad_prior.py` | Free-atom SAD prior |
| `build_shards_from_xyz.py` | XYZ → JSON shards |
| `latest_run_dir.sh` | Newest resumable `results/revision/…` run dir |
| `paths.py` | Repo-root lookup (walks up to `pyproject.toml`) |
| `fetch_python_headers.sh` | CPython 3.12.9 `Include/` for Triton gcc (`tools/python312-headers/`) |
| `fetch_dftbplus.sh` | DFTB+ 24.1 binary + 3ob-3-1 SK files (`tools/dftbplus/`) |
| `triton_headers.py` | `CPATH` helper + explicit AIMNet2 device |
| `dftb_calculator.py` | DFTB/3ob SCC factory for Figure 2 (`DftbPlus24Calculator`) |

## Water (`water/`)

| Script | Role |
| --- | --- |
| `generate_water_cluster_geoms.py` | (H2O)_n minima + thermal frames |
| `generate_ood_water.py` | Ice / droplet OOD ladder |
| `generate_malonaldehyde.py` | Proton-transfer scan |
| `thicken_with_md.py` | Phase A: 10× each parent by subsampled GFN2 Langevin (`--n-keep 40` for md40x) |
| `generate_new_motif_parents.py` | Phase B: extra isomers, random-pack minima, more ice/malon |
| `import_rackers_parents.py` | Rackers/Rakshit XYZ → train / size-OOD / `ood_rackers` hold-out |
| `merge_xyz_trees.py` | Concatenate geometry trees and write `by_n` manifests |
| `plan_md40x_budget.py` | Expected frames and ORCA core hours for n_keep=40 |
| `select_stratified_subset.py` | 10% stratified draw for the hybrid DFT wave |
| `launch_md_expansion_waves.sh` | Wave 1 PBE-D4 / wave 2 ωB97M-V 10% / wave 3 Phase B / md40x |
| `launch_md40x_waves.sh` | PBE all, then hybrid all, on `*_md40x/` |
| `launch_water_campaign.sh` | Build shards and submit ORCA arrays |
| `watch_water_campaign.sh` | Resubmit shards, then train, then OOD analysis |
| `assemble_water_labels.sh` | Merge shard results into training NPYs |
| `campaign_status.py` | Outstanding-shard report for the watchdog |
| `evaluate_hbond_metrics.py` | O···O / O–H histograms |
| `relax_water_clusters.py` | DenSNet relaxation of n=8,10,12 |

## Analysis (`analysis/`)

| Script | Role |
| --- | --- |
| `eval_dipole_mae.py` | DFT `dipole_moment` vs predicted μ (Debye); same origin |
| `run_ood_analysis.sh` | CVs, overlap, H-bond, error-vs-distance |
| `ood_overlap_report.py` | Train/test CV separation |
| `error_vs_distance.py` | Headline error-versus-distance curve |
| `pbe_vs_hybrid_table.py` | PBE vs ωB97M-V table |
| `analyze_density_metrics.py` | SAD correction on a grid |
| `benchmark_figure2.py` | Timing table for Figure 2 |
| `ir_vs_length.py` | IR vs trajectory window |
| `run_gpu_campaign.sh` | Wave C tasks: `quick`, `pbe0`, `ethanol-md`, … |

## QM7-X (`qm7x/`) and CSH (`csh/`)

| Script | Role |
| --- | --- |
| `qm7x/submit_qm7x_orca.sh` | `download`, `smoke`, `train`, `val`, `assemble` |
| `qm7x/submit_qm7x_orca_remaining_train.sh` | Start nohup watchdog for remaining train ORCA |
| `qm7x/watch_qm7x_campaign.sh` | ORCA arrays → assemble → stage-D energy (nohup) |
| `submit_qm7x_energy.sbatch` | Frozen-density energy training (strict two-stage) |
| `csh/csh_shard_to_df.py` | HDF5 Fock → DF coefficients |
| `csh/csh_shards_to_npy.py` | Package CSH shards for training |
| `csh/omol_csh_*.py` | AO-ordering / permutation probes |

Geometry scripts need only the Python standard library. DFT / MD / analysis
scripts need `.venv` (`pip install -e . -r requirements.txt` plus `mace-torch`,
`aimnet[ase]`, `tblite`).
