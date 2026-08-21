# JACS Resubmission Tracker — ja-2026-12808m

**Manuscript:** Enhancing molecular dynamics with equivariant machine-learned densities  
**Editor:** Connor Coley  
**Decision:** reject; resubmission invited  
**Hard deadline:** 13 February 2027  
**Immutable review text:** [ja-2026-12808m-reviews.md](../ja-2026-12808m-reviews.md)  
**Response-letter outline:** [jacs_response_letter_outline.md](jacs_response_letter_outline.md)  
**Campaign scripts:** [scripts/revision/](../scripts/revision/)

Status values: `not_started` | `scripts_ready` | `geoms_ready` | `running` | `computed` | `written` | `declined`

Action types: `new DFT` | `new MD` | `analysis` | `manuscript` | `decline` | `recover`

---

## Recovered assets (2026-08-17)

Searched this checkout, `paper/`, `datasets/`, and `/home/ml-dft/equiv_dens/datasets` (path missing).

| Asset | Status | Path |
| --- | --- | --- |
| Ethanol paper-split densities `ethanol_dft_*.npy` | **Still missing** by that name | not found on scratch, LFS, or the Hasyim clone |
| Ethanethiol / resorcinol / thiophene extras | **Local only** | `datasets/{ethanethiol_*,resorcinol_*,thiophene_*}` (excluded from this branch) |
| Published DenSNet checkpoints | **Local only** (103 `*.pth`, 12G) | `paper/models/` including ethanol `2024-03-22_96w7KyGG` |
| Water-monomer DFT (dev set, not H-bond study) | Present | `datasets/h2o_small_{train,valid,test}_augccpvdz*.npy` |
| SAD prior (PBE) | Present | `datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy` |
| Ethanol geometry stub | Present | `datasets/ethanol_train_10.{npy,xyz}` (10 frames) |
| Ethanol AIMD | Present | `results/aimd_benchmark/aimd_ethanol_rep0_10ps.traj` (**10 ps**) |
| g-xTB binary | Present | `g-xtb/binary/gxtb` |
| Manuscript / SI TeX | **Local only** | `scratch/recovered_manuscript/` (copied from sibling before delete) |
| Classical-MD logs (FF, length, SHAKE) | **Missing** | cannot recover SI S2.2 from git |

Recovered checkpoints/datasets stay untracked (`.git/info/exclude`). Sibling `/scratch/mh7373/projects/equiv-paper` and the Hasyim clone were deleted after copy.

---

## Reviewer checklist

### Reviewer 1

| ID | Request | Action | Status | Deliverable |
| --- | --- | --- | --- | --- |
| R1.1 | H-bond / size extrapolation (water clusters; NMA/MeOH/AcAc optional) | `new DFT` + train | `running` | **Labelling complete: 513/513 shards at both ωB97M-V/def2-TZVPD and PBE-D4, every cluster-size bucket at 100%.** Assembly had never once written a training file — `mol_from_pack` dropped the return of the classmethod `gto.Mole.unpack`, so every frame carried `natm=0` and the `--min-complete` gate correctly refused all of them; that is why each training job died within 90 s. Training now running on `a100_chemistry`. Predicted H-bond histograms come from `relax_water_clusters.py`, which minimises the n=8,10,12 clusters under the model, feeding `evaluate_hbond_metrics.py --pred-xyz` (now emits binned predictions, overlap, Jensen–Shannon and peak shift, not just means) |
| R1.2 | Intentional OOD conformations | `new DFT` | `running` | Four-tier OOD ladder built and its separation quantified ([Calculation 7](#calculation-7--water-ood-ladder-and-hybrid-labels)). Distance from training, in units of the training set's own width: ID test 0.01, size-OOD 0.36, ice 1.90, droplets 2.79. Ethanol OOD DFT ~80/190 as before |
| R1.3 | Shorten polythiophene; move Fig. 6 | `manuscript` | `written` | outline §R1.3 |
| R1.4 | Move Fig. 7 to SI | `manuscript` | `written` | outline §R1.4 |
| R1.5 | SI S2.2 details + train/test overlap evidence | `analysis` + `recover` | `computed` | resorcinol median min-RMSD 0.093 Å (1000/1000 <0.2 Å); ethanethiol 0.600 Å (0/1000 <0.2 Å); thiophene-4mer 1.15 Å; thiophene-6mer 0.53 Å; 12-mer is size-OOD. SI already has PINY_MD / NHC / 1 fs; FF/SHAKE/length not recovered |
| R1.6 | Deduplicate 755 ms vs 5 min | `manuscript` | `computed` | CPU Figure 2 ethanol: DenSNet 1.62 s, g-xTB 0.70 s, GFN2-xTB 3.05 s, MACE-OFF 4.48 s; **GPU DenSNet 467 ms/step** (0.093 ns/day). AIMNet2/MACE GPU still fail Triton/`Python.h` |
| R1.7 | Retract unsupported dielectric / H-bond-network claims | `manuscript` | `written` | retracted in `scratch/recovered_manuscript/main/nat_manuscript.tex` Discussion |

### Reviewer 2

| ID | Request | Action | Status | Deliverable |
| --- | --- | --- | --- | --- |
| R2.1 | Cite Hazra–Sanvito JCP 2025 density→IR workflow | `manuscript` | `written` | outline citations |
| R2.2 | Compare general IR methods (MACE4IR, AIMNet2, TranSpec, AIQM) | `new MD` + `manuscript` | `running` | CPU timings include DenSNet 1.62 s/step (~0.027 ns/day); ethanol 500 ps MD launched on GPU 1; GPU Figure 2 + AIMNet2 retry launched |
| R2.3 | Tougher extrapolation than hexamer-trained oligomers | `manuscript` | `running` | Size hold-out alone is the weak tier and is reported as such (0.36 training widths). The tougher tests are crystalline ice, low-density droplets and the malonaldehyde proton transfer ([Calculation 7](#calculation-7--water-ood-ladder-and-hybrid-labels)) |
| R2.4 | Companion network: embedding, autodiff, energy conservation | `new MD` + `manuscript` | `computed` | 20 fs CPU NVE: std 5.4 meV, drift 0.30 eV/ps; **GPU 0.1 ps NVE**: std 8.4 meV, drift 0.18 eV/ps (same energy-head caveat) |
| R2.5 | Who produces dipoles vs energy/forces | `manuscript` | `written` | outline §R2.5 |
| R2.6 | One framework vs two networks at MD time | `manuscript` | `written` | outline §R2.6 |
| R2.7 | Electron-count normalization vs delta-learning | `manuscript` | `written` | outline §R2.7 |
| R2.8 | Softplus vs signed delta-density | `manuscript` + `analysis` | `computed` | Measured on a grid rather than inferred from fitting coefficients (the old DF-coefficient figures were not densities and were not comparable between molecules — an over-complete auxiliary basis lets large cancelling coefficients leave the density untouched). On the 30-frame water ID test set at PBE-D4: **84% of the volume carries a negative Δρ**, so the correction is genuinely signed and the softplus constrains ρ_SAD + Δρ, not Δρ. Reference fits integrate to the right electron count to ~1e-4 ([density_metrics_delta_vs_direct_pbe.json](../results/revision/density_metrics_delta_vs_direct_pbe.json)) |
| R2.9 | State max AIMD time in main text | `manuscript` | `written` | recovered: ethanol AIMD **10 ps** on disk |
| R2.10 | Units for density errors | `manuscript` | `written` | use **e/a0³** and relative MAE |
| R2.11 | Numerical evidence DenSNet ≈ DFT equilibria | `new MD` | `blocked` | Ethanol PBE reference minimum now computed (−4214.07 eV). The DenSNet side is **not quotable**, and the reason is now known precisely rather than guessed. Three loader defects were real and are fixed — `--args-file` ranked below the run directory's `args.txt` so it was ignored, `L0_start` defaulted on and added untrained `radial_L0_map` layers, and `atom_dens_path` was overwritten with the revision prior — and the checkpoint now loads deterministically with 0 missing / 0 unexpected tensors. But its `VarianceScaling` factor is absent (it predates the buffer) and `datasets/ethanol_dft_pyscf_ccpvdz_train.npy` is gone, so predictions are in normalised units and the relaxation barely moves. Fitting the constant on training geometries gives 90.36 with a predicted-vs-reference force correlation of only 0.646, so scale is not the only thing missing: **this needs the 2024 training set restored** ([output_scaling_ethanol.json](../results/revision/output_scaling_ethanol.json)) |
| R2.12 | Cite Cuevas-Zuviría / Pacios JCIM 2020 | `manuscript` | `written` | outline citations |
| R2.13 | Corresponding authors MS vs SI | `manuscript` | `written` | outline §R2.13 |

### Reviewer 3

| ID | Request | Action | Status | Deliverable |
| --- | --- | --- | --- | --- |
| R3.1 | Hybrid DFT densities (PBE0+MBD / ωB97M-V) | `new DFT` | `running` | ωB97M-V/def2-TZVPD, the level Reviewer 3 named, is running on every water and malonaldehyde frame, with PBE-D4 on the *same* geometries so the PBE-vs-hybrid table isolates the reference data ([Calculation 7](#calculation-7--water-ood-ladder-and-hybrid-labels)). In-house PBE0 70/70 labels done. OMol25/CSH 8-element run continues separately |
| R3.2 | Compare SO3LR, MACE-OFF, GFN-xTB, DFTB; expand Fig. 2 | `new MD` | `running` | CPU Figure 2 includes DenSNet 1.62 s; GPU Figure 2 (DenSNet/MACE-OFF/AIMNet2/SO3LR) launched; DFTB binary still missing |
| R3.3 | Relative density errors + uncertainty | `analysis` | `running` | The trainer exposed only `--split_seed`, which varies the train/valid/test partition; an ensemble built on it measures sensitivity to the split, not run-to-run spread, and left weight initialisation unseeded and irreproducible. New `--init_seed` seeds python/numpy/torch before any module is built; three wB97M-V water members queued, each with its own `save_dir` so they cannot resume from one another. `pbe_vs_hybrid_table.py` aggregates them as mean ± std per tier |
| R3.4 | Direct vs delta cost and correction magnitude | `analysis` | `computed` | **The SAD prior already supplies 89% of the density: ∫\|Δρ\|/∫\|ρ\| = 0.110** over the water ID test set, so the network predicts a tenth of what direct learning would have to. The prior costs 0.18 s/frame as a free-atom lookup against a full SCF. Direct-learning arm (`RUN=water_direct`, `--remove_atom_density=False`) queued so the two differ only in the target ([density_metrics_delta_vs_direct_pbe.json](../results/revision/density_metrics_delta_vs_direct_pbe.json)) |
| R3.5 | Hyperparameter / cutoff optimization | `analysis` | `running` | Cutoff `{4,5,6,8}` and seed overlays on `water_clusters_001.txt`; the polling loop that gated them was retired in favour of the campaign watchdog |
| R3.6 | IR convergence vs trajectory length | `new MD` | `running` | Neither trajectory could have produced a spectrum as configured. Both MD configs named `datasets/revision/sad_pbe_augccpvdz.npy`, a prior with no `df_coeffs`: for ethanol that swapped the delta reference and the architecture block was missing entirely (density branch built with 32 radial basis functions against the checkpoint's 128); for the thiophene `_ext` checkpoint it halved the energy head's L=0 width from 60 to 30, which is the size mismatch that stopped that run from ever starting. The `_minimized` prior supplies exactly the 5+11+14 coefficients for H, C and S. Both now load with 0 missing / 0 unexpected, and `--dpm_intor` is on so dipoles are actually logged. Ethanol additionally needs the R2.11 scaling blocker resolved before its spectrum means anything |
| R3.7 | General-purpose density model / active learning | `manuscript` | `written` | discussion only; outline §R3.7 |
| R3.8 | Soften “spectra match across the full range” | `manuscript` | `written` | wording updated in recovered `nat_manuscript.tex` |
| R3.m1 | Two-stage training cost | `analysis` | `scripts_ready` | log GPU-hours in calculation log when jobs run |
| R3.m2 | Figure 1 architecture in workflow | `manuscript` | `written` | outline §R3.m2 |
| R3.m3 | Broader electronic properties from ρ | `manuscript` | `written` | outline §R3.m3 |

---

## Calculation 1 — Water clusters

**Why:** R1.1 / R1.7. Highest-leverage new science.

**Design**

| Split | Sizes | Target frames |
| --- | --- | --- |
| Train | n = 2–6 | 250 per size |
| Val | n = 2–6 | 50 per size |
| ID test | n = 2–6 | 50 per size |
| Size-OOD test | n = 8, 10, 12 | 100 per size |

**Geometry command**

```bash
python scripts/revision/generate_water_cluster_geoms.py \
  --output-dir datasets/revision/water_clusters
```

**DFT (PBE+D4 / aug-cc-pVDZ + aug-cc-pVQZ-JKFIT)**

```bash
for split in train val id_test ood_size; do
  python scripts/revision/generate_dft_labels.py \
    --trajectory datasets/revision/water_clusters/${split}.xyz \
    --output-dir datasets/revision/water_clusters \
    --output-prefix water_${split} \
    --xc pbe --d4 --df
done
```

**Train**

```bash
python run.py train @config/training/water_clusters_001.txt
```

**Evaluate H-bonds on n = 8–12**

```bash
python scripts/revision/evaluate_hbond_metrics.py \
  --ref datasets/revision/water_clusters/ood_size.xyz \
  --pred-traj path/to/densnet_ood.npy \
  --out results/revision/water_hbond_metrics.json
```

**Metrics:** density MAE and relative MAE ∫|Δρ|/∫ρ; energy/force MAE; dipole MAE; O···O and O–H histograms.

**Data paths:** `datasets/revision/water_clusters/`

---

## Calculation 2 — OOD conformations

**Why:** R1.2. Evaluate published models first; do not retrain.

```bash
python scripts/revision/generate_ood_geometries.py \
  --input datasets/ethanol_train_10.xyz \
  --molecule ethanol \
  --output-dir datasets/revision/ood
```

Then DFT-label and score an existing checkpoint:

```bash
python scripts/revision/generate_dft_labels.py \
  --trajectory datasets/revision/ood/ethanol_ood.xyz \
  --output-dir datasets/revision/ood --output-prefix ethanol_ood \
  --xc pbe --d4 --df

python scripts/training/eval_model_npy.py \
  --model paper/models/ethanol/2024-03-22_96w7KyGG \
  --dataset datasets/revision/ood/ethanol_ood_npy.npy
```

Repeat for ethanethiol when a source XYZ/NPY is restored.

**Report:** error vs SOAP / pairwise-distance to the training set (`analyze_train_test_overlap.py --ood`).

---

## Calculation 3 — Hybrid DFT (PBE0)

**Why:** R3.1. Proof of principle only (~400 frames). No full-paper recompute.

```bash
python scripts/revision/generate_dft_labels.py \
  --trajectory datasets/ethanol_train_10.xyz \
  --output-dir datasets/revision/pbe0 \
  --output-prefix ethanol_pbe0 \
  --xc pbe0 --d4 --df

python scripts/revision/build_sad_prior.py \
  --xc pbe0 --elements H,C,O \
  --output datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_pbe0.npy
```

Train: `python run.py train @config/training/ethanol_pbe0_001.txt`  
ωB97M-V is optional stretch after PBE0 works.

---

## Calculation 4 — MLFF / semiempirical comparison

**Why:** R2.2, R3.2. Compare on *our* molecules, not Crambin.

```bash
python scripts/revision/benchmark_figure2.py \
  --molecules ethanol,ethanethiol,resorcinol,thiophene2 \
  --methods densnet,maceoff,aimnet2,gfn2xtb,gxtb,aimd \
  --out results/revision/figure2_timing.json
```

IR: existing `scripts/md/mace_off_md_run.py`, `run.py gxtb-md`, `scripts/analysis/compute_ir_maceoff_vs_mldft.py`.  
**Decline:** DenSNet training on Crambin-in-water, paracetamol, FGG (foundation-MLIP benchmarks).

---

## Calculation 5 — IR length

**Why:** R3.6, R2.9.

```bash
python run.py md @config/md/nn/ethanol_500ps.txt
python run.py md @config/md/nn/polythiophene_2mer_500ps.txt
python scripts/revision/ir_vs_length.py \
  --hdf5 path/to/md.h5 \
  --windows 50,100,200,500 \
  --out results/revision/ir_convergence
```

AIMD max time for the main text: ethanol **10 ps** (`results/aimd_benchmark/aimd_ethanol_rep0_10ps.traj`). Confirm polythiophene AIMD length from original notes when recovered.

---

## Calculation 6 — Equilibria and NVE

**Why:** R2.11, R2.4.

```bash
python scripts/revision/optimize_densnet.py \
  --model paper/models/ethanol/2024-03-22_96w7KyGG \
  --structure datasets/ethanol_train_10.xyz \
  --dft-xc pbe --out results/revision/geoopt_ethanol.json

python scripts/revision/nve_energy_drift.py \
  --hdf5 path/to/nve.h5 \
  --out results/revision/nve_ethanol.json
```

Paper MD configs already use `--langevin=False`.

---

## Calculation 7 — Water OOD ladder and hybrid labels

**Why:** R1.1, R1.2, R2.3, R3.1 at once. One set of geometries answers the
out-of-distribution request, the hydrogen-bonded-system request and the
above-PBE request, which is what makes it affordable.

### Level of theory

ωB97M-V/def2-TZVPD is the headline, because Reviewer 3 named it and it is the
level OMol25/CSH already uses, so both hybrid exhibits share one reference.
PBE-D4/aug-cc-pVDZ runs on the *same* geometries, so a PBE-versus-hybrid table
isolates the reference data from the architecture. Auxiliary basis stays
aug-cc-pVQZ-JKFIT so the existing orbital/radial tables work untouched.

Measured on the water dimer (`results/revision/calibration/calib_dimer_coulomb.json`):

| Comparison | Δ energy (Ha) | fit-coefficient L2 | Coulomb-metric density |
| --- | --- | --- | --- |
| PBE-D4, ORCA vs PySCF | 3.3e-05 | 9.3e-06 | 7.1e-07 |
| ωB97M-V, RIJCOSX vs exact integrals | 1.8e-04 | 1.5e-04 | 2.8e-05 |
| ωB97M-V, exact-integral ORCA vs PySCF | 2.3e-06 | 8.0e-06 | 3.8e-07 |

The coefficient norm and the Coulomb metric disagree by a factor of five because
aug-cc-pVQZ-JKFIT is over-complete: coefficients can move along near-null
directions without moving the density. The Coulomb metric — the self-energy of
the density difference — is what the paper reports and what the gate uses.

Two consequences. PBE labels are engine-independent, so ORCA and PySCF frames
can share a split. ωB97M-V is pinned to ORCA alone: RIJCOSX is the only
approximation that reaches the 24-water clusters, PySCF has no COSX to match it,
and RIJK's analytic gradient aborts in ORCA 6.1.1. Since this is the
size-scaling experiment, a size-correlated switch between exact and approximate
integrals would contaminate the very trend being measured.

**The approximation error does not grow with cluster size**, which is what
licenses the size-scaling claim and is the first thing a reviewer will ask.
Relative L2 over fit coefficients (`calib_n3_n6.json`):

| n water | RIJCOSX vs exact integrals | exact-integral ORCA vs PySCF |
| --- | --- | --- |
| 2 | 1.45e-04 | 8.0e-06 |
| 3 | 1.80e-04 | 2.2e-06 |
| 6 | 1.74e-04 | 3.8e-06 |

Both are flat while the system triples. The RIJCOSX term is a fixed ~1.7e-04
offset shared by every frame, not a size-dependent bias that could manufacture
the trend being measured; and the exact-integral agreement staying at a few
parts in 10⁶ across sizes confirms the AO mapping holds as the basis grows.
Max ‖CᵀSC − I‖ over all calibration frames is 9.0e-08.

PBE's ORCA-vs-PySCF energy difference (4.3e-05 Ha at n=3, 8.7e-05 at n=6) sits
above the 1e-05 gate, but it is entirely the D4 dispersion term: the densities
agree to 7.3e-06 and 9.7e-06 respectively. The gate reports it, and it does not
affect the density labels.

### The OOD ladder

| Tier | Content | Frames | Distance from training |
| --- | --- | --- | --- |
| 0, ID | n = 2–6 thermal noise around motif minima | 1250 / 250 / 250 | 0.01 |
| 1, size | n = 8, 10, 12 | 300 | 0.36 |
| 2, order | ice Ih, XI, Ic, II cutouts, n = 12–24, 50 K and 150 K | 192 | 1.90 |
| 3, density | droplets melted from ice cutouts under a spherical wall, n = 16–24 | 120 | 2.79 |
| 4, reaction coordinate | malonaldehyde proton transfer | 400 / 80 / 125 | see below |

Distance is the nearest training frame in standardised collective-variable
space, in units of the training set's own width (its median pairwise distance).
Reported in *width* rather than nearest-neighbour-spacing units because with
1250 frames drawn from a handful of motifs the spacing is ~45x finer than the
extent, which would put the ice tier at 294 — arithmetically right and useless
to a reader.

Tier 1 is the weakest and is reported as such: it is precisely the extrapolation
Reviewers 1 and 2 called insufficient. Tiers 2 and 3 are where the separation
becomes unarguable: no ice or droplet frame lies within the training set's own
spacing, and 95% of ice frames exceed the training 99th percentile in
tetrahedral order.

Tier 4 supplies the one axis that is a real reaction coordinate rather than a
descriptor: δ = r(O_donor–H) − r(H–O_acceptor). Training spans δ ∈ [−1.43,
−0.42] Å, one enol basin only; the relaxed scan runs to +0.68 Å, through the
transition state into the other basin. 78% of scan frames lie beyond the
training maximum, and that portion overlaps the training distribution at 0.085.
The whole-scan overlap of 0.44 is not a failure — the scan begins inside the
training basin deliberately, so the error curve has an in-distribution anchor.

### Running it

```bash
# geometries
python scripts/revision/generate_ood_water.py --output-dir datasets/revision/water_ood
python scripts/revision/generate_malonaldehyde.py --output-dir datasets/revision/malonaldehyde

# shards and first submission, per level of theory
scripts/revision/launch_water_campaign.sh wb97mv_def2tzvpd
scripts/revision/launch_water_campaign.sh pbe_d4_avdz

# one watchdog for the whole campaign: re-submits only outstanding shards,
# recovers claims whose Slurm task is gone, and launches training when a level
# of theory is complete
scripts/revision/watch_water_campaign.sh

# progress at any time
python scripts/revision/campaign_status.py --root datasets/revision/shards

# everything downstream; safe to re-run, stops where the inputs run out.
# skips a split evaluated within the last 6 h; FORCE_EVAL=1 for a final pass
scripts/revision/run_ood_analysis.sh wb97mv_def2tzvpd
```

### What actually limits the throughput

The binding constraint is memory, not cores. The `cpu48` QOS allows
`cpu=3000,mem=6000G`, and at the original flat `--mem=48G` per 8-core task the
account hit the memory cap at 125 tasks — 1000 cores of the 3000 it was entitled
to. The calibration measures ≤6.8 GB for the largest n=6 frame at ωB97M-V and
far less for malonaldehyde, so memory is now requested per split (12 G for
malonaldehyde, 16 G for n=2–6, 32 G for n=8–10, 96 G for the large tier). ORCA's
`%maxcore` is derived from `SLURM_MEM_PER_NODE` rather than fixed, because it is
a *per-process* ceiling: a flat 3000 MB across 8 ranks promises ORCA 24 GB, which
was consistent with the old 48 G request and would OOM inside the new 16 G one.

Two things were costing more than any of that, though, and neither was visible
in a per-job log:

- **Partition pinning.** `--partition=cs` looked harmless and was not. The site
  submit filter chooses the partition list from the walltime, and under 6 h it
  returns `cs,cpu_short`; naming `cs` discards the alternative. With this
  account's fair-share factor of 0.006 that is decisive — a matched pair of probe
  jobs had the unpinned one running within a second while the pinned one was
  still pending forty minutes later.
- **Submission order.** Under a memory cap, whichever split is offered first gets
  the memory, since Slurm breaks ties by age. Walking the shard tree
  alphabetically put `water_train_small` ninth of ten. The result was 0 of its 65
  ωB97M-V shards labelled while 75 tasks of the tier the plan itself calls the
  weakest held 3600 GB — and no model at the end of it, which makes every OOD
  number unobtainable, since each is the error of a model that needs that split
  to exist. `split_rank` in the watchdog now orders them: train and validation,
  then malonaldehyde and its scan, then the ID control, then the OOD tiers
  cheapest first. Re-ranking took `water_train_small` from 0 to 25 of 32 done at
  PBE-D4 within ninety minutes.

**Deliverable:** `results/revision/figures/error_vs_distance_*.png` — relative
density error against distance from training, one series per tier, beside the
same error against the malonaldehyde proton-transfer coordinate.
`run_ood_analysis.sh` now runs on every watchdog pass, so the figure appears as
soon as the first model has a checkpoint and refreshes as the rest arrive.

### What makes the figure trustworthy

The whole chain — labels, training, checkpoint, evaluation, figure — was run end
to end before the campaign's own labels were ready, on the 1110 PySCF PBE frames
the retired serial loop had already produced. That was worth doing on its own:
the chain had never executed, and it does not execute until days into a campaign,
unattended, with any traceback going to a watchdog log. It found a fault that
nothing else could have (see the last entry below).

The deliverable also joins two things computed in different places — a
per-structure density error and a per-structure descriptor — and the ways that
join can go wrong are all silent. Four faults were found and closed:

- **Frame identity.** `csh_evaluate.py` reported a structure's row in the
  assembled dataset, while the collective variables are numbered over the source
  geometry file. Assembly drops frames ORCA failed on (deliberately:
  `--min-complete` tolerates up to 40% of a size bucket missing), so every error
  after the first gap belonged to a different structure than its position
  implied. Frames now carry `source_index` from the assembler through to the
  evaluation report, and the join uses it. Nothing about the old behaviour
  failed; the points were simply in the wrong places along the x-axis.
- **Restart.** Training writes into a timestamped directory *inside*
  `--save_dir`, so the restart check found no checkpoint and every 24 h window began
  again at step zero. At 300k steps a run does not fit one window, so the
  campaign would have re-trained from scratch indefinitely. Both the restart and
  the analysis now resolve the run directory via `latest_run_dir.sh`.
- **Distance units.** Reported in training-set widths (median pairwise distance)
  rather than nearest-neighbour spacings; see the ladder table above.
- **Evaluating a model that predicts forces.** `csh_evaluate.py` ran the forward
  pass under `torch.no_grad()`. A model with an energy head gets its forces by
  differentiating the energy with respect to the positions, so under `no_grad`
  it raises `element 0 of tensors does not require grad` and returns no density
  at all. Every CSH model is density-only, which is why this had never shown;
  every model in this campaign has the head. This one is not subtle — it is a
  hard failure — but it would have surfaced only once the labels finished. Both
  it and the obvious wrong fix (pre-marking the positions, which makes
  `DFTNetwork`'s internal clone a non-leaf and fails on the way back out) are
  pinned in `tests/test_evaluate_forward.py` against a stub, so the check costs
  milliseconds and needs no checkpoint.

The malonaldehyde tier deliberately does *not* clear the `BC < 0.05` bar set for
tiers 2–4 (whole-scan overlap 0.43). The scan starts inside the training basin
on purpose so the error curve has an in-distribution anchor, and the separation
lives in the tail: 78% of scan frames beyond the training maximum in δ,
overlapping at 0.085.

---

## Analysis-only

```bash
# R1.5 overlap (needs restored train/test NPYs)
python scripts/revision/analyze_train_test_overlap.py \
  --train datasets/ethanol_dft_train.npy \
  --test datasets/ethanol_dft_test.npy \
  --out results/revision/overlap_ethanol.json

# R2.8, R3.3, R3.4
python scripts/revision/analyze_density_metrics.py \
  --dens-ref datasets/h2o_small_test_augccpvdz_df_augccpvqzjkfit.npy \
  --atom-dens datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy \
  --out results/revision/density_metrics_h2o.json
```

Cutoff sweep: `config/training/water_clusters_cutoff_{4,5,6,8}.txt`.

**Classical MD (R1.5) — still unknown:** force field, duration, timestep, H constraints. Fill below when notes are found:

| Molecule | FF | Length | dt | H constraints | Thermostat |
| --- | --- | --- | --- | --- | --- |
| Ethanethiol | _unknown_ | _unknown_ | _unknown_ | _unknown_ | 300 K |
| Resorcinol | _unknown_ | _unknown_ | _unknown_ | _unknown_ | 300 K |
| Polythiophene train MD | _unknown_ | _unknown_ | _unknown_ | _unknown_ | _unknown_ |

---

## Calculation log

Record each production job here.

| Date | System | Level | nframes | Script | Output | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-08-17 | water n=2–12 | geometries only (analytic minima + thermal noise) | see manifest | `scripts/revision/generate_water_cluster_geoms.py` | `datasets/revision/water_clusters/` | re-thermalize with g-xTB/TIP3P before production DFT if desired |
| 2026-08-17 | ethanol OOD | geometries only (strain + high-T noise) | see manifest | `scripts/revision/generate_ood_geometries.py` | `datasets/revision/ood/` | 10 parent frames from `ethanol_train_10.xyz` |
| 2026-08-17 | water + ethanol OOD | g-xTB single points | 2240 (1 ethanol fail) | `run_gxtb_labels.py` (fixed leftover-energy reuse) | `results/revision/gxtb/*.jsonl` | water train mean −305.69 Eh (n=2–6 mix) |
| 2026-08-17 | ethanol 10-parent vs OOD | overlap | 10 / 190 | `analyze_train_test_overlap.py` | `results/revision/overlap_ethanol_ood.json` | median min-RMSD 0.202 Å; 94/190 below 0.2 Å |
| 2026-08-17 | water train vs ID test | overlap | 1250 / 250 | `analyze_train_test_overlap.py` | `results/revision/overlap_water_id.json` | all 250 ID frames <0.2 Å — same-minima thermal noise |
| 2026-08-17 | water H-bond hists | geometry only | train/id/ood | `evaluate_hbond_metrics.py` | `results/revision/water_hbond_*.json` | OOD n=8–12 mean O···O 3.19 Å vs train 2.96 Å |
| 2026-08-17 | H,C,N,O,S | PBE and PBE0 SAD | 5 atoms | `build_sad_prior.py` | `datasets/revision/sad_{pbe,pbe0}_augccpvdz.npy` | rebuilt for current SciPy pickle |
| 2026-08-17 | water dimer | PBE+D4+DF smoke | 1 | `generate_dft_labels.py` | `datasets/revision/water_clusters/water_dimer_smoke_*` | E=−152.661674 Eh; ~32 s CPU |
| 2026-08-17 | water train + ethanol OOD + PBE0 subset | PBE/PBE0+D4+DF | in progress | `run_dft_campaign.sh` | `datasets/revision/**/` | tmux `dft-pbe-train`, `dft-pbe-rest`, `dft-pbe0`; `--no-gpu` |
| 2026-08-17 | ethanol | CPU Figure 2 timings | 1 geom | `benchmark_figure2.py` | `results/revision/figure2_timing.json` | g-xTB 0.70 s; GFN2 3.05 s; MACE-OFF 4.48 s; AIMNet2 failed (triton/Python.h) |
| 2026-08-17 | water dimer | PBE BFGS geo-opt | 1 | `optimize_densnet.py` | `results/revision/geoopt_water_dimer_dft.json` | −4155.66 eV; DenSNet skipped |
| 2026-08-17 | resorcinol / ethanethiol / thiophene | train–test overlap | 1000–5000 | `analyze_train_test_overlap.py` | `results/revision/overlap_{resorcinol,ethanethiol,thiophene*}.json` | resorcinol all test frames <0.2 Å; ethanethiol/thiophene tests are farther |
| 2026-08-17 | ethanol | CPU DenSNet timing + 20 fs NVE | 1 geom | `benchmark_figure2.py`, `cpu_mlip_suite.py` | `figure2_timing.json`, `mlip_cpu_densnet/` | 1.62 s/step; NVE drift 0.30 eV/ps; energy head mismatch vs 2024 ckpt |
| 2026-08-17 | ethanol+water PBE0 subset | PBE0+D4+DF | 70/70 | `run_dft_campaign.sh pbe0` | `datasets/revision/pbe0/` | labels complete |
| 2026-08-18 | ethanol PBE0 train | DenSNet | 70 | `run_gpu_campaign.sh pbe0` | `results/revision/ethanol_pbe0_001/` | local GPU 0 after quick tests; also `sbatch` `dens-pbe0` |
| 2026-08-18 | ethanol 500 ps MD | DenSNet NVT/NVE | 1e6 steps | `run_gpu_campaign.sh ethanol-md` | MD logs | local GPU 1; energy-head caveat |
| 2026-08-18 | thiophene 2-mer 500 ps | DenSNet | queued | Slurm `dens-thiophene-md` 15934890 | l40s_public | Priority behind current 2-GPU wrap |
| 2026-08-18 | ethanol | GPU Figure 2 + 0.1 ps NVE | 1 geom | `run_gpu_campaign.sh quick` | `figure2_timing_gpu.json`, `nve_drift_gpu.json` | DenSNet 467 ms/step; NVE drift 0.18 eV/ps |
| 2026-08-19 | water dimer | ORCA vs PySCF, PBE-D4 and ωB97M-V | 1 | `calibrate_theory.py` | `results/revision/calibration/calib_dimer_coulomb.json` | PBE engine-independent (7.1e-07 Coulomb metric); RIJCOSX costs 2.8e-05; AO permutation good to 6.5e-07 |
| 2026-08-19 | water dimer | PySCF grid convergence, PBE | 1 | `probe_grid_sensitivity.py` | `results/revision/calibration/grid_sensitivity.json` | level 3 → 9 moves the density by 2.8e-07, so the grid is not a cross-code error source |
| 2026-08-19 | water n=3, n=6, n=8 | ORCA vs PySCF, both levels, size dependence | 3 | `calibrate_theory.py` | `results/revision/calibration/calib_n3_n6.json` | RIJCOSX-vs-exact is flat at 1.5–1.8e-04 from n=2 to n=6 and exact-vs-PySCF flat at a few e-06, so neither approximation can manufacture a size trend; max ‖CᵀSC−I‖ 9.0e-08. n=8 ωB97M-V 1186 s on 8 cores |
| 2026-08-19 | ice Ih/XI/Ic/II cutouts | geometries (genice2 + rigid-body thermal displacement) | 192 | `generate_ood_water.py` | `datasets/revision/water_ood/ood_order.xyz` | n = 12–24, 50 K and 150 K |
| 2026-08-19 | water droplets | geometries (ice cutouts melted under a spherical wall, GFN2-xTB) | 120 | `generate_ood_water.py` | `datasets/revision/water_ood/ood_density.xyz` | n = 16–24; packing a droplet directly gave a gas, melting an ice cutout gives an H-bond network |
| 2026-08-19 | malonaldehyde | geometries (GFN2-xTB basin sampling + relaxed proton-transfer scan) | 400 / 80 / 125 | `generate_malonaldehyde.py` | `datasets/revision/malonaldehyde/` | training filtered to one enol basin, else the scan would not be OOD |
| 2026-08-19 | all water + malonaldehyde splits | train/test separation in CV space | 2237 | `water_collective_variables.py`, `malonaldehyde_collective_variables.py`, `ood_overlap_report.py` | `results/revision/ood_overlap*.json` | ladder is monotone: ID 0.01, size 0.36, ice 1.90, droplets 2.79 training widths |
| 2026-08-19 | water n=2–24 + malonaldehyde | ωB97M-V/def2-TZVPD and PBE-D4/aug-cc-pVDZ, EnGrad + DF | 513 shards | `launch_water_campaign.sh`, `watch_water_campaign.sh` | `results/revision/water_orca/` | 20 arrays; small tier 8 cores/4 h, large tier 16 cores/24 h; serial PySCF loop retired |
| 2026-08-19 | — | scheduling: stopped pinning the arrays to `cs` | — | `submit_water_orca*.sbatch` | — | the whole campaign sat on `Priority` (fair-share 0.006). The site filter derives a partition list from the walltime, and under 6 h returns `cs,cpu_short`; two identical probe jobs differing only in the pin had the unpinned one running in 1 s and the pinned one still queued 40 min later. 0 → 123 tasks running |
| 2026-08-19 | — | claim protocol: a requeued task may retake its own shard | — | `shard_claim.py` | — | a preempted task is killed outright, so its claim survives; the restarted incarnation read it as another worker's and exited, and the scheduler still listed the id as live, so nothing else took the shard either |
| 2026-08-19 | — | join correctness for the headline figure | — | `qm7x_orca_common.py`, `csh_evaluate.py`, `error_vs_distance.py` | — | errors were keyed to dataset row, descriptors to source frame; any ORCA failure shifted every later point along the x-axis. `source_index` now travels from assembler to report |
| 2026-08-19 | — | training restart and analysis paths | — | `latest_run_dir.sh`, `submit_water_train.sbatch`, `run_ood_analysis.sh` | — | run.py writes into a timestamped dir inside `--save_dir`; the restart check and the analysis both looked one level too high, so training restarted from step 0 each window and the error curves would never have been produced |
| 2026-08-19 | water n=2–6 (PBE/aug-cc-pVDZ, the retired serial loop's 1110 frames) | end-to-end rehearsal of the whole downstream chain | 1110 train / 250 test / 45 size-OOD | `run.py`, `csh_evaluate.py`, `error_vs_distance.py` | `results/revision/smoke_water_pbe/`, `figures/error_vs_distance_pilot_pbe_pyscf.png` | 30-step CPU model, so the numbers mean nothing; the point was that the chain had never run. Found the `no_grad` fault below, and confirmed `latest_run_dir.sh` skips the two launches that died before their first checkpoint |
| 2026-08-19 | — | evaluation of models with an energy head | — | `csh_evaluate.py`, `tests/test_evaluate_forward.py` | — | the forward pass ran under `torch.no_grad()`, under which a force head raises before returning any density. CSH models are density-only so this had never shown; every model here has the head. Would have failed days in, in a watchdog log |
| 2026-08-19 | — | per-run peak RSS in the calibration | — | `calibrate_theory.py` | `results/revision/calibration/` | `getrusage`'s `ru_maxrss` is a high-water mark that only rises, so each record reported the largest run so far — hence ORCA and PySCF coming back byte-identical at every size. It also misses ORCA entirely, which is a subprocess. Now sampled from `/proc` over the process tree |
| 2026-08-19 | — | submission order and per-split memory | — | `watch_water_campaign.sh`, `submit_water_orca*.sbatch` | — | memory, not cores, caps the account (6000 GB); alphabetical order put `water_train_small` ninth of ten and it sat at 0/65 while the weakest OOD tier held 3600 GB. Ranked explicitly; `%maxcore` now derived from the allocation. 0 → 25/32 done at PBE-D4 in 90 min |

---

## Suggested order

1. Restore paper densities/models from the original machine.
2. Submit water-cluster DFT (long pole), then train.
3. OOD + overlap analysis in parallel.
4. PBE0 ethanol subset + PBE0 SAD.
5. Figure 2 / MLFF IR timings (existing runners).
6. 200–500 ps ML-MD, geo-opt, NVE.
7. Density-metric / cutoff / uncertainty jobs.
8. Manuscript + annotated revision + response letter.
