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
| R1.1 | H-bond / size extrapolation (water clusters; NMA/MeOH/AcAc optional) | `new DFT` + train | `running` | [Calculation 1](#calculation-1--water-clusters); CPU PBE+D4 DFT in tmux `dft-pbe-train` |
| R1.2 | Intentional OOD conformations | `new DFT` | `running` | [Calculation 2](#calculation-2--ood-conformations); ethanol OOD DFT in `dft-pbe-rest`; overlap vs 10 parents computed |
| R1.3 | Shorten polythiophene; move Fig. 6 | `manuscript` | `written` | outline §R1.3 |
| R1.4 | Move Fig. 7 to SI | `manuscript` | `written` | outline §R1.4 |
| R1.5 | SI S2.2 details + train/test overlap evidence | `analysis` + `recover` | `computed` | resorcinol median min-RMSD 0.093 Å (1000/1000 <0.2 Å); ethanethiol 0.600 Å (0/1000 <0.2 Å); thiophene-4mer 1.15 Å; thiophene-6mer 0.53 Å; 12-mer is size-OOD. SI already has PINY_MD / NHC / 1 fs; FF/SHAKE/length not recovered |
| R1.6 | Deduplicate 755 ms vs 5 min | `manuscript` | `computed` | CPU Figure 2 ethanol: DenSNet 1.62 s, g-xTB 0.70 s, GFN2-xTB 3.05 s, MACE-OFF 4.48 s; AIMNet2 still broken |
| R1.7 | Retract unsupported dielectric / H-bond-network claims | `manuscript` | `written` | retracted in `scratch/recovered_manuscript/main/nat_manuscript.tex` Discussion |

### Reviewer 2

| ID | Request | Action | Status | Deliverable |
| --- | --- | --- | --- | --- |
| R2.1 | Cite Hazra–Sanvito JCP 2025 density→IR workflow | `manuscript` | `written` | outline citations |
| R2.2 | Compare general IR methods (MACE4IR, AIMNet2, TranSpec, AIQM) | `new MD` + `manuscript` | `running` | CPU timings include DenSNet 1.62 s/step (~0.027 ns/day); 200–500 ps IR still needs a free GPU |
| R2.3 | Tougher extrapolation than hexamer-trained oligomers | `manuscript` | `written` | water-cluster size hold-out is the new experiment |
| R2.4 | Companion network: embedding, autodiff, energy conservation | `new MD` + `manuscript` | `computed` | 20 fs CPU NVE: energy std 5.4 meV, drift 0.30 eV/ps (architecture mismatch caveat); production NVE waits for GPU + matching energy head |
| R2.5 | Who produces dipoles vs energy/forces | `manuscript` | `written` | outline §R2.5 |
| R2.6 | One framework vs two networks at MD time | `manuscript` | `written` | outline §R2.6 |
| R2.7 | Electron-count normalization vs delta-learning | `manuscript` | `written` | outline §R2.7 |
| R2.8 | Softplus vs signed delta-density | `manuscript` + `analysis` | `computed` | h2o_small_test 46.7% negative DF coeffs; ethanethiol combo 51.0%; SAD e-counts H/C/N/O/S = 1/6/7/8/16 |
| R2.9 | State max AIMD time in main text | `manuscript` | `written` | recovered: ethanol AIMD **10 ps** on disk |
| R2.10 | Units for density errors | `manuscript` | `written` | use **e/a0³** and relative MAE |
| R2.11 | Numerical evidence DenSNet ≈ DFT equilibria | `new MD` | `running` | Water-dimer PBE BFGS −4155.66 eV; ethanol DenSNet BFGS RMSD 0.0 Å vs start (already below fmax=0.05); energy ~0.24 eV is **not** physical — current energy head has extra untrained layers vs 2024 `96w7KyGG` |
| R2.12 | Cite Cuevas-Zuviría / Pacios JCIM 2020 | `manuscript` | `written` | outline citations |
| R2.13 | Corresponding authors MS vs SI | `manuscript` | `written` | outline §R2.13 |

### Reviewer 3

| ID | Request | Action | Status | Deliverable |
| --- | --- | --- | --- | --- |
| R3.1 | Hybrid DFT densities (PBE0+MBD / ωB97M-V) | `new DFT` | `computed` | PBE0+D4+DF 70/70 frames written (`ethanol_water_pbe0_pyscf_augccpvdz_pbe0.npy`); train waits for a free GPU |
| R3.2 | Compare SO3LR, MACE-OFF, GFN-xTB, DFTB; expand Fig. 2 | `new MD` | `running` | CPU Figure 2 now includes DenSNet 1.62 s; AIMNet2/`Python.h`; SO3LR/DFTB not installed; no GPU IR |
| R3.3 | Relative density errors + uncertainty | `analysis` | `computed` | DF-coeff / SAD diagnostics; 3-seed DenSNet ensemble still needs training |
| R3.4 | Direct vs delta cost and correction magnitude | `analysis` | `scripts_ready` | SAD timing in `benchmark_figure2.py`; Δρ histograms in `analyze_density_metrics.py` |
| R3.5 | Hyperparameter / cutoff optimization | `analysis` | `scripts_ready` | `config/training/water_clusters_cutoff_*.txt` |
| R3.6 | IR convergence vs trajectory length | `new MD` | `scripts_ready` | [Calculation 5](#calculation-5--ir-length) |
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
| 2026-08-17 | ethanol+water PBE0 subset | PBE0+D4+DF | 70/70 | `run_dft_campaign.sh pbe0` | `datasets/revision/pbe0/` | labels complete; train not started on gl056 |

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
