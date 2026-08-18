# Response-letter outline — ja-2026-12808m

Use this as the skeleton of the file uploaded with the resubmission. Fill quantitative results when calculation-log rows in [jacs_revision_todo.md](jacs_revision_todo.md) move to `computed`.

---

## Letter to the Editor

Thank Professor Coley. State that this is a resubmission addressing the enclosed reviews in full, with new hydrogen-bond size-extrapolation data, OOD tests, a hybrid-DFT demonstration, MLFF/semiempirical IR and timing comparisons on the original molecules, IR-length convergence, and geometry-optimization / NVE checks. Note the annotated manuscript and this point-by-point response.

---

## Reviewer 1

**R1.1 Hydrogen bonding / size extrapolation.**  
New DenSNet trained on (H2O)_2–6, tested on (H2O)_8,10,12. Report density/energy/force/dipole MAE and H-bond histograms. NMA, methanol–water, and AcAc are left as future work; water clusters are the protocol the reviewer named.

**R1.2 OOD conformations.**  
High-T and affinely strained ethanol (and ethanethiol when source frames are restored). Errors plotted against descriptor distance to the training set. Poor OOD numbers will be reported as the domain of applicability.

**R1.3 / R1.4 Figures 6 and 7.**  
Move Fig. 6 (or most of it) and Fig. 7 to the SI. Combine Fig. 5a with Fig. 4 if it keeps the main text focused. Keep one compact polythiophene IR comparison in the main text.

**R1.5 Dataset documentation.**  
SI already records PINY_MD, massive NHC (length 4, τ = 20 fs), 1 fs timestep, 300/500 K, and k-means train selection. Force field, SHAKE/H constraints, and trajectory length were not recovered. Overlap (Hungarian RMSD to nearest train frame):

- Resorcinol k-means train vs test: median min-RMSD **0.093 Å**; **1000/1000** frames < 0.2 Å (same basin + thermal noise; “separate MD ⇒ no overlap” is not supported).
- Ethanethiol combo vs rand: median **0.600 Å**; **0/1000** < 0.2 Å (a real hold-out).
- Thiophene 4-mer / 6-mer tests vs mixed train: median **1.15 Å** / **0.53 Å**; 12-mer is size-OOD (no same-`n_atoms` train group).
- Ethanol paper-split `ethanol_dft_*.npy` still missing; ethanol train is MD17 per SI.

**R1.6 Timing duplication.**  
One regenerated CPU table (`figure2_timing.json`, ethanol): DenSNet **1.62 s/step**, g-xTB **0.70 s**, GFN2-xTB **3.05 s**, MACE-OFF **4.48 s**. AIMNet2 compile still fails. State this once.

**R1.7 Speculative dielectric claims.**  
Retracted in the recovered `nat_manuscript.tex` Discussion. Water-cluster size-OOD (Wave C) may restore a limited H-bond statement if the data support it.

---

## Reviewer 2

**R2.1 Hazra, Patil, Sanvito, *J. Chem. Phys.* 2025, 163, 174104.** DOI [10.1063/5.0292221](https://doi.org/10.1063/5.0292221).  
Jacobi–Legendre cluster expansion of the DFT density, then energy/forces/dipole/gap; demonstrated on gas-phase uracil IR. Distinguish: different density representation (real-space cluster expansion vs equivariant network + DF), no oligomer size extrapolation, no SAD delta-learning.

**R2.2 General IR methods.**  
- Pracht et al., *J. Chem. Theory Comput.* 2024. DOI [10.1021/acs.jctc.4c01157](https://doi.org/10.1021/acs.jctc.4c01157) — MACE-OFF23 + double-harmonic / composite dipoles.  
- TranSpec, *J. Am. Chem. Soc.* 2025. DOI [10.1021/jacs.5c05010](https://doi.org/10.1021/jacs.5c05010) — spectrum ↔ SMILES translation, not AIMD-quality dipoles.  
- AIQM chemRxiv [10.26434/chemrxiv-2024-604wb](https://doi.org/10.26434/chemrxiv-2024-604wb).  
- MACE4IRmol, AIMNet2 (already cited).  

We compare transferable MLIPs and xTB **on our molecules** (Calculation 4). CPU ethanol single-points: DenSNet 1.62 s, g-xTB 0.70 s, GFN2-xTB 3.05 s, MACE-OFF 4.48 s (~0.01 ns/day). DenSNet is system-specific; the return is a DFT-quality density and analytic dipoles from one trained model. 200–500 ps IR bake-off waits for a free GPU.

**R2.3 Extrapolation.**  
Acknowledge hexamer-trained oligomer spectra. The new water-cluster hold-out is a harder H-bond size test. Soften polymer-limit language where errors grow.

**R2.4 Companion network, autodiff, energy conservation.**  
The energy/force head consumes the frozen (or jointly trained) equivariant density features. Forces are −∇_R E through automatic differentiation of the energy head, including the coordinate dependence of the density features. A 20 fs CPU NVE on ethanol (`mlip_cpu_densnet/`) has energy std 5.4 meV and a raw drift of 0.30 eV/ps; this is **not** a production conservation number because the current energy head has extra untrained layers versus the 2024 `96w7KyGG` checkpoint. Production NVE waits for a matching code/checkpoint pair on a free GPU.

**R2.5 Dipoles vs “electronic observables”.**  
Dipoles are computed from the **density network** (analytic `int1e_r` / `--dpm-intor`), not from the energy companion head. The companion head outputs energy and forces. Page 4 wording will be corrected.

**R2.6 One framework, two networks.**  
Both networks run at every MD step: density features → energy/forces (and optional dipole). No post-processing trajectory is required for IR if `--dpm_intor` is on.

**R2.7 Normalization vs delta-learning.**  
Electron-count normalization weights the loss by the integrated density so core and valence errors are not on incommensurate raw-grid scales. It does **not** replace a physically motivated prior. SAD delta-learning changes the *target* (ρ − ρ_SAD). Both are kept; they do different jobs.

**R2.8 Softplus and signed Δρ.**  
SAD corrections are signed. Softplus is applied so the **reconstructed total density** ρ_SAD + Δρ_pred stays non-negative, not so that Δρ itself is positive. DF-coefficient diagnostics: **46.7%** negative on h2o_small_test and **51.0%** on ethanethiol combo (50 frames). SAD occupancies H/C/N/O/S = 1/6/7/8/16.

**R2.9 AIMD duration.**  
Ethanol AIMD on disk: **10 ps** (`results/aimd_benchmark/aimd_ethanol_rep0_10ps.traj`). State this in the main text. Add other AIMD lengths when recovered.

**R2.10 Density-error units.**  
Report MAE in **e/a0³** (and e/Å³ in parentheses if useful) plus relative MAE ∫|Δρ| dV / N_elec.

**R2.11 Equilibria.**  
Water-dimer PBE BFGS: −4155.66 eV. Ethanol DenSNet BFGS from `ethanol_train_10.xyz` did not move (RMSD 0.0 Å vs start; already below fmax = 0.05 eV/Å). The reported DenSNet energy (~0.24 eV) is **not** a DFT-comparable total energy: the current energy head has layers absent from `96w7KyGG`. A production DenSNet↔DFT RMSD requires a matching energy architecture.

**R2.12 Cuevas-Zuviría and Pacios, *J. Chem. Inf. Model.* 2020, 60, 3831–3842.** DOI [10.1021/acs.jcim.0c00197](https://doi.org/10.1021/acs.jcim.0c00197) (A2MD / A2MDnet analytical density model). Add to the introduction.

**R2.13 Corresponding authors.**  
Align the title page and SI. Confirm with Tuckerman/Burke/Müller before resubmission.

---

## Reviewer 3

**R3.1 Hybrid DFT.**  
No limitation in the architecture. PBE0+D4 / aug-cc-pVDZ labels for the 70-frame ethanol+water subset are **complete** (`datasets/revision/pbe0/`). PBE0 SAD is written. Training the hybrid subset waits for a free GPU. ωB97M-V / OMol25 not opened this pass.

**R3.2 SOTA MLFFs and biomolecule IR benchmarks.**  
We compare MACE-OFF, AIMNet2, GFN2-xTB, and g-xTB on ethanol, ethanethiol, resorcinol, and thiophene 2-mer, and expand Figure 2.  

**Decline (with justification):** retraining DenSNet on Crambin in water (*J. Am. Chem. Soc.* 2025, 147, 33723–33734), paracetamol (*J. Am. Chem. Soc.* 2025, 147, 17598–17611), or FGG (*Chem. Eur. J.* 2005, 11, 6803–6817). Those papers benchmark *transferable* MLIPs. DenSNet is a system-specific density model; repeating those benchmarks would be a separate foundation-model paper. The fair test of “does learning ρ help IR?” is transferable MLIPs vs DenSNet on the systems we trained.

**R3.3 Relative errors and uncertainty.**  
Relative MAE plus a 3-seed ensemble mean ± std.

**R3.4 Delta vs direct cost.**  
SAD prior is a free-atom lookup; wall time is measured. Histogram |Δρ| / ρ and signed Δρ to show the correction is smaller than the total density.

**R3.5 Hyperparameters.**  
Document paper cutoffs (5.0 Å thiophene; 15 Bohr small molecules). Cutoff sweep {4, 5, 6, 8} Å on water clusters. Not a full grid search.

**R3.6 IR length.**  
IR at 50/100/200/500 ps for ethanol and thiophene 2-mer. 100 ps is kept only if peak positions have flattened.

**R3.7 General-purpose model.**  
Discussion: currently impractical at hybrid-DFT-density cost; a future path is active learning on a diverse organic set (QM7-X style), not a result of this paper.

**R3.8 Spectral language.**  
Replace “match across the entire frequency range” with agreement of **peak positions** in named windows; acknowledge intensity / area differences.

**R3.m1** Two-stage GPU-hours in the SI.  
**R3.m2** Add architecture blocks to Figure 1.  
**R3.m3** Density-derived properties: Hirshfeld charges, ESP, polarizability (finite field), core-level shifts — listed as feasible future observables, not new claims.

---

## Citations to add

1. Hazra, Patil, Sanvito, *J. Chem. Phys.* **2025**, 163, 174104. [10.1063/5.0292221](https://doi.org/10.1063/5.0292221)  
2. Cuevas-Zuviría, Pacios, *J. Chem. Inf. Model.* **2020**, 60, 3831–3842. [10.1021/acs.jcim.0c00197](https://doi.org/10.1021/acs.jcim.0c00197)  
3. Pracht et al., *J. Chem. Theory Comput.* **2024**. [10.1021/acs.jctc.4c01157](https://doi.org/10.1021/acs.jctc.4c01157)  
4. Hu et al. (TranSpec), *J. Am. Chem. Soc.* **2025**. [10.1021/jacs.5c05010](https://doi.org/10.1021/jacs.5c05010)  
5. AIQM chemRxiv. [10.26434/chemrxiv-2024-604wb](https://doi.org/10.26434/chemrxiv-2024-604wb)  

---

## Claims to delete or qualify before resubmission

- Frequency-dependent dielectric response of solvated H-bond networks (p. 25) unless water-cluster / condensed-phase data exist.  
- “Spectra match experiment/DFT across the entire frequency range.”  
- “Single network” if two modules run at MD time.  
- Companion network produces electronic observables.  
- Train/test non-overlap solely because MD runs were separate.  
- DenSNet equilibria “similar to DFT” without RMSD numbers.
