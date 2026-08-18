# FORAGENT — JACS resubmission handoff (ja-2026-12808m)

Cloud agent: finish environment setup and the revision calculations. This file is the pending to-do list. Do **not** cancel other users’ or this user’s Slurm / GPU jobs on the node.

**Deadline:** resubmit to JACS by **13 February 2027** (Editor: Connor Coley).  
**Decision:** reject; resubmission invited.

---

## Do not touch

On the L40S node (`gl056`, job `15907642`) these must keep running:

- `gpu_burn.py` on GPU 0 and GPU 1
- tps-cofolding oneopes (mek1, cryptic)
- Any other Slurm jobs (`squeue -u mh7373`): `shd_*` on A100, `pg_boltz`, pending `tm0287v2`

A local `pip install` into `.venv` was **stopped on purpose** (2026-08-17 ~22:16 ET) so you can finish setup cleanly.

---

## Canonical docs (read these first)

| File | What it is |
| --- | --- |
| [ja-2026-12808m-reviews.md](ja-2026-12808m-reviews.md) | Immutable editor + 3 reviewer letters |
| [docs/jacs_revision_todo.md](docs/jacs_revision_todo.md) | Living reviewer checklist (partially stale vs OMol25 expansion) |
| [docs/jacs_response_letter_outline.md](docs/jacs_response_letter_outline.md) | Response-letter skeleton |
| [scripts/revision/README.md](scripts/revision/README.md) | Campaign scripts already written |
| Plans (Cursor): `JACS revision calculations` and `OMol25 revision expansion` | Full scientific design |

---

## Environment status (incomplete)

| Env | Python | Status |
| --- | --- | --- |
| `.venv-revision` | 3.9 | **Usable for CPU DFT only.** PySCF 2.14, ASE 3.26, dftd4. Too old for DenSNet / AIMNet2. |
| `.venv` | 3.12.9 | **PyTorch 2.13.0+cu126 works** (`cuda True`). `pip install -e . -r requirements.txt` was **interrupted** mid “Installing collected packages”. MACE / AIMNet / tblite **not** installed yet. |

**Pending env work**

1. Finish `pip install -e . -r requirements.txt` in `.venv` (resume; do not force-reinstall torch unless broken).
2. `pip install 'mace-torch>=0.3.0' 'aimnet[ase]' tblite`
3. Verify: `equiv_dens`, `schnetpack`, `mace.calculators.mace_off`, `aimnet`, `tblite.ase.TBLite`
4. Optional later: `equiv-dens[gpu-cuda12]` / SO3LR from the Crambin JACS 2025 paper. **Do not run GPU jobs on gl056 while tps-cofolding owns the GPUs.**

g-xTB binary is ready: `g-xtb/binary/gxtb` (executable), params in `g-xtb/parameters/`.

---

## Already done

- Reviewer letters copied to `ja-2026-12808m-reviews.md`
- Tracker + response outline under `docs/`
- Revision scripts in `scripts/revision/` (water geoms, OOD geoms, DFT labels, SAD, overlap, density metrics, H-bond histograms, geo-opt, NVE, IR-vs-length, Figure 2 timing, g-xTB labels)
- Configs: `config/training/water_clusters_001.txt`, `ethanol_pbe0_001.txt`, cutoff overrides, `config/md/nn/*_500ps.txt`
- Water-cluster XYZs: `datasets/revision/water_clusters/` (train 1250, val 250, id_test 250, ood_size 300)
- Ethanol OOD XYZs: `datasets/revision/ood/ethanol_ood.xyz` (190 frames from 10 parents)
- **g-xTB single points finished:** `results/revision/gxtb/water_{train,val,id_test,ood_size}.jsonl` and `ethanol_ood.jsonl`

---

## Missing data (owner must restore; do not invent)

- Paper densities: `datasets/ethanol_dft_*.npy`, ethanethiol/resorcinol/thiophene `*_d4.npy`
- Paper models: `paper/models/{ethanol,resorcinol,polythiophene}/` empty; ethanethiol only `.pth.bak`
- Manuscript / SI TeX
- Classical-MD logs for SI S2.2 (FF, length, SHAKE)
- `/home/ml-dft/equiv_dens/datasets` not on this machine

Ethanol AIMD on disk: `results/aimd_benchmark/aimd_ethanol_rep0_10ps.traj` (**10 ps**).

---

## Pending science (do these)

### A. Finish setup, then OMol25 (preferred data path)

OMol25 is ωB97M-V/def2-TZVPD (ORCA 6) — answers Reviewer 3’s hybrid/meta-GGA request. Electronic structure (densities / GBW / `density_mat.npz`) is the **4M split** via Globus ([MDF](https://www.materialsdatafacility.org/spotlight/omol25)), **not** the full 500 TB. Index first from [facebook/OMol25](https://huggingface.co/facebook/OMol25) `train_4M` ASE-DB.

1. Accept HF license; download **4M index only**.
2. Register Globus/MDF electronic-structure access (human approval).
3. Filter 4M index (neutral singlets) for:
   - NMA dimers (`C6H14N2O2` / two `C3H7NO`)
   - Methanol–water (`CH4O`+`H2O`)
   - Malonaldehyde `C3H4O2`, acetylacetone `C5H8O2`
   - Water clusters `(H2O)_n` (train n=2–6; hold out n≥8 if enough)
   - Paracetamol `C8H9NO2`
   - FGG / small peptides `C13H17N3O4` and ≤80-atom peptides
   - Optional 20k CHNO slice for a general-purpose density model
4. Globus-pull **only matching** `density_mat.npz` + ORCA outs. Estimate size first.
5. Write **ORCA DM → DenSNet DF-coeff** converter (paper model uses PySCF `aug-cc-pVQZ-JKFIT`; OMol25 is def2-TZVPD). Smoke-test on 10 waters.
6. Rebuild SAD at ωB97M-V ([scripts/revision/build_sad_prior.py](scripts/revision/build_sad_prior.py)).
7. Train two-stage DenSNet on **5k–30k** filtered frames, hold out by composition. Do **not** train on 4M.

If a formula is missing from 4M elec, fall back to in-house PBE+D4 via [scripts/revision/generate_dft_labels.py](scripts/revision/generate_dft_labels.py).

**Crambin in water:** do **not** train DenSNet (OMol25 max ~350 atoms). Run transferable MLIPs only (MACE-OFF, SO3LR, GFN2-xTB) vs JACS 2025, 147, 33723.

### B. In-house calculations (still required)

1. **PBE(+D4) DFT labels** on water-cluster XYZs and ethanol OOD (script ready; use `.venv-revision` or finished `.venv`, `--no-gpu` if the node’s GPUs are occupied).
2. Train DenSNet on water n=2–6; evaluate size extrapolation + H-bonds on n=8–12.
3. Score published ethanol/ethanethiol models on OOD (after checkpoints restored). Overlap analysis: [scripts/revision/analyze_train_test_overlap.py](scripts/revision/analyze_train_test_overlap.py).
4. Small **PBE0+D4** subset (~400 ethanol frames) only if OMol25 ωB97M-V conversion slips; otherwise OMol25 supersedes this for R3.1.
5. **IR bake-off** on paper molecules + paracetamol: DenSNet vs MACE-OFF vs SO3LR vs GFN2-xTB vs g-xTB vs DFTB (if it installs). Rebuild Figure 2 timings.
6. Ethanol + thiophene 2-mer ML-MD **200–500 ps**; IR vs length ([config/md/nn/ethanol_500ps.txt](config/md/nn/ethanol_500ps.txt)).
7. Geo-opt vs DFT + NVE drift ([scripts/revision/optimize_densnet.py](scripts/revision/optimize_densnet.py), [nve_energy_drift.py](scripts/revision/nve_energy_drift.py)).
8. Relative density MAE, signed Δρ, 3-seed uncertainty, cutoff sweep `{4,5,6,8}` Å.

### C. Manuscript (no new DFT)

- Cite Hazra–Sanvito JCP 2025, Cuevas-Zuviría JCIM 2020, Pracht JCTC 2024, TranSpec, AIQM, OMol25.
- Clarify companion network, dipoles, two-network MD, softplus+delta, normalization.
- Soften “spectra match across the full range”; move Figs 6–7 per R1.
- Units e/a0³; AIMD max time 10 ps ethanol.
- Corresponding authors MS vs SI.
- Response letter from [docs/jacs_response_letter_outline.md](docs/jacs_response_letter_outline.md).

---

## Suggested order for the cloud agent

1. Finish `.venv` install + import checks (no GPU compute).
2. HF + Globus access; 4M index; filter; size estimate.
3. Converter smoke test; then filtered Globus pull.
4. In-house water/OOD PBE DFT on CPU if GPUs are busy.
5. Train / eval / IR bake-off on a **free** GPU node, not on top of tps-cofolding.
6. Update [docs/jacs_revision_todo.md](docs/jacs_revision_todo.md) as items move to `computed`.
