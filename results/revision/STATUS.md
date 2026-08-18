# Revision campaign status (2026-08-17, OMol25 skipped)

## Environment

`.venv` (Python 3.12.9) is complete. Do not force-reinstall torch.

| Import | Status |
| --- | --- |
| `equiv_dens` | OK |
| `schnetpack` | OK |
| `from mace.calculators import mace_off` | OK (`mace.calculators.mace_off` is a function, not a module) |
| `aimnet` | OK; AIMNet2 ASE compile fails without `Python.h` / triton |
| `tblite.ase.TBLite` | OK |

`.venv-revision` remains the CPU-DFT fallback. Node GPUs are still owned by `gpu_burn` + tps-cofolding; all new work used `CUDA_VISIBLE_DEVICES=""`.

## Running (tmux on gl056)

| Session | Job |
| --- | --- |
| `dft-pbe-train` | PBE+D4+DF water `train.xyz` (1250) |
| `dft-pbe-rest` | PBE+D4+DF ethanol OOD → water val → id_test → ood_size |
| `dft-pbe0` | PBE0+D4+DF 70-frame ethanol+water subset |

Resume with `bash scripts/revision/run_dft_campaign.sh {pbe-train,pbe-rest,pbe0}`.

## Computed

- g-xTB single points for all water splits + ethanol OOD (1 ethanol failure, index 146). Leftover-`energy` reuse bug fixed.
- Ethanol OOD overlap vs 10 parents; water ID overlap (too tight: all <0.2 Å).
- Water H-bond histograms (geometry-only).
- PBE / PBE0 SAD priors.
- Water-dimer PBE+D4+DF smoke + PBE BFGS geo-opt.
- CPU Figure 2 timings on ethanol.

## Recovered locally (not on the PR branch)

Copied from the sibling `equiv-paper` tree, Mihail `paper-project` LFS, and the Hasyim clone. Local-only via `.git/info/exclude` so they cannot be committed.

| Asset | Location |
| --- | --- |
| Published DenSNet weights (103 `*.pth`, 12G) | `paper/models/` |
| Ethanol paper ID `96w7KyGG` | `paper/models/ethanol/2024-03-22_96w7KyGG/best_96w7KyGG.pth` |
| Thiophene / resorcinol / qm7x / ethanethiol extras | `datasets/` |
| Water-102 CC/DFT labels | `datasets/revision/water_102/` |
| Manuscript TeX (sibling-only) | `scratch/recovered_manuscript/` |

Deleted after copy: `/scratch/mh7373/tmp/equiv_dens_ml_hasyim` and `/scratch/mh7373/projects/equiv-paper`.

Still missing by original name: `ethanol_dft_*.npy` paper-split densities.

## Blocked (need free GPU + finished water DFT)

- DenSNet train/eval, 3-seed uncertainty, cutoff sweep `{4,5,6,8}`, 200–500 ps IR, NVE with the density model. Weights are now local; these still need a free GPU and finished water PBE labels.
- AIMNet2 timings until `python3-devel` or `TORCHINDUCTOR` is disabled cleanly.
- OMol25 (explicitly skipped this pass).
