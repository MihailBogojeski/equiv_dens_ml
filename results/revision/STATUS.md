# Revision campaign status (2026-08-18)

## Environment

`.venv` (Python 3.12.9) is complete. Do not force-reinstall torch. `gpu_burn` still holds ~1.2 GB / GPU; remaining L40S VRAM is used for reviewer jobs. Do not kill `gpu_burn` or other users.

## Running (tmux on gl056)

| Session | Job |
| --- | --- |
| `dft-pbe-train` | PBE+D4+DF water train (CPU; resume-safe) |
| `dft-pbe-rest` | ethanol OOD then water val / id_test / ood_size |
| `rev-gpu0-quick-pbe0` | GPU 0: Figure 2 + geo-opt + NVE + OOD force score, then PBE0 train |
| `rev-gpu1-ethanol-md` | GPU 1: ethanol 500 ps DenSNet MD |
| `rev-wait-water` | DFT count poller (water train is on the Slurm queue) |

Resume DFT with `bash scripts/revision/run_dft_campaign.sh {pbe-train,pbe-rest,pbe0}`.

## Submitted Slurm (may pend: qos `gpu168` max 4 GPUs / user)

`bash scripts/revision/submit_revision_gpu_jobs.sh` queues `dens-pbe0`, `dens-etohmd`, `dens-thio2`, `dens-water`.

## Wave A (done)

- Paper-split overlap, CPU Figure 2, density-sign diagnostics, water-dimer DFT geo-opt.
- Ethanol DenSNet energy is **not** physical until the 2024 energy head is restored (owner).

## Wave B (partial)

PBE0 labels **70/70**. Water / ethanol-OOD PBE still running. Do not start water DenSNet until train 1250 + val 250 + ood 300 exist.

## OMol25 (back on the live list)

Preferred R3.1 path: ωB97M-V/def2-TZVPD densities from the **4M electronic-structure split**, then train 5k–30k filtered frames. Blocked on human HF + Globus/MDF group approval. Do **not** pull the 500 TB dump.

## Wave C (launched)

`ALLOW_GL056=1` now permits sharing this allocation.

| Local GPU | Status |
| --- | --- |
| GPU 0 Figure 2 | DenSNet **467 ms/step** (0.093 ns/day). MACE-OFF/AIMNet2 still fail Triton/`Python.h`; SO3LR not installed |
| GPU 0 geo-opt | RMSD 0.0 Å; energy 0.224 eV (not physical until energy head is restored) |
| GPU 0 NVE 0.1 ps | std 8.4 meV, drift 0.18 eV/ps, 0.069 ns/day |
| GPU 0 OOD forces | **80 frames**: force MAE 5.01 eV/Å, energy MAE ~4206 eV (not physical) |
| GPU 0 PBE0 train | reached first forward; died on energy DF feature 16 vs 27 (needs paper DF SAD / energy-head fix). GPU 0 now running ASE 500 ps MD |
| GPU 0 ASE MD | `ase_densnet_md.py` 1e6 steps — this calculator path already works |
| GPU 1 | ethanol 500 ps `run.py md` (SAD `mo_basis` patched; ASE Langevin fallback if it dies) |

Slurm (qos `gpu48`, Priority): `dens-pbe0` 15934887, `dens-ethanol-md` 15934889, `dens-thiophene-md` 15934890.

## Recovered assets (local only)

`paper/models/` (12G), extra `datasets/{thiophene*,resorcinol_*,qm7x_*,ethanethiol_*}`, manuscript TeX under `scratch/recovered_manuscript/`.
