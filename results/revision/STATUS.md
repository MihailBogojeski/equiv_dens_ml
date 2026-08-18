# Revision campaign status (2026-08-17/18)

## Environment

`.venv` (Python 3.12.9) is complete. Do not force-reinstall torch. Node GPUs remain owned by `gpu_burn` + tps-cofolding.

## Running (tmux on gl056)

| Session | Job |
| --- | --- |
| `dft-pbe-train` | PBE+D4+DF water train (~275/1250) |
| `dft-pbe-rest` | ethanol OOD (~80/190) then water val / id_test / ood_size |
| `dft-pbe0` | PBE0 subset **70/70 labels written**; session may still be wrapping up |

Resume with `bash scripts/revision/run_dft_campaign.sh {pbe-train,pbe-rest,pbe0}`.

## Wave A (done)

- Paper-split overlap: `overlap_resorcinol.json` (median 0.093 Å, all <0.2 Å), `overlap_ethanethiol.json` (0.600 Å, none <0.2 Å), thiophene 4/6-mer and 12-mer notes.
- Figure 2 CPU: DenSNet **1.62 s/step**, g-xTB 0.70 s, GFN2 3.05 s, MACE-OFF 4.48 s.
- Ethanol DenSNet geo-opt RMSD 0.0 Å vs start; energy ~0.24 eV is **not** physical (2024 energy head ≠ current architecture).
- 20 fs CPU NVE: std 5.4 meV, drift 0.30 eV/ps (same caveat).
- Density metrics: h2o 46.7% and ethanethiol 51.0% negative DF coeffs.
- Manuscript wording + citations in `scratch/recovered_manuscript/` (gitignored).

## Wave B (partial)

Geometry NPYs written for water splits, ethanol OOD, and PBE0. OOD DFT not finished — **do not** run `eval_model_npy.py` until 190 frames. **Do not train on gl056.** Status: `wave_b_dft_status.json`.

## Wave C (ready, not launched)

`scripts/revision/run_gpu_campaign.sh` refuses to start on gl056 (exit 2). Submit `submit_gpu_campaign.sbatch` to a **different** Greene GPU node.

## Recovered assets (local only)

`paper/models/` (12G), extra `datasets/{thiophene*,resorcinol_*,qm7x_*,ethanethiol_*}`, manuscript TeX under `scratch/recovered_manuscript/`. Sibling repo and Hasyim clone deleted after copy.
