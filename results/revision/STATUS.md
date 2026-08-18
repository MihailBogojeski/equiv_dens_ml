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

## Wave C (launched)

`ALLOW_GL056=1` now permits sharing this allocation. Local GPUs run quick tests + PBE0 + ethanol MD. Longer / duplicate jobs are queued for a free node.

## Recovered assets (local only)

`paper/models/` (12G), extra `datasets/{thiophene*,resorcinol_*,qm7x_*,ethanethiol_*}`, manuscript TeX under `scratch/recovered_manuscript/`.
