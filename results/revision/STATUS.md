# Revision campaign status (2026-08-21)

## Environment

`.venv` (Python 3.12.9) is complete. Do not force-reinstall torch.

`gpu_burn` on the `wrap` allocation (job 16137251, `gl063`) is **paused** via
`logs/gpu_burn/PAUSE` in the sibling `cofolding-boltzmann` checkout. Both of
that job's GPUs are running real tps-cofolding work (`oneopes` mek1 and
cryptic), which started after gpu-burn did, so the burn was duty-cycling
against them. Delete the PAUSE file to restore it. No cofolding job was touched.

## Labelling: complete

513/513 shards at **both** ωB97M-V/def2-TZVPD and PBE-D4, every cluster-size
bucket at 100%: water train 1250, val 250, ID test 250, size-OOD 300,
order-OOD 192, density-OOD 120, malonaldehyde 400/80/125.

Assembly had never written a single training file. `mol_from_pack` called
`gto.Mole.unpack` on an instance and discarded the return of what is a
classmethod, so every assembled frame had `natm=0`, the size histogram
collapsed to one "0 atoms" bucket, and `--min-complete` refused every split.
Each training job then died on `FileNotFoundError` about ninety seconds in.
The gate is the only reason nothing trained on empty geometries.

## Compute

Trainings run on **`a100_chemistry`**, this account's own partition. They had
been pinned to `l40s_public`, which sits at 263/272 GPUs with the rest
`PLANNED` for higher-priority users; `h200_public` caps all users at 24 GPUs in
aggregate. With fair-share at 0.006 that queue was not going to clear. Seven of
eight trainings started within thirty seconds of offering `a100_chemistry`.

Small evaluations go to the **CPU** partition. Ethanol is nine atoms; asking for
a GPU put minutes of arithmetic behind hours of queue under a shared
`QOSGrpGRES` cap.

## Running

| Job | What |
| --- | --- |
| `tr-water-{wb97mv_def2tzvpd,pbe_d4_avdz}` | the two headline models |
| `tr-malonaldehyde-*` | proton-transfer axis, both theories |
| `tr-cutoff_{4,5,6,8}-pbe_d4_avdz` | R3.5 sweep |
| `tr-water-wb97mv-s{7,123,20260821}` | R3.3 ensemble, `--init_seed` |
| `tr-water_direct-pbe_d4_avdz` | R3.4 direct-learning arm |
| `ir-md-{ethanol,thiophene2}` | R3.6 trajectories with dipoles |
| `ood-{pbe_d4_avdz,wb97mv_def2tzvpd}` | analysis chain, re-run by the watchdog |

All are resume-safe and the watchdog re-submits anything that hits a walltime.

## Established results

**R3.4 / R2.8 — the SAD correction, measured on a grid.** Over the water ID test
set at PBE-D4: `∫|Δρ| / ∫|ρ| = 0.110`, so the free-atom prior already supplies
89% of the density and the network predicts the remaining tenth. 84% of the
volume carries a *negative* correction, which is the direct answer to R2.8: the
softplus constrains `ρ_SAD + Δρ`, not `Δρ`. Reference fits integrate to the
correct electron count to ~1e-4. The previous figures were means of DF
coefficients, which are not densities and are not comparable across molecules.

**Published ethanol checkpoint — three loader defects fixed, one blocker left.**
`load_densnet_calculator` ranked an explicitly passed `--args-file` *below* the
run directory's own `args.txt`, so it was ignored for every published model;
`L0_start` then defaulted on and built `radial_L0_map` layers the 2024 model
never trained, left random by the non-strict load; and `atom_dens_path` was
overwritten unconditionally with the revision prior. The checkpoint now loads
deterministically with 0 missing and 0 unexpected tensors, and the loader raises
instead of returning a quietly wrong model.

That was not the whole story. `VarianceScaling.std` — the training-set force
standard deviation the outputs are expressed in — was a plain float, so it never
entered `state_dict`, and `datasets/ethanol_dft_pyscf_ccpvdz_train.npy` is gone.
Fitting the constant on the training geometries gives 90.36 but leaves the
predicted-versus-reference force correlation at **0.646** and an
in-distribution force MAE of 0.574 eV/Å against a 0.747 eV/Å reference. A scale
factor can only rescale, so this checkpoint's energy head cannot be validated
without its training set.

**R1.2, R2.4 and R2.11 are therefore blocked on restoring
`datasets/ethanol_dft_pyscf_ccpvdz_train.npy`.** Do not quote force or energy
numbers from `2024-03-22_96w7KyGG` until then — in particular the OOD force MAE
of 6.52 eV/Å, which equals the mean reference force to four digits because the
model's output is flat, not because it fails out of distribution.

`std` is now a registered buffer, so every model trained from here carries its
own scale and this cannot recur.
