# JACS revision campaign scripts

Commands and status live in [docs/jacs_revision_todo.md](../../docs/jacs_revision_todo.md).

| Script | Role |
| --- | --- |
| `generate_water_cluster_geoms.py` | (H2O)_n minima + thermal frames (stdlib) |
| `generate_ood_geometries.py` | High-T noise and affine strain (stdlib) |
| `generate_dft_labels.py` | PySCF PBE or PBE0 + optional D4/DF labels |
| `build_sad_prior.py` | Free-atom SAD prior at a chosen XC |
| `analyze_train_test_overlap.py` | RMSD / pairwise-distance overlap |
| `analyze_density_metrics.py` | Relative MAE, signed Δρ, SAD magnitude |
| `evaluate_hbond_metrics.py` | O···O and O–H histograms |
| `optimize_densnet.py` | DenSNet vs DFT geometry optimization |
| `nve_energy_drift.py` | NVE total-energy drift |
| `ir_vs_length.py` | IR vs trajectory window |
| `benchmark_figure2.py` | Timing table for Figure 2 (includes `densnet` via ASE wrapper) |
| `run_gpu_campaign.sh` | Wave C tasks: `quick`, `pbe0`, `ethanol-md`, `thiophene-md`, `water`, `all`. Set `ALLOW_GL056=1` to share this allocation |
| `submit_gpu_campaign.sbatch` | One Slurm GPU job; `TASK=` selects the campaign |
| `submit_revision_gpu_jobs.sh` | Queue PBE0, ethanol MD, thiophene MD, and water-wait train |
| `launch_local_gpu.sh` | tmux: GPU0 quick+PBE0, GPU1 ethanol 500 ps |
| `score_ood_forces.py` | Force/energy MAE of `96w7KyGG` vs labeled OOD frames |
| `slice_labeled_pair.py` | Align geometry NPY length to a partial DFT NPY |
| `wait_and_train_water.sh` | Poll water PBE labels, then train + cutoff/seed sweeps |
| `download_omol25_index.sh` | Public 19 GB `train_4M` ASE-DB index (no HF token). Densities still need Globus |
| `download_qm7x.sh` | Official QM7-X Zenodo 4288677 HDF5 download (no densities) |
| `qm7x_extract_geoms.py` | HDF5 → JSONL geometries + official ePBE0+MBD / totFOR |
| `qm7x_build_shards.py` | JSON shards from DensNet `*_base.npy` or official JSONL |
| `qm7x_orca_worker.py` | One shard: ORCA PBE0/aug-cc-pVDZ EnGrad → DensNet calc_dict |
| `qm7x_assemble_npy.py` | Merge shard results into dens + base npy |
| `submit_qm7x_orca.sbatch` | CPU-only Slurm array: one frame per task (partition `cs`, qos `cpu48`) |
| `submit_qm7x_orca.sh` | One frame per Slurm task; 250-way throttle; train chunks of 2500 (cpu48 submit cap) |

| `run_dft_campaign.sh` | Resume-safe CPU PBE / PBE0 queue (`smoke`, `pbe-train`, `pbe-rest`, `pbe0`) |
| `run_gxtb_labels.py` | g-xTB single points (deletes leftover `energy`/`gradient` each frame) |
| `cpu_mlip_suite.py` | CPU geo-opt / short NVE for g-xTB, GFN2, MACE-OFF, AIMNet2 |
| `summarize_gxtb.py` | JSONL energy summary |
| `xyz_to_npy.py` | XYZ → positions/atom_numbers NPY |
| `prepare_pbe0_subset.py` | Small hybrid-DFT XYZ without paper ethanol frames |

Geometry scripts need only the Python standard library. DFT / MD / analysis scripts need `.venv` (`pip install -e . -r requirements.txt` plus `mace-torch`, `aimnet[ase]`, `tblite`). Live status: [results/revision/STATUS.md](../../results/revision/STATUS.md).
