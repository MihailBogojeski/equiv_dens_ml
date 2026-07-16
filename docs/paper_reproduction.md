# Paper Reproduction

One-command MD and inference for each molecule. Requires `git lfs pull` for models and trajectories.

## Models and paths

| Molecule | Model ID | Path |
|----------|----------|------|
| Polythiophene | 2024-03-23_1XDL67zp_ext | `paper/models/polythiophene/2024-03-23_1XDL67zp_ext` |
| Ethanethiol | 2024-02-29_NUmID4hT_ext4 | `paper/models/ethanethiol/2024-02-29_NUmID4hT_ext4` |
| Resorcinol | 2024-03-18_Ozf6CkNF_ext1 | `paper/models/resorcinol/2024-03-18_Ozf6CkNF_ext1` |
| Ethanol | 2024-03-22_96w7KyGG | `paper/models/ethanol/2024-03-22_96w7KyGG` |

## One-command examples

| Molecule | MD | Inference |
|----------|-----|-----------|
| Ethanethiol | `python run.py md @config/md/nn/ethanethiol.txt` | `python run.py infer --model paper/models/ethanethiol/2024-02-29_NUmID4hT_ext4 --trajectory path/to/traj.npy --dpm-intor` |
| Polythiophene | `python run.py md @config/md/nn/polythiophene_2mer.txt` | `python run.py infer --model paper/models/polythiophene/2024-03-23_1XDL67zp_ext --trajectory datasets/thiophene2mer_md.npy --dpm-intor` |
| Ethanol | Blocked (missing datasets) | `python run.py infer --model paper/models/ethanol/2024-03-22_96w7KyGG --trajectory path/to/traj.npy --dpm-intor` |
| Resorcinol | Blocked (missing density files) | `python run.py infer --model paper/models/resorcinol/2024-03-18_Ozf6CkNF_ext1 --trajectory path/to/traj.npy --dpm-intor` |

## Initial model extraction

If models are missing after clone:

```bash
./scripts/setup_paper_models.sh
```

Place zip archives in `paper/archives/` or set `ARCHIVES_DIR`. See [SETUP.md](SETUP.md) for details.
