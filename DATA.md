# Data Hosting and Download

Large trajectory and archive files (~9.7 GB) were moved off Git LFS to stay within GitHub's free tier (1 GiB). They are hosted on Zenodo/Figshare with persistent DOIs.

## Figure-reproduction kit (private Hub repo)

Everything the JACS-revision figures, tables and the step-timing benchmark read
(label arrays, SAD priors, atomic energies, basis tables, the minimal checkpoints
of every scored model, and the score/evaluation outputs) is packaged as the
private dataset repo
[muhammadhasyim/equiv_dens_ml_dataset](https://huggingface.co/datasets/muhammadhasyim/equiv_dens_ml_dataset)
(about 16 GiB), laid out exactly like this checkout (`datasets/`,
`results/revision/`, `paper/models/`). Once your Hugging Face account has been
granted access:

```bash
pip install -U huggingface_hub && hf auth login
scripts/data/fetch_reproduction_kit.sh --verify
```

This downloads the kit (default `/scratch/$USER/data/equiv_dens_ml_dataset`,
override with `--root` or `EQUIV_DENS_KIT_ROOT`), checks every file against the
kit's `manifest.json`, and symlinks the entries that are missing from the
checkout into place without touching tracked or local files. The kit's
`README.md` lists the command behind each figure. `scripts/data/build_reproduction_kit.py`
builds, uploads and verifies the kit.

## `datasets/` (Hugging Face Hub)

The `datasets/` tree (~103 GB of paper-era `.npy` labels plus the JACS revision
campaign: water clusters, malonaldehyde, QM7-X, OMol25/CSH manifests) lives on
[muhammadhasyim/equiv_dens_ml](https://huggingface.co/datasets/muhammadhasyim/equiv_dens_ml).
The checkout itself only tracks small authored geometry/manifest files
(`.xyz`, `.json`, ~4.3 MB); everything else under `datasets/` is a symlink
into a local data root, populated by:

```bash
pip install -U huggingface_hub  # provides the `hf` CLI
scripts/download_hf_datasets.sh
```

This downloads into `/scratch/$USER/data/equiv_dens_ml` by default (override
with `EQUIV_DENS_DATA_ROOT` or `--root DIR`), then symlinks the result into
`datasets/` without touching git-tracked files. Use `--subset water|qm7x|paper`
to fetch only part of the Hub repo.

**Not hosted on the Hub repo** (re-download from upstream instead):

- `datasets/revision/omol_csh/*.h5` (~318 GB) — raw OMol25 CSH electronic-structure
  dump. Fetch from [facebook/OMol25](https://huggingface.co/facebook/OMol25); the
  largest shard alone (~258 GB) exceeds the Hub's per-file limit.
- `datasets/revision/omol25/` (~20 GB) — OMol25 `train_4M` index mirror. Also from
  [facebook/OMol25](https://huggingface.co/facebook/OMol25).
- `datasets/revision/*/shards*/`, `datasets/revision/qm7x/shards/` — regenerated
  ORCA job-tracking manifests (`scripts/revision/lib/build_shards_from_xyz.py`),
  not scientific data.

See the [Hub dataset card](https://huggingface.co/datasets/muhammadhasyim/equiv_dens_ml)
for the full directory map and file formats.

## Automated download

After uploading to Zenodo and updating the record IDs in `scripts/download_data.sh`, run:

```bash
./scripts/download_data.sh
```

Use `--dry-run` to print planned downloads without fetching:

```bash
./scripts/download_data.sh --dry-run
```

## Datasets

| Dataset | Size | DOI / URL | Description |
|--------|------|-----------|-------------|
| Polythiophene full MD | ~3.6 GB | *(pending)* | 12 `.npy` files: 8/10/12mer trajectories |
| Polythiophene with velocity | ~4.2 GB | *(pending)* | 12 `.npy` files including velocities |
| thiophene_md_trajectories.zip | ~1.9 GB | *(pending)* | MD trajectory archive |
| polythiophene_1ps_mid_trajectories.zip | ~3.1 MB | *(pending)* | 1 ps mid trajectories |

## Manual download

1. Upload each dataset to [Zenodo](https://zenodo.org/) or [Figshare](https://figshare.com/).
2. Obtain the DOI and record URL.
3. Update `scripts/download_data.sh` with the Zenodo record IDs (e.g. `1234567` from `https://zenodo.org/record/1234567`).
4. Update this table with the DOIs.
5. Run `./scripts/download_data.sh`.

## Backup and purge

See [docs/LFS_BACKUP_AND_PURGE.md](docs/LFS_BACKUP_AND_PURGE.md) for instructions on backing up before running `git filter-repo` and for the purge workflow.
