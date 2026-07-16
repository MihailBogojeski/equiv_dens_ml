# Data Hosting and Download

Large trajectory and archive files (~9.7 GB) were moved off Git LFS to stay within GitHub's free tier (1 GiB). They are hosted on Zenodo/Figshare with persistent DOIs.

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
