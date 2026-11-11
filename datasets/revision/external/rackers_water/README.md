# Rackers / Rakshit water-cluster geometries

Published set named for the revision comparison (Rackers, Tecot, Geiger, Smidt,
*Mach. Learn.: Sci. Technol.* **4**, 015027 (2023); arXiv:2201.03726).

- Geometries: subset of the Rakshit water-cluster minima (n=3–30).
- Zenodo densities: https://doi.org/10.5281/zenodo.5563139
- Their DFT is PBE0/aug-cc-pVTZ (and CCSD) in a def2-universal-JFIT basis.
  Do not train DenSNet on those coefficients.

Place a multi-frame XYZ at `raw.xyz` (download from Zenodo or the UW minima
atlas), then:

```bash
./.venv/bin/python scripts/revision/water/import_rackers_parents.py \
  --xyz datasets/revision/external/rackers_water/raw.xyz \
  --existing-xyz datasets/revision/water_clusters/train.xyz \
                 datasets/revision/water_clusters/ood_size.xyz
```

n=3–6 → train parents; n=8,10,12 → size-OOD; n≥15 → `ood_rackers` (held out).
