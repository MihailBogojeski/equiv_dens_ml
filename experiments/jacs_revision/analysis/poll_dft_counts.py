#!/usr/bin/env python3
"""Print current CPU DFT label counts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PAIRS = [
    ("water_train", "datasets/revision/water_clusters/water_train_pyscf_augccpvdz_pbe.npy", 1250),
    ("water_val", "datasets/revision/water_clusters/water_val_pyscf_augccpvdz_pbe.npy", 250),
    ("water_id", "datasets/revision/water_clusters/water_id_test_pyscf_augccpvdz_pbe.npy", 250),
    ("water_ood", "datasets/revision/water_clusters/water_ood_size_pyscf_augccpvdz_pbe.npy", 300),
    ("ethanol_ood", "datasets/revision/ood/ethanol_ood_pyscf_augccpvdz_pbe.npy", 190),
    ("pbe0", "datasets/revision/pbe0/ethanol_water_pbe0_pyscf_augccpvdz_pbe0.npy", 70),
]


def main():
    for name, rel, target in PAIRS:
        path = Path(rel)
        if not path.exists():
            print(f"{name}: missing / {target}")
            continue
        n = len(np.load(path, allow_pickle=True))
        print(f"{name}: {n}/{target}")


if __name__ == "__main__":
    main()
