"""Compare a completed paper model using an explicitly copied, isolated fixture.

Run with each checkout's own interpreter, source path and fixture directory.
The full small training array is retained to reconstruct output scaling.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet

from equiv_dens.md.dft_network_calculator import load_densnet_calculator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    random.seed(1729)
    np.random.seed(1729)
    torch.manual_seed(1729)
    torch.set_num_threads(1)
    dataset_path = root / "datasets/ethanol_dft_pyscf_ccpvdz_train.npy"
    data = np.load(dataset_path, allow_pickle=True).item()
    calculator = load_densnet_calculator(
        root / "paper/models/ethanol/2024-03-22_96w7KyGG",
        np_dataset=dataset_path,
        repo=root,
        use_gpu=False,
        pyscf_grid=False,
    )
    frames = []
    for index in (0, 1, 2):
        atoms = Atoms(numbers=data["atom_numbers"][index],
                      positions=data["positions"][index])
        atoms.calc = calculator
        frames.append({"energy": atoms.get_potential_energy(),
                       "forces": atoms.get_forces().tolist()})
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    VelocityVerlet(atoms, timestep=0.2 * units.fs).run(5)
    result = {
        "frames": frames,
        "md": {"positions": atoms.positions.tolist(),
               "momenta": atoms.get_momenta().tolist(),
               "energy": atoms.get_potential_energy(),
               "forces": atoms.get_forces().tolist()},
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
