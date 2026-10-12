"""Check the installed legacy binary through the project's actual parser."""
import json
import tempfile
from pathlib import Path

import equiv_dens.compat  # noqa: F401
import numpy as np
from ase import Atoms
from equiv_dens.md.gxtb_calculator import GxtbMDCalculator


def main():
    with tempfile.TemporaryDirectory(prefix='densnet-gxtb-') as directory:
        calculator = GxtbMDCalculator('/opt/gxtb/bin/gxtb',
                                     '/opt/gxtb/parameters', Path(directory))
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
        energy, gradient = calculator._run_gxtb(atoms)
        assert np.isfinite(energy) and gradient.shape == (2, 3)
        assert np.isfinite(gradient).all()
        assert np.max(np.abs(gradient.sum(axis=0))) < 1e-5
        print(json.dumps({'gxtb': 'v1.1.0', 'project_energy_gradient_parser': 'pass',
                          'energy_Eh': energy, 'gradient_Eh_per_bohr': gradient.tolist()}, indent=2))


if __name__ == '__main__':
    main()
