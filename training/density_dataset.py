"""
This module contains all functionalities required to load atomistic data,
generate batches and compute statistics. It makes use of the ASE database
for atoms [#ase2]_.

References
----------
.. [#ase2] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
   Groves, Hammer, Hargus:
   The atomic simulation environment -- a Python library for working with atoms.
   Journal of Physics: Condensed Matter, 9, 27. 2017.
"""

import logging
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from schnetpack.data.partitioning import train_test_split
from pyscf import gto
from pyscf.dft import numint
from pyscf.lib import param
from .grids import spherical_grid, rot_spherical_sampling

logger = logging.getLogger(__name__)


class AtomsDataError(Exception):
    pass


class AtomsDensityData(Dataset):
    ENCODING = "utf-8"
    available_properties = None

    def __init__(
        self,
        np_path,
        density_path,
        orbitals_path,
        density_n_samp=10000,
        subset=None,
        required_properties=[],
        center_positions=True,
        radial_coeffs_file=None,
        grid_fn=spherical_grid,
        sampling_fn=rot_spherical_sampling,
        dtype=torch.float32
    ):
        print('Starting atomsdata density init')
        self.density_path = density_path
        self.np_path = np_path
        self.orbitals_path = orbitals_path
        self.density_n_samp = density_n_samp
        self.subset = subset
        self.required_properties = required_properties
        self.radial_coeffs_file = radial_coeffs_file
        self.grid_rn = grid_fn
        self.sampling_fn = sampling_fn
        self.dtype = dtype
        print('Some variables')
        if required_properties is None:
            self.required_properties = self.available_properties
        self.centered = center_positions
        self.atoms = np.load(np_path, allow_pickle=True).item()
        orbital_basis = np.load(orbitals_path, allow_pickle=True).item()
        self.orbitals = []
        for t in self.atoms['atom_types']:
            self.orbitals.append(orbital_basis[t])
        calc_results = np.load(density_path, allow_pickle=True)
        self.mols = []
        self.coeffs = []
        for i in range(len(calc_results)):
            mol_dict, coeff_dict = calc_results[i]
            mol = gto.Mole(**mol_dict)
            mol.build()
            self.mols.append(mol)
            self.coeffs.append(coeff_dict)

        if radial_coeffs_file != 'none':
            self.radial_coeffs = []
            radial_coeffs_atoms = np.load(radial_coeffs_file, allow_pickle=True).item()
            for t in self.atoms['atom_types']:
                self.radial_coeffs.append(radial_coeffs_atoms[t])
        else:
            self.radial_coeffs = None

        self.grid_spec = grid_fn(self.mols)

        print('finished init')

    def create_splits(self, num_train=None, num_val=None, split_file=None):
        warnings.warn(
            "create_splits is deprecated, " +
            "use schnetpack.data.train_test_split instead",
            DeprecationWarning,
        )
        return train_test_split(self, num_train, num_val, split_file)

    def create_subset(self, idx):
        """
        Returns a new dataset that only consists of provided indices.
        Args:
            idx (numpy.ndarray): subset indices

        Returns:
            schnetpack.data.AtomsData: dataset with subset of original data
        """
        idx = np.array(idx)
        subidx = (
            idx if self.subset is None or len(idx) == 0 else np.array(self.subset)[idx]
        )
        return type(self)(
            self.np_path,
            self.density_path,
            self.orbitals_path,
            self.density_n_samp,
            subidx,
            self.required_properties,
            self.centered,
            self.radial_coeffs_file,
            self.grid_fn,
            self.sampling_fn,
            self.dtype
        )

    def __len__(self):
        if self.subset is None:
            return len(self.mols)
        return len(self.subset)

    def __getitem__(self, idx):
        properties = self.get_properties(idx)
        properties["_idx"] = torch.LongTensor(np.array([idx], dtype=np.int))

        return properties

    def _subset_index(self, idx):
        # get row
        if self.subset is not None:
            idx = self.subset[idx]
        return idx

    def get_properties(self, idx):
        idx = self._subset_index(idx)
        if not hasattr(idx, '__len__'):
            idx = [idx]

        # extract properties
        properties = {}
        for pname in self.required_properties:
            # new data format
                # fallback for properties stored directly
                # in the row
            if pname != 'density':
                properties[pname] = torch.from_numpy(self.atoms[pname][idx])
            else:

                sample_coords, coord_weights = self.sampling_fn(self.grid_spec, self.density_n_samp,
                                                                self.atoms['atom_types'],
                                                                self.atoms['positions'][idx])
                properties[pname] = self.sample_density(idx, sample_coords)
                # print('density nans', torch.sum(torch.isnan(properties[pname])))
                properties['coords'] = torch.from_numpy(sample_coords).type(self.dtype)
                properties['coord_weights'] = torch.from_numpy(coord_weights).type(self.dtype)
                # print('coords_space', properties['coords_space'])

        # extract/calculate structure
        properties['atom_numbers'] = torch.LongTensor(self.atoms['atom_numbers'])
        positions = self.atoms['positions'][idx]
        properties['idx'] = idx
        # print('positions', positions)
        if self.centered:
            # print('atom center', positions.mean(axis=0))
            positions -= positions.mean(axis=0)
        properties['positions'] = torch.from_numpy(positions).type(self.dtype)

        return properties

    def sample_density(self, idx, sample_coords):
        scaled_sample_coords = sample_coords / param.BOHR  # convert Angstrom grid to Bohr
        dens = torch.zeros((sample_coords.shape[0], sample_coords.shape[1]), dtype=self.dtype)
        for c, i in enumerate(idx):
            # print('c, i', c, i)
            mol = self.mols[i]
            coeff_dict = self.coeffs[i]
            ao = numint.eval_ao(mol, scaled_sample_coords[c])
            rho = numint.eval_rho2(mol, ao, **coeff_dict)
            dens[c, :] = torch.from_numpy(rho).type(self.dtype)

        return dens
