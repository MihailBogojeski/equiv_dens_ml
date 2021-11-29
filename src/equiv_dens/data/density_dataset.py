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
from equiv_dens.utils.grids import spherical_grid,\
    spherical_radial_sampling, treutler_atomic_radii_adjust 
import equiv_dens.utils.base as utils
from dftpy.formats import ase_io
from pyscf.dft import radi
# import time

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
        L0_coeffs_file=None,
        grid_fn=spherical_grid,
        sampling_fn=spherical_radial_sampling,
        dtype=torch.float32,
        grid_extent=None,
        grid_origin=0,
        fixed_properties={},
        verbose=0,
        use_gpu=False,
        radii_adjust=True,
    ):
        print('Starting atomsdata density init')
        self.density_path = density_path
        self.np_path = np_path
        self.orbitals_path = orbitals_path
        self.density_n_samp = density_n_samp
        self.subset = subset
        self.required_properties = required_properties
        self.radial_coeffs_file = radial_coeffs_file
        self.L0_coeffs_file = L0_coeffs_file
        self.grid_fn = grid_fn
        self.sampling_fn = sampling_fn
        self.dtype = dtype
        self.grid_extent = grid_extent
        self.grid_origin = grid_origin
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.radii_adjust = radii_adjust
        print('Some variables')
        if required_properties is None:
            self.required_properties = self.available_properties
        self.centered_positions = center_positions
        self.atoms = np.load(np_path, allow_pickle=True).item()
        orbital_basis = np.load(orbitals_path, allow_pickle=True).item()
        self.orbitals = []
        print('atoms keys', self.atoms.keys())
        for t in self.atoms['atom_types']:
            self.orbitals.append(orbital_basis[t])
        self.atoms['shifted_positions'] = self.atoms['positions'] - grid_origin
        if self.density_path is not None:
            calc_results = np.load(density_path, allow_pickle=True)
        self.mols = []
        self.coeffs = []
        self.ions = []
        ase_atoms = utils.npy_to_ase(self.atoms['shifted_positions'], self.atoms['atom_types'])
        # for i in range(10):
        for i in range(self.atoms['positions'].shape[0]):
            if self.verbose > 3:
                print('loading sample', i)
            if self.density_path is not None:
                mol_dict, calc_dict = calc_results[i]
                coeff_dict = {'mo_coeff': calc_dict['mo_coeff'], 'mo_occ': calc_dict['mo_occ']}
                mol = gto.Mole(**mol_dict)
                self.mols.append(mol)
                self.coeffs.append(coeff_dict)
            a = ase_atoms[i]
            a.set_cell(grid_extent)
            self.ions.append(ase_io.ase2ions(a))

        if radial_coeffs_file is not None:
            self.radial_coeffs = []
            radial_coeffs_atoms = np.load(radial_coeffs_file, allow_pickle=True).item()
            for t in self.atoms['atom_types']:
                self.radial_coeffs.append(radial_coeffs_atoms[t])
        else:
            self.radial_coeffs = None

        if L0_coeffs_file is not None:
            self.L0_coeffs = {}
            L0_coeffs_types = np.load(L0_coeffs_file, allow_pickle=True).item()
            for coeff_type in L0_coeffs_types.keys():
                self.L0_coeffs[coeff_type] = []
                for t in self.atoms['atom_types']:
                    L0_atom = L0_coeffs_types[coeff_type][t]
                    self.L0_coeffs[coeff_type].append(L0_atom)
        else:
            self.L0_coeffs = None

        self.grid_spec = grid_fn(self.atoms)

        for key in self.grid_spec.keys():
            self.grid_spec[key] = (self.grid_spec[key][0].type(self.dtype),
                                   self.grid_spec[key][1].type(self.dtype))
        if self.use_gpu:
            for key in self.grid_spec.keys():
                self.grid_spec[key] = (self.grid_spec[key][0].cuda(),
                                       self.grid_spec[key][1].cuda())  # convert Bohr grid to Angstrom
        self.fixed_properties = fixed_properties
        if self.verbose > 1:
            print('dataset init grid_spec type', self.grid_spec['H'][0].type())
            print('dataset radial coeffs', self.radial_coeffs)
            print('dataset L0 coeffs', self.L0_coeffs)

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
            np_path=self.np_path,
            desnity_path=self.density_path,
            orbitals_path=self.orbitals_path,
            density_n_samp=self.density_n_samp,
            subset=subidx,
            required_propertie=self.required_properties,
            center_positions=self.centered_positions,
            radial_coeffs_file=self.radial_coeffs_file,
            grid_fn=self.grid_fn,
            sampling_fn=self.sampling_fn,
            dtype=self.dtype,
            grid_extent=self.grid_extent,
            grid_origin=self.grid_origin,
            fixed_properties=self.fixed_properties,
        )

    def __len__(self):
        if self.subset is None:
            if len(self.mols) > 0:
                return len(self.mols)
            else:
                return self.atoms['positions'].shape[0]
        return len(self.subset)

    # property collection is done by the get_properties function when passed as collate_fn
    def __getitem__(self, idx):
        return idx

    def _subset_index(self, idx):
        # get row
        if self.subset is not None:
            idx = self.subset[idx]
        return idx

    def center_energy(self, energy_mean):
        print('pre centering:', self.atoms['energy'][:20])
        self.atoms['energy'] -= energy_mean
        print('post centering:', self.atoms['energy'][:20])

    # collects the molecular properties for the batch, should be used as collate_fn
    def get_properties(self, idx):
        idx = self._subset_index(idx)
        if not hasattr(idx, '__len__'):
            idx = [idx]

        # extract properties
        properties = {}
        positions = torch.from_numpy(self.atoms['positions'][idx]).type(self.dtype)
        print('required properties', self.required_properties)
        for pname in self.required_properties:
            # fallback for properties stored directly
            # in the row
            if pname == 'coords' or pname == 'density':
                properties['coords'], properties['coord_weights'] = self.get_coords(positions, self.atoms['atom_types'])
                if pname == 'density':
                    properties[pname] = self.sample_density(idx, properties['coords'])
            else:
                properties[pname] = torch.from_numpy(self.atoms[pname][idx])

        # extract/calculate structure
        properties['atom_numbers'] = torch.LongTensor(self.atoms['atom_numbers']).unsqueeze(0).repeat(len(idx), 1)
        properties['idx'] = torch.LongTensor(idx).unsqueeze(-1)
        # properties['ions'] = [self.ions[i] for i in idx]
        # print('positions', positions)
        if self.centered_positions:
            # print('atom center', positions.mean(axis=0))
            positions -= positions.mean(0)
        properties['positions'] = positions
        print('properties positions type', properties['positions'].type())
        properties['shifted_positions'] = torch.from_numpy(self.atoms['shifted_positions'][idx]).type(self.dtype)
        properties["_idx"] = torch.LongTensor(np.array(idx, dtype=np.int))
        for prop in self.fixed_properties.keys():
            properties[prop] = self.fixed_properties[prop]

        return properties

    def get_coords(self, positions, atom_types):
        if self.use_gpu:
            positions = positions.cuda()
        if self.radii_adjust:
            f_radii_adjust = treutler_atomic_radii_adjust(utils.symbols_to_numbers(atom_types),
                                                          radi.BRAGG_RADII)
            sample_coords, coord_weights = self.sampling_fn(self.grid_spec, self.density_n_samp,
                                                            atom_types,
                                                            positions, radii_adjust=f_radii_adjust)
        else:
            sample_coords, coord_weights = self.sampling_fn(self.grid_spec, self.density_n_samp,
                                                            atom_types,
                                                            positions)
        # print('density nans', torch.sum(torch.isnan(properties[pname])))
        coords = sample_coords.type(self.dtype)
        coord_weights = coord_weights.type(self.dtype)

        return coords, coord_weights

    def sample_density(self, idx, sample_coords):
        scaled_sample_coords = sample_coords.detach().cpu().numpy() / param.BOHR  # convert Angstrom grid to Bohr
        dens = torch.zeros((sample_coords.shape[0], sample_coords.shape[1]), dtype=self.dtype)
        if len(self.mols) > 0:
            for c, i in enumerate(idx):
                # mol_start = time.time()
                # print('c, i', c, i)
                mol = self.mols[i]
                if not mol._built:
                    # build_start = time.time()
                    if self.verbose > 3:
                        print('building mol', i)
                    mol.build()
                    # print('build time', time.time() - build_start)
                coeff_dict = self.coeffs[i]
                # ao_start = time.time()
                ao = numint.eval_ao(mol, scaled_sample_coords[c])
                # print('ao time', time.time() - ao_start)
                # rho_start = time.time()
                rho = numint.eval_rho2(mol, ao, **coeff_dict)
                # print('rho time', time.time() - rho_start)
                dens[c, :] = torch.from_numpy(rho).type(self.dtype)
                # print('mol_time', time.time() - mol_start)

        return dens

    def add_fixed_properties(self, property_dict):
        for prop in property_dict.keys():
            self.fixed_properties[prop] = property_dict[prop]
