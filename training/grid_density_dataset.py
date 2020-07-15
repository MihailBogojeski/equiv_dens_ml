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
import h5py

logger = logging.getLogger(__name__)


class AtomsDataError(Exception):
    pass


class AtomsDensityDataHDF5(Dataset):
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
        dtype=torch.float32
    ):
        print('Starting atomsdata density init')
        self.density_path = density_path
        self.np_path = np_path
        self.orbitals_path = orbitals_path
        self.density_n_samp = density_n_samp
        self.subset = subset
        self.required_properties = required_properties
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
        f = h5py.File(density_path, 'r')
        self.density_dset = f['densities']
        print('dataset shape', self.density_dset.shape)
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
        )

    def __len__(self):
        if self.subset is None:
            return self.density_dset.shape[0]
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

        # extract properties
        properties = {}
        for pname in self.required_properties:
            # new data format
                # fallback for properties stored directly
                # in the row
            if pname != 'density':
                properties[pname] = torch.from_numpy(self.atoms[pname][idx])
            else:
                samples = np.random.choice(self.density_dset.shape[1],
                                           size=(self.density_n_samp, ),
                                           replace=False)
                samples = np.sort(samples)
                # print('density data shape', self.density_dset.shape)
                properties[pname] = torch.from_numpy(self.density_dset[:, samples]).type(self.dtype)
                properties[pname] = properties[pname][idx]
                coords = np.array(np.unravel_index(samples, (125, 125, 125))).T
                coords = coords[None, :]
                properties['indices'] = torch.from_numpy(samples).type(self.dtype)
                properties['coords_mat'] = torch.from_numpy(coords).type(self.dtype)
                properties['coords_space'] = torch.from_numpy((coords - 62) * (20 / 125)) * 0.529117
                properties['coords_space'] = properties['coords_space'].type(self.dtype)

        # extract/calculate structure
        properties['atom_numbers'] = torch.LongTensor(self.atoms['atom_numbers'])
        positions = self.atoms['positions'][idx]
        # print('positions', positions)
        if self.centered:
            # print('atom center', positions.mean(axis=0))
            positions -= positions.mean(axis=0)
        properties['positions'] = torch.from_numpy(positions).type(self.dtype)

        return properties

    # def collate_fn(self, batch):
    #     # print('batch', batch)
    #     # print(batch[0]['density'].shape)
    #     return self.get_properties(batch)


class AtomsDensityDataNpy(Dataset):
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
        dtype=torch.float32
    ):
        print('Starting atomsdata density init')
        self.density_path = density_path
        self.np_path = np_path
        self.orbitals_path = orbitals_path
        self.density_n_samp = density_n_samp
        self.subset = subset
        self.required_properties = required_properties
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
        self.density_dset = np.load(density_path, allow_pickle=True)
        print('dataset shape', self.density_dset.shape)
        # print('dataset nans', np.sum(np.isnan(self.density_dset)))
        if radial_coeffs_file != 'none':
            self.radial_coeffs = []
            radial_coeffs_atoms = np.load(radial_coeffs_file, allow_pickle=True).item()
            for t in self.atoms['atom_types']:
                self.radial_coeffs.append(radial_coeffs_atoms[t])
        else:
            self.radial_coeffs = None

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
        )

    def __len__(self):
        if self.subset is None:
            return self.density_dset.shape[0]
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

        # extract properties
        properties = {}
        for pname in self.required_properties:
            # new data format
                # fallback for properties stored directly
                # in the row
            if pname != 'density':
                properties[pname] = torch.from_numpy(self.atoms[pname][idx])
            else:
                # samples = np.random.choice(self.density_dset.shape[1],
                #                            size=(self.density_n_samp, ),
                #                            replace=False)
                # samples = np.sort(samples)
                # # print('samples shape', samples.shape)
                # # print('density dest shape', self.density_dset.shape)
                # # print('idx', idx)
                # # print('density idx', self.density_dset[idx, :10])
                # # print('density data shape', self.density_dset.shape)
                # properties[pname] = torch.from_numpy(self.density_dset[:, samples]).type(self.dtype)
                # properties[pname] = properties[pname][idx, :]
                # properties[pname] = properties[pname][idx]
                X, Y, Z = np.meshgrid(np.arange(50) + 38, np.arange(50) + 38, np.arange(50) + 38)
                X = X.flatten()
                Y = Y.flatten()
                Z = Z.flatten()
                indices = np.ravel_multi_index((X, Y, Z), (125, 125, 125))
                # print('indices shape', indices.shape)
                properties[pname] = torch.from_numpy(self.density_dset[:, indices]).type(self.dtype)
                properties[pname] = properties[pname][idx, :]
                # print('density nans', torch.sum(torch.isnan(properties[pname])))
                coords = np.array(np.unravel_index(indices, (125, 125, 125))).T
                coords = coords[None, :]
                properties['indices'] = torch.from_numpy(indices).type(self.dtype)
                properties['coords_mat'] = torch.from_numpy(coords).type(self.dtype)
                properties['coords_space'] = torch.from_numpy((coords - 62) * (20 / 125)) * 0.529117
                properties['coords_space'] = properties['coords_space'].type(self.dtype)
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
