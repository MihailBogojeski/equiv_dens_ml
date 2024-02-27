import torch.nn as nn
import numpy as np
import torch
from equiv_dens.utils.scaling import UnitConversion, VarianceScaling


class DFTNetwork(nn.Module):
    """
    Neural network for computing molecular electronic densities and density-derived properties
    """

    def __init__(self, density_repr_model,
                 property_model_dict,
                 calculate_forces_dict=None,
                 verbose=0,
                 memory=False,
                 conversions_in=UnitConversion(),
                 conversions_out=UnitConversion(),
                 scaling=VarianceScaling(),
                 remove_atom_density=False,):
        super().__init__()
        self.density_repr_model = density_repr_model
        self.property_models = nn.ModuleDict(property_model_dict)
        self.verbose = verbose
        self.conversions_in = conversions_in
        self.conversions_out = conversions_out
        self.memory = memory
        self.scaling = scaling
        if calculate_forces_dict is None:
            calculate_forces_dict = {key: False for key in property_model_dict.keys()}
            self.calculate_forces = False
        else:
            self.calculate_forces = np.any(list(calculate_forces_dict.values()))

        # separate properties that require forces to calculate them first
        self.force_props = [key for key in calculate_forces_dict if calculate_forces_dict[key]]
        self.no_force_props = [key for key in calculate_forces_dict if not calculate_forces_dict[key]]
        print('force props', self.force_props)
        print('no force props', self.no_force_props)
        self.remove_atom_density = remove_atom_density
        if self.verbose > 0:
            print('force props', self.force_props)
            print('no force props', self.no_force_props)

    def transform_input(self, data):
        """
        Convert and scale the input data to improve training.

        Args:
            data: Dictionary containing a collection of atoms and their various properties
        Returns:
            data: Updated dictionary containing the requested adjusted properties
        """
        data = self.conversions_in(data)
        data = self.scaling(data)
        return data

    def transform_density(self, data):
        """
        Transform density by removing the free atom densities if required.

        Args:
            data: Dictionary containing a collection of atoms and their various properties
        Returns:
            data: Updated dictionary containing the original properties
        """
        if 'density' in data.keys() and self.remove_atom_density:
            data['density'] -= data['atom_density']
        return data

    def transform_back_input(self, data):
        """
        Convert and scale the transformed input data back to its original form.

        Args:
            data: Dictionary containing a collection of atoms and their various properties
        Returns:
            data: Updated dictionary containing the original properties
        """
        data = self.scaling.transform_back(data)
        data = self.conversions_out(data)
        return data

    def transform_back_density(self, data):
        """
        Transform the density back to it's original form with the atom densities added back.

        Args:
            data: Dictionary containing a collection of atoms and their various properties
        Returns:
            data: Updated dictionary containing the original properties
        """
        if 'density' in data.keys() and self.remove_atom_density:
            data['density'] += data['atom_density']
        return data


    def forward(self, data):
        """
        Compute the electron density and needed density-derived properties.

        inputs:
            atoms: Dictionary containing a collection of atoms and their various properties
        outputs:
            atoms: Updated dictionary containing the requested predicted properties
        """
        if self.memory:
            print('dft network forward start:')
            print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
            print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        atoms = {}
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                atoms[key] = data[key].clone()
            else:
                atoms[key] = data[key]

        if self.calculate_forces:
            atoms['positions'].requires_grad = True

        atoms = self.density_repr_model(atoms)
        # run the models that require forces first, then turn off gradient for the positions
        for key in self.force_props:
            atoms = self.property_models[key](atoms)
            if self.memory:
                print('dft network forward', key, ':')
                print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
                print('Memory cached', torch.cuda.memory_cached() / 1024**2)
            # if self.verbose > 2:
            #     print('dft network forward after prop:', key, torch.cuda.memory_summary())

        atoms['positions'].requires_grad = False
        if self.training:
            for key in self.no_force_props:
                # if key == 'core_density':
                #     print('dft network forward', key, ':')
                #     print('spherical coeffs before', atoms['spherical_coeffs'][0][(1, 0)])
                #     print('radial scale before', atoms['radial_scale'][0][(1, 0)])
                #     print('radial width before', atoms['radial_width'][0][(1, 0)])
                atoms = self.property_models[key](atoms)
                if key == 'density' and self.remove_atom_density:
                    atoms['density'] += atoms['atom_density']
                # if key == 'core_density':
                #     print('spherical coeffs after', atoms['spherical_coeffs'][0][(1, 0)])
                #     print('radial scale after', atoms['radial_scale'][0][(1, 0)])
                #     print('radial width after', atoms['radial_width'][0][(1, 0)])
                if self.memory:
                    print('dft network forward', key, ':')
                    print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
                    print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        else:
            with torch.no_grad():
                for key in self.no_force_props:
                    atoms = self.property_models[key](atoms)
                    if key == 'density' and self.remove_atom_density:
                        atoms['density'] += atoms['atom_density']
                    if self.memory:
                        print('dft network forward', key, ':')
                        print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
                        print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        if not self.training:
            atoms = self.scaling.transform_back(atoms)
            atoms = self.conversions_out(atoms)

        return atoms
