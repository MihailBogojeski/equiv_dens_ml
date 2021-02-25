import torch.nn as nn
import numpy as np
import torch


class DFTNetwork(nn.Module):
    """
    Neural network for computing molecular electronic densities and density-derived properties
    """

    def __init__(self, density_repr_model,
                 property_model_dict,
                 calculate_forces_dict=None,
                 verbose=0):  # maximum nuclear charge ( + 1, i.e. 87 for up to Rn) for embeddings, can be kept at default
        super().__init__()
        self.density_repr_model = density_repr_model
        self.property_models = nn.ModuleDict(property_model_dict)
        self.verbose = verbose
        if calculate_forces_dict is None:
            calculate_forces_dict = {key: False for key in property_model_dict.keys()}
            self.calculate_forces = False
        else:
            self.calculate_forces = np.any(list(calculate_forces_dict.values()))

        # separate properties that require forces to calculate them first
        self.force_props = [key for key in calculate_forces_dict if calculate_forces_dict[key]]
        self.no_force_props = [key for key in calculate_forces_dict if not calculate_forces_dict[key]]
        if self.verbose > 0:
            print('force props', self.force_props)
            print('no force props', self.no_force_props)

    def forward(self, data):
        """
        Computes the electron density and needed density-derived properties

        inputs:
            atoms: Dictionary containing a collection of atoms and their various properties
        outputs:
            atoms: Updated dictionary containing the requested predicted properties
        """

        if self.verbose > 2:
            print('dft network forward:', torch.cuda.memory_summary())
            print(torch.cuda.memory_allocated() / 1024**2)
            print(torch.cuda.memory_cached() / 1024**2)
        atoms = {}
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                atoms[key] = data[key].clone()
            else:
                atoms[key] = data[key]
        if self.calculate_forces:
            atoms['positions'].requires_grad = True

        atoms = self.density_repr_model(atoms)
        if self.verbose > 2:
            print('dft network forward after repr:', torch.cuda.memory_summary())
        # run the models that require forces first, then turn off gradient for the positions
        for key in self.force_props:
            atoms = self.property_models[key](atoms)
            if self.verbose > 2:
                print('dft network forward after prop:', key, torch.cuda.memory_summary())

        atoms['positions'].requires_grad = False
        for key in self.no_force_props:
            atoms = self.property_models[key](atoms)
            if self.verbose > 2:
                print('dft network forward after prop:', key, torch.cuda.memory_summary())

        return atoms
