import torch.nn as nn
import numpy as np
import torch
from equiv_dens.nn.modules.unit_conversion import UnitConversion


class DFTNetwork(nn.Module):
    """
    Neural network for computing molecular electronic densities and density-derived properties
    """

    def __init__(self, density_repr_model,
                 property_model_dict,
                 calculate_forces_dict=None,
                 verbose=0,
                 conversions_in=UnitConversion(),
                 conversions_out=UnitConversion()):
        super().__init__()
        self.density_repr_model = density_repr_model
        self.property_models = nn.ModuleDict(property_model_dict)
        self.verbose = verbose
        self.conversions_in = conversions_in
        self.conversions_out = conversions_out
        if calculate_forces_dict is None:
            calculate_forces_dict = {key: False for key in property_model_dict.keys()}
            self.calculate_forces = False
        else:
            self.calculate_forces = np.any(list(calculate_forces_dict.values()))

        # separate properties that require forces to calculate them first
        self.force_props = sorted([key for key in calculate_forces_dict if calculate_forces_dict[key]])
        self.no_force_props = sorted([key for key in calculate_forces_dict if not calculate_forces_dict[key]])
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
            print('dft network forward start:')
            print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
            print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        atoms = {}
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                atoms[key] = data[key].clone()
            else:
                atoms[key] = data[key]

        atoms = self.conversions_in(atoms)
        if self.calculate_forces:
            atoms['positions'].requires_grad = True

        atoms = self.density_repr_model(atoms)
        # if self.verbose > 2:
        #     print('dft network forward after repr:', torch.cuda.memory_summary())
        # run the models that require forces first, then turn off gradient for the positions
        for key in self.force_props:
            atoms = self.property_models[key](atoms)
            if self.verbose > 2:
                print('dft network forward', key, ':')
                print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
                print('Memory cached', torch.cuda.memory_cached() / 1024**2)
            # if self.verbose > 2:
            #     print('dft network forward after prop:', key, torch.cuda.memory_summary())

        atoms['positions'].requires_grad = False
        if self.training:
            for key in self.no_force_props:
                    atoms = self.property_models[key](atoms)
                    if self.verbose > 2:
                        print('dft network forward', key, ':')
                        print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
                        print('Memory cached', torch.cuda.memory_cached() / 1024**2)
        else:
            with torch.no_grad():
                for key in self.no_force_props:
                        atoms = self.property_models[key](atoms)
                        if self.verbose > 2:
                            print('dft network forward', key, ':')
                            print('Memory allocated', torch.cuda.memory_allocated() / 1024**2)
                            print('Memory cached', torch.cuda.memory_cached() / 1024**2)
            # if self.verbose > 2:
            #     print('dft network forward after prop:', key, torch.cuda.memory_summary())
        atoms = self.conversions_out(atoms)

        return atoms
