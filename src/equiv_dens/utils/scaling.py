import torch.nn as nn
import numpy as np


class UnitConversion(nn.Module):
    """Helper class for unit conversions"""

    def __init__(self, en_conversion_func=nn.Identity,
                 dist_conversion_func=nn.Identity):
        super().__init__()
        self.en_conversion_func = en_conversion_func
        self.dist_conversion_func = dist_conversion_func

    def forward(self, atoms):
        atoms['positions'] = self.dist_conversion_func(atoms['positions'])
        if 'energy' in atoms.keys():
            atoms['energy'] = self.en_conversion_func(atoms['energy'])
        if 'forces' in atoms.keys():
            atoms['forces'] = self.en_conversion_func(atoms['forces']) / self.dist_conversion_func(1)

        return atoms


class VarianceScaling(nn.Module):
    """Helper class for unit conversions"""

    def __init__(self, x=None):
        super().__init__()
        if x is None:
            self.std = 1
        else:
            self.std = np.std(x)

    def forward(self, atoms):
        atoms['energy'] = atoms['energy'] / self.std
        atoms['forces'] = atoms['forces'] / self.std

        return atoms

    def transform_back(self, atoms):
        atoms['energy'] = atoms['energy'] * self.std
        atoms['forces'] = atoms['forces'] * self.std

        return atoms
