import torch.nn as nn


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
