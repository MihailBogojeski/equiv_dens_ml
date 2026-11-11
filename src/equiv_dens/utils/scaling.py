import torch
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
    """Rescale energies and forces by the training-set force standard deviation.

    ``std`` is a registered buffer so it travels with the checkpoint. It used to
    be a plain float, which meant it was absent from ``state_dict`` and had to be
    recomputed from the training data at load time. Anything loading a model
    without that data -- an ASE calculator, an MD driver, an evaluation script --
    silently got ``std = 1`` and every predicted energy and force came back
    divided by the true scale.

    That is not a small error and it does not look like one. Predictions stay
    smooth and finite, just far too flat: the published ethanol model returned
    forces of about 0.005 eV/A on every geometry it was shown, whether the
    reference force was 0.5 or 7.6 eV/A, so its OOD force MAE came out equal to
    the mean reference force to five digits. Read without the reference scale
    beside it, that reads as a model which simply fails out of distribution.

    Old checkpoints have no ``std`` entry, so loading one non-strictly leaves
    this buffer at its constructed value; pass the training forces to recover
    the scale, or accept that predictions are in normalised units.
    """

    def __init__(self, x=None):
        super().__init__()
        std = 1.0 if x is None else float(np.std(x))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.get_default_dtype()))

    def forward(self, atoms):
        if 'energy' in atoms.keys():
            atoms['energy'] = atoms['energy'] / self.std
        if 'forces' in atoms.keys():
            atoms['forces'] = atoms['forces'] / self.std

        return atoms

    def transform_back(self, atoms):
        if 'energy' in atoms.keys():
            atoms['energy'] = atoms['energy'] * self.std
        if 'forces' in atoms.keys():
            atoms['forces'] = atoms['forces'] * self.std

        return atoms
