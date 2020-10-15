import torch
from .ewald import Ewald
import torch.nn as nn
import numpy as np


class LDAFunctional(nn.Module):
    def __init__(self, a_num, use_PME=False, grid_shape=np.array([20, 20, 20])):
        super().__init__()
        self.a_num = a_num
        self.use_PME = use_PME
        self.ewald = Ewald(a_num, PME=use_PME, grid_shape=grid_shape)

    def forward(self, rho, grid, pos):
        total_e = 0
        ewald_e = self.ewald(rho, grid, pos)
        return total_e + ewald_e
