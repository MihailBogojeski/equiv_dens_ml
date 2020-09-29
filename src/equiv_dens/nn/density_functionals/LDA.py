import torch
from .ewald import ewald_energy
import torch.nn as nn

def LDA_functional(rho=rho, pos=pos, a_num=a_num, use_PME=False):
    total_e = 0
    ewald_e = ewald_energy(rho, pos, a_num, PME=usePME)
