import torch
from .ewald import Ewald
import torch.nn as nn
import numpy as np
import torch.fft as tfft


class LDAFunctional(nn.Module):
    def __init__(self, a_num, use_PME=False, pseudo_pot=None):
        super().__init__()
        self.a_num = a_num
        self.use_PME = use_PME
        self.ewald = Ewald(a_num, PME=use_PME)

    def forward(self, rho, grid, pos, pseudo_pot):
        total_e = 0
        ewald_e = self.ewald(rho, grid, pos)
        tf_e = thomas_fermi_en(rho, grid)
        print('thomas fermi energy', tf_e)
        vw_e = von_weizsacker_en(rho, grid)
        print('von weizsacker energy', vw_e)
        lda_e = lda_en(rho, grid)
        print('lda energy', lda_e)
        h_e = hartree_en(rho, grid)
        print('hartree energy', h_e)
        pseudo_e = pseudo_en(rho, grid, pseudo_pot)
        print('pseudo energy', pseudo_e)
        return total_e + ewald_e + tf_e + vw_e + lda_e + h_e + pseudo_e


def thomas_fermi_en(rho, grid):
    e_dens = rho**(5 / 3)
    en = torch.einsum('ijk->', e_dens)
    en *= (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * grid.point_volume
    return en


def von_weizsacker_en(rho, grid):
    pot = von_weizsacker_pot(rho, grid)
    en = torch.einsum("i, i->", rho[rho > 0], pot[rho > 0]) * grid.point_volume
    return en


def von_weizsacker_pot(rho, grid):
    tol = 1e-30
    rhom = rho.clone()
    mask = rho > 0
    mask2 = torch.logical_not(mask)
    rhom[mask2] = tol
    rec_grid = grid.get_reciprocal_grid()
    gg = rec_grid.gg
    sq_rho = torch.sqrt(rhom)
    sq_rho_fft = tfft.rfftn(sq_rho) * grid.point_volume
    n2_sq_rho = sq_rho_fft * gg

    a = tfft.irfftn(n2_sq_rho) / grid.point_volume
    if a.dtype == torch.complex128:
        a = a.real
    a *= 0.5
    sq_rho[torch.abs(sq_rho) < tol] = tol  # for safe
    pot = a / sq_rho
    return pot


def lda_en(rho, grid):
    a = (0.0311, 0.01555)
    b = (-0.048, -0.0269)
    c = (0.0020, 0.0007)
    d = (-0.0116, -0.0048)
    gamma = (-0.1423, -0.0843)
    beta1 = (1.0529, 1.3981)
    beta2 = (0.3334, 0.2611)

    rho_cbrt = rho**(1 / 3)
    rho_cbrt[rho_cbrt < 1e-30] = 1e-30
    rs = (3.0 / (4.0 * np.pi))**(1 / 3) / rho_cbrt
    rs1 = rs < 1
    rs2 = rs >= 1
    rs2sqrt = torch.sqrt(rs[rs2])

    ex_rho = -3.0 / 4.0 * (3.0 / np.pi)**(1 / 3) * rho_cbrt
    ex_rho[rs1] += a[0] * torch.log(rs[rs1]) + b[0] + c[0] * rs[rs1] *\
        torch.log(rs[rs1]) + d[0] * rs[rs1]
    ex_rho[rs2] += gamma[0] / (1.0 + beta1[0] * rs2sqrt + beta2[0] * rs[rs2])
    ene = np.einsum("ijk, ijk->", ex_rho, rho) * grid.point_volume

    return ene


def hartree_en(rho, grid):
    rec_grid = grid.get_reciprocal_grid()
    gg = rec_grid.gg.clone()
    rho_of_g = tfft.rfftn(rho)
    # v_h = rho_of_g.copy()
    # mask = gg != 0
    # v_h[mask] = rho_of_g[mask]*gg[mask]**(-1)*4*np.pi
    gg[0, 0, 0] = 1.0
    v_h = rho_of_g / gg * 4 * np.pi
    gg[0, 0, 0] = 0.0
    v_h[0, 0, 0] = 0.0
    v_h_of_r = tfft.irfftn(v_h)
    if v_h_of_r.dtype == torch.complex128:
        v_h_of_r = v_h_of_r.real

    e_h = torch.einsum("ijk, ijk->", v_h_of_r, rho) * grid.point_volume / 2.0
    return e_h


def pseudo_en(rho, grid, pseudo_pot):
    if pseudo_pot is None:
        return 0
    else:
        pseudo_v = torch.tensor(pseudo_pot.vreal)
        en = torch.einsum("ijk, ijk->", pseudo_v, rho) * grid.point_volume
        return en
