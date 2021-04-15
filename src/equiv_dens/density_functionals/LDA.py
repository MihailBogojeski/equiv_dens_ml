import torch
from .ewald import Ewald
import torch.nn as nn
import numpy as np
import torch.fft as tfft
from equiv_dens.utils.base import angstrom_to_bohr


class LDAFunctional(nn.Module):
    def __init__(self, a_num, use_PME=False, energy_offset=False, store_energy=False, verbose=0):
        super().__init__()
        self.a_num = a_num
        self.use_PME = use_PME
        self.ewald = Ewald(a_num, PME=use_PME, verbose=verbose)
        self.verbose = verbose
        self.store_energy = store_energy
        if energy_offset:
            self.en_offset = nn.Parameter(torch.zeros((1,)))
        else:
            self.register_buffer('en_offset', torch.zeros((1,)))

    def forward(self, atoms):
        rho, v_real, grid, pos = prepare_functional_vars(atoms)
        ewald_e = self.ewald(grid, pos)
        tf_e = thomas_fermi_en(rho, grid)
        vw_e = von_weizsacker_en(rho, grid)
        lda_e = lda_en(rho, grid)
        h_e = hartree_en(rho, grid)
        pseudo_e = pseudo_en(rho, grid, v_real)
        if self.verbose > 1:
            print('ewald energy', ewald_e)
            print('thomas fermi energy', tf_e)
            print('von weizsacker energy', vw_e)
            print('tf_vw en', tf_e + vw_e)
            print('lda energy', lda_e)
            print('hartree energy', h_e)
            print('pseudo energy', pseudo_e)
        total_e = ewald_e + tf_e + vw_e + lda_e + h_e + pseudo_e
        atoms['energy_min'] = total_e
        if self.store_energy:
            atoms['energy'] = total_e + self.en_offset
        return atoms


def prepare_functional_vars(atoms):
    grid = atoms['grid']
    dftpy_grid = atoms['dftpy_grid']
    rho = atoms['density'].view(-1, *grid.shape)
    pos = angstrom_to_bohr(atoms['shifted_positions'])
    pseudo_pot = atoms['pseudo_pot']
    v_real = []
    for i in atoms['idx']:
        pseudo_pot.restart(grid=dftpy_grid, ions=atoms['ions'][i])
        pseudo_pot(rho[i].detach().cpu().numpy())
        v_real.append(torch.from_numpy(pseudo_pot.vreal))

    v_real = torch.stack(v_real, dim=0).to(rho)

    return rho, v_real, grid, pos


def thomas_fermi_en(rho, grid):
    # print('rho', rho)
    e_dens = rho**(5 / 3)
    # print('e_dens', e_dens)
    en = torch.einsum('sijk->s', e_dens)
    en *= (3.0 / 10.0) * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * grid.point_volume
    return en


def von_weizsacker_en(rho, grid):
    pot = von_weizsacker_pot(rho, grid)
    # print('pot type', pot.type())
    # print('rho type', rho.type())
    en = torch.einsum("bijk, bijk->b", rho, pot) * grid.point_volume
    return en


def von_weizsacker_pot(rho, grid):
    rec_grid = grid.get_reciprocal_grid()
    gg = rec_grid.gg.clone()
    sq_rho = torch.sqrt(rho)
    sq_rho_fft = tfft.rfftn(sq_rho, dim=[1, 2, 3]) * grid.point_volume
    n2_sq_rho = sq_rho_fft * gg

    a = tfft.irfftn(n2_sq_rho, dim=[1, 2, 3]) / grid.point_volume
    if a.dtype == torch.complex128:
        a = a.real
    a *= 0.5
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

    # print('rho', rho)
    # print('min rho', torch.min(rho))
    # print('max rho', torch.max(rho))
    # print('point volume', grid.point_volume)
    rho_cbrt = rho**(1 / 3)
    # print('min cbrt', torch.min(rho_cbrt))
    # print('max cbrt', torch.max(rho_cbrt))
    # print('rho cbrt', rho_cbrt)
    rs = (3.0 / (4.0 * np.pi))**(1 / 3) / rho_cbrt
    # print('rs', rs)
    rs1 = (rs < 1).to(rs)
    rs2 = (rs >= 1).to(rs)
    rs2sqrt = torch.sqrt(rs)

    ex_rho = -3.0 / 4.0 * (3.0 / np.pi)**(1 / 3) * rho_cbrt
    ex_rho += (a[0] * torch.log(rs) + b[0] + c[0] * rs *
               torch.log(rs) + d[0] * rs) * rs1
    ex_rho += (gamma[0] / (1.0 + beta1[0] * rs2sqrt + beta2[0] * rs)) * rs2
    ene = torch.einsum("bijk, bijk->b", ex_rho, rho) * grid.point_volume

    return ene


def hartree_en(rho, grid):
    rec_grid = grid.get_reciprocal_grid()
    gg = rec_grid.gg.clone()
    rho_of_g = tfft.rfftn(rho, dim=(1, 2, 3))
    # v_h = rho_of_g.copy()
    # mask = gg != 0
    # v_h[mask] = rho_of_g[mask]*gg[mask]**(-1)*4*np.pi
    gg[0, 0, 0] = 1.0
    v_h = rho_of_g / gg * 4 * np.pi
    # gg[0, 0, 0] = 0.0
    # v_h[0, 0, 0] = 0.0
    mask = torch.ones_like(v_h)
    mask[:, 0, 0, 0] = 0.0
    v_h = v_h * mask
    v_h_of_r = tfft.irfftn(v_h, dim=(1, 2, 3))
    if v_h_of_r.dtype == torch.complex128:
        v_h_of_r = v_h_of_r.real

    e_h = torch.einsum("bijk, bijk->b", v_h_of_r, rho) * grid.point_volume / 2.0
    return e_h


def pseudo_en(rho, grid, v_real):
    if v_real is None:
        return 0
    else:
        en = torch.einsum("bijk, bijk->b", v_real, rho) * grid.point_volume
        return en
