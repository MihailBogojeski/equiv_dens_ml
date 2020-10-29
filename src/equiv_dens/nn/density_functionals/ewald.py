import torch
import torch.nn as nn
from scipy import special as sp
import numpy as np


class Ewald(nn.Module):
    def __init__(
        self, a_num, PME=False, prec=1.0e-8, eta=0.73
    ):
        super().__init__()
        self.a_num = a_num
        self.PME = PME
        self.prec = prec
        self.eta = eta
        self.mask = None

    def forward(self, rho, grid, pos):
        self.real_en = self.real_energy(rho, grid, pos)
        print("real energy", self.real_en)
        self.corr_en = self.corr_energy(rho, grid, pos)
        print("corr energy", self.corr_en)
        self.rec_en = self.rec_energy(rho, grid, pos)
        print("rec energy", self.rec_en)
        ewald_en = self.real_en + self.corr_en + self.rec_en

        return ewald_en

    def real_energy(self, rho, grid, pos):
        L = torch.sqrt(torch.einsum("ij->i", grid.lattice ** 2))
        prec = sp.erfcinv(self.prec / 3.0)
        rmax = prec / np.sqrt(self.eta)
        N = torch.ceil(rmax / L)

        charges = []
        positions = []
        for ix in np.arange(-N[0], N[0] + 1):
            for iy in np.arange(-N[1], N[1] + 1):
                for iz in np.arange(-N[2], N[2] + 1):
                    # R=np.einsum('j,ij->i',np.array([ix,iy,iz],dtype=np.float),rho.grid.lattice.transpose())
                    R = torch.einsum(
                        "j,ij->i",
                        torch.tensor([ix, iy, iz], dtype=torch.double),
                        grid.lattice,
                    )
                    for i in np.arange(len(self.a_num)):
                        charges.append(self.a_num[i])
                        positions.append(pos[i] - R)

        esum = 0.0
        rtol = 0.001
        rcut = rmax
        eta_sqrt = np.sqrt(self.eta)
        positions = torch.stack(positions)
        charges = torch.tensor(charges)

        for i in range(len(self.a_num)):
            dists = torch.cdist(positions, pos[i].view((1, 3))).view(-1)
            index = torch.logical_and(dists < rcut, dists > rtol)
            esum += self.a_num[i] * torch.sum(
                charges[index] * sp.erfc(eta_sqrt * dists[index]) / dists[index]
            )
        esum /= 2.0

        return esum

    def corr_energy(self, rho, grid, pos):
        const = -np.sqrt(self.eta / np.pi)
        charge_sq_sum = 0
        for i in range(len(self.a_num)):
            charge_sq_sum += self.a_num[i] ** 2
        dc_term = const * charge_sq_sum

        # G=0 term of local_PP - Hartree
        const = -4.0 * np.pi * (1.0 / (4.0 * self.eta * grid.volume) / 2.0)
        charge_sum = 0
        for i in range(len(self.a_num)):
            charge_sum += self.a_num[i]
        gzero_limit = const * charge_sum ** 2

        energy = dc_term + gzero_limit

        return energy

    def rec_energy(self, rho, grid, pos):
        rec_grid = grid.get_reciprocal_grid()
        a = torch.exp(-1j * torch.einsum("lijk,l->ijk", rec_grid.coords, pos[0]))
        strf = a * self.a_num[0]
        for i in np.arange(1, len(self.a_num)):
            a = torch.exp(-1j * torch.einsum("lijk,l->ijk", rec_grid.coords, pos[i]))
            strf += a * self.a_num[i]
        strf_sq = torch.conj(strf) * strf
        gg = rec_grid.gg.clone()
        gg[0, 0, 0] = 1.0
        invgg = 1.0 / gg
        invgg[0, 0, 0] = 0.0
        gg[0, 0, 0] = 0.0
        mask = rec_grid.mask
        energy = torch.sum(
            strf_sq[mask] * torch.exp(-gg[mask] / (4.0 * self.eta)) * invgg[mask]
        )
        energy = 4.0 * np.pi * energy.real / grid.volume
        return energy

    # def get_gmax(self, grid):
    #     gg = grid.get_reciprocal().gg
    #     gmax_x = np.sqrt(np.amax(gg[:, 0, 0]))
    #     gmax_y = np.sqrt(np.amax(gg[0, :, 0]))
    #     gmax_z = np.sqrt(np.amax(gg[0, 0, :]))
    #     gmax = np.amin([gmax_x, gmax_y, gmax_z])
    #     return gmax
    #
    # def get_best_eta(precision, gmax, pos, a_num):
    #     # charge
    #     charge = 0.0
    #     chargeSquare = 0.0
    #     for i in np.arange(len(ions.pos)):
    #         charge += ions.Zval[ions.labels[i]]
    #         chargeSquare += ions.Zval[ions.labels[i]] ** 2
    #
    #     # eta
    #     eta = 1.6
    #     NotGoodEta = True
    #     while NotGoodEta:
    #         # upbound = 2.0 * charge**2 * np.sqrt ( eta / np.pi) * sp.erfc ( np.sqrt (gmax / 4.0 / eta) )
    #         upbound = (
    #             4.0 * np.pi * ions.nat * chargeSquare * np.sqrt(eta / np.pi) * sp.erfc(gmax / 2.0 * np.sqrt(1.0 / eta))
    #         )
    #         if upbound < precision:
    #             NotGoodEta = False
    #         else:
    #             eta = eta - 0.01
    #     return eta
