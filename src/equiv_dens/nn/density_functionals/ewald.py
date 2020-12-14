import torch
import torch.nn as nn
from scipy import special as sp
import numpy as np


class Ewald(nn.Module):
    def __init__(
        self, a_num, PME=False, prec=1.0e-8, eta=None, verbose=0
    ):
        super().__init__()
        self.a_num = a_num
        self.PME = PME
        self.prec = prec
        self.eta = eta
        self.mask = None
        self.verbose = verbose

    def forward(self, rho, grid, pos):
        gmax = self.get_gmax(grid)
        # print('gmax', gmax)
        if self.eta is None:
            self.eta = self.get_best_eta(gmax)
        # print('self eta', self.eta)
        self.real_en = self.real_energy(rho, grid, pos)
        self.corr_en = self.corr_energy(rho, grid, pos)
        self.rec_en = self.rec_energy(rho, grid, pos)
        if self.verbose > 0:
            print("Ewald real energy", self.real_en)
            print("Ewald corr energy", self.corr_en)
            print("Ewald rec energy", self.rec_en)
        ewald_en = self.real_en + self.corr_en + self.rec_en

        return ewald_en

    def real_energy(self, rho, grid, pos):
        L = torch.sqrt(torch.einsum("ij->i", grid.lattice ** 2))
        prec = sp.erfcinv(self.prec / 3.0)
        rmax = prec / torch.sqrt(self.eta)
        N = torch.ceil(rmax / L)

        charges = []
        positions = []
        for ix in torch.arange(-N[0], N[0] + 1):
            for iy in torch.arange(-N[1], N[1] + 1):
                for iz in torch.arange(-N[2], N[2] + 1):
                    # R=np.einsum('j,ij->i',np.array([ix,iy,iz],dtype=np.float),rho.grid.lattice.transpose())
                    R = torch.einsum(
                        "j,ij->i",
                        torch.tensor([ix, iy, iz]).to(rho),
                        grid.lattice,
                    )
                    for i in np.arange(len(self.a_num)):
                        charges.append(self.a_num[i])
                        positions.append(pos[i] - R)

        esum = 0.0
        rtol = 0.01
        rcut = rmax
        eta_sqrt = torch.sqrt(self.eta)
        positions = torch.stack(positions)
        charges = torch.tensor(charges).to(rho)
        for i in range(len(self.a_num)):
            dists = torch.cdist(positions, pos[i].view((1, 3))).view(-1)
            index = torch.logical_and(dists < rcut, dists > rtol)
            esum += self.a_num[i] * torch.sum(
                charges[index] * torch.erfc(eta_sqrt * dists[index]) / dists[index]
            )
        esum /= 2.0

        return esum

    def corr_energy(self, rho, grid, pos):
        const = -torch.sqrt(self.eta / np.pi)
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

    def get_gmax(self, grid):
        rec_grid = grid.get_reciprocal_grid()
        gg = rec_grid.gg
        gmax_x = torch.sqrt(torch.amax(gg[:, 0, 0]))
        gmax_y = torch.sqrt(torch.amax(gg[0, :, 0]))
        gmax_z = torch.sqrt(torch.amax(gg[0, 0, :]))
        gmax = np.amax([gmax_x, gmax_y, gmax_z])
        return gmax

    def get_best_eta(self, gmax):
        # charge
        charge = 0.0
        charge_sq = 0.0
        for i in np.arange(len(self.a_num)):
            charge += self.a_num[i]
            charge_sq += self.a_num[i] ** 2

        # eta
        eta = torch.tensor(1.6).to(gmax)
        NotGoodEta = True
        while NotGoodEta:
            # upbound = 2.0 * charge**2 * np.sqrt ( eta / np.pi) * sp.erfc ( np.sqrt (gmax / 4.0 / eta) )
            upbound = (
                4.0 * np.pi * len(self.a_num) * charge_sq * torch.sqrt(eta / np.pi) * torch.erfc(gmax / 2.0 * torch.sqrt(1.0 / eta))
            )
            if upbound < self.prec:
                NotGoodEta = False
            else:
                eta = eta - 0.01
        return eta
