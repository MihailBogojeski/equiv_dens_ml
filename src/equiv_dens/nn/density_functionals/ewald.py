import torch
import torch.nn as nn
from scipy import special as sp
import numpy as np


class Ewald(nn.Module):
    def __init__(
        self, a_num, PME=False, grid_shape=np.array([20, 20, 20]), prec=1.0e-8, eta=0.73
    ):
        super().__init__()
        self.a_num = a_num
        self.PME = PME
        self.prec = prec
        self.eta = eta
        self.grid_shape = grid_shape
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
        lattice = torch.max(grid, dim=0)[0] - torch.min(grid, dim=0)[0]
        lattice = torch.diag(lattice)
        print(torch.max(grid, dim=0)[0])
        print(torch.min(grid, dim=0)[0])
        L = torch.sqrt(torch.einsum("ij->i", lattice ** 2))
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
                        lattice,
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
        lattice = torch.max(grid, dim=0)[0] - torch.min(grid, dim=0)[0]
        grid_volume = torch.prod(lattice)
        const = -4.0 * np.pi * (1.0 / (4.0 * self.eta * grid_volume) / 2.0)
        charge_sum = 0
        for i in range(len(self.a_num)):
            charge_sum += self.a_num[i]
        gzero_limit = const * charge_sum ** 2

        energy = dc_term + gzero_limit

        return energy

    def rec_energy(self, rho, grid, pos):
        lattice = torch.max(grid, dim=0)[0] - torch.min(grid, dim=0)[0]
        grid_volume = torch.prod(lattice)
        rec_grid = self.get_reciprocal_grid(grid)
        gg = np.einsum("lijk,lijk->ijk", rec_grid, rec_grid)

        a = np.exp(-1j * np.einsum("lijk,l->ijk", rec_grid, pos[0]))
        strf = a * self.a_num[0]
        for i in np.arange(1, len(self.a_num)):
            a = np.exp(-1j * np.einsum("lijk,l->ijk", rec_grid, pos[i]))
            strf += a * self.a_num[i]
        strf_sq = np.conjugate(strf) * strf
        gg[0, 0, 0] = 1.0
        invgg = 1.0 / gg
        invgg[0, 0, 0] = 0.0
        gg[0, 0, 0] = 0.0
        mask = self.get_mask(rec_grid)
        energy = np.sum(
            strf_sq[mask] * np.exp(-gg[mask] / (4.0 * self.eta)) * invgg[mask]
        )
        energy = 4.0 * np.pi * energy.real / grid_volume
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

    def get_reciprocal_grid(self, grid):
        lattice = torch.max(grid, dim=0)[0] - torch.min(grid, dim=0)[0]
        lattice = np.array(torch.diag(lattice))
        print('lattice', lattice)
        fac = 2 * np.pi
        bg = fac * np.linalg.inv(lattice)
        reciprocal_lat = bg.T
        ax = []
        for i in range(3):
            dd = 1 / self.grid_shape[i]
            if i == 2:
                ax.append(np.fft.rfftfreq(self.grid_shape[i], d=dd))
            else:
                freq = np.fft.fftfreq(self.grid_shape[i], d=dd)

                ax.append(freq)
        S0, S1, S2 = np.meshgrid(ax[0], ax[1], ax[2], indexing="ij")

        # S_cart = s2r(S, self)
        # S_cart = np.asarray([S2, S1, S0])
        S_cart = np.asarray([S0, S1, S2])
        S_cart = np.einsum("j...,kj->k...", S_cart, reciprocal_lat)

        return S_cart

    def get_mask(self, grid):
        if self.mask is None:
            grid_shape = np.array(grid.shape[1:])
            # Dnr = nr[:3]//2
            # Dmod = nr[:3]%2
            # mask = np.ones((nr[0], nr[1], Dnr[2]+1), dtype = bool)
            Dnr = grid_shape // 2
            Dmod = grid_shape % 2
            mask = np.ones(grid_shape, dtype=bool)
            mask[:, :, Dnr[2] + 1 :] = False

            mask[0, 0, 0] = False
            mask[0, Dnr[1] + 1 :, 0] = False
            mask[Dnr[0] + 1 :, :, 0] = False
            if Dmod[2] == 0:
                mask[0, 0, Dnr[2]] = False
                mask[0, Dnr[1] + 1 :, Dnr[2]] = False
                mask[Dnr[0] + 1 :, :, Dnr[2]] = False
                if Dmod[1] == 0:
                    mask[0, Dnr[1], Dnr[2]] = False
                if Dmod[0] == 0:
                    mask[Dnr[0], 0, Dnr[2]] = False
                    mask[Dnr[0], Dnr[1] + 1 :, Dnr[2]] = False
            if Dmod[0] == 0:
                mask[Dnr[0], Dnr[1] + 1 :, 0] = False
                if Dmod[1] == 0:
                    mask[Dnr[0], Dnr[1], 0] = False
            if Dmod[1] == 0:
                mask[0, Dnr[1], 0] = False
            if all(Dmod == 0):
                mask[Dnr[0], Dnr[1], Dnr[2]] = False
            self.mask = mask
        return self.mask
