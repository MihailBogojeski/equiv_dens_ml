import numpy as np
import torch
import torch.nn as nn
from pyscf import gto

from equiv_dens.utils import orbitals
from equiv_dens.utils import base as utils

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False


class DipoleMomentCalc(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, atoms, center_coordinates=True):
        density = atoms['density']
        positive_dens = (density >= 0).to(density)
        density = density * positive_dens
        n_electrons = orbitals.get_n_electrons(atoms['batch_atom_numbers'])
        dens_sum = torch.sum(density * atoms['coord_weights'], dim=1, keepdim=True)
        dens_sum = torch.clamp(dens_sum, min=1e-30)  # avoid inf scaling
        scaling_factor = n_electrons / dens_sum
        density = density * scaling_factor
        if center_coordinates:
            center_of_mass = torch.sum(atoms['batch_positions'] * atoms['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)\
                             / torch.sum(atoms['batch_atom_numbers'].unsqueeze(-1), dim=1, keepdim=True)
        else:
            center_of_mass = torch.zeros(atoms['batch_positions'].shape[0], 1, atoms['batch_positions'].shape[-1]).to(atoms['batch_positions'])
        positive_dipole_moment = torch.sum((atoms['batch_positions'] - center_of_mass) * atoms['batch_atom_numbers'].unsqueeze(-1), dim=1)
        # positive_dipole_moment = torch.sum(atoms['batch_positions'] * atoms['batch_atom_numbers'].unsqueeze(-1), dim=1)
        # print('positive_dipole_moment', positive_dipole_moment)
        weighted_dens = density * atoms['coord_weights']
        # # print('weighted dens shape', weighted_dens.shape)
        # # print('atom numbers shape', atoms['atom_numbers'].shape)
        # weighted_dens = weighted_dens / torch.sum(weighted_dens, dim=1, keepdim=True) * \
        #                 torch.sum(atoms['batch_atom_numbers'], dim=1, keepdim=True)

        negative_moment = weighted_dens.unsqueeze(-1) * (atoms['coords'] - center_of_mass)
        # negative_moment = weighted_dens.unsqueeze(-1) * atoms['coords']
        negative_dipole_moment = torch.sum(negative_moment, dim=1)
        # print('negative_dipole_moment', negative_dipole_moment)

        atoms['dipole_moment'] = positive_dipole_moment - negative_dipole_moment
        return atoms


class DipoleMomentIntorCalc(nn.Module):
    """Dipole from density coefficients via int1e_r (PySCF or dipole_gpu).

    Nuclear term plus -<r>_elec. Set remove_atom_density to subtract
    free-atom contributions first. For MD trajectories, cache_integrals
    skips rebuilding int1e_r when atomic positions move less than
    integral_cache_threshold (Angstrom). dipole_every_n_steps reuses the
    last electronic dipole on intermediate steps.
    """

    def __init__(self, orbital_basis, remove_atom_density=False, cache_integrals=True,
                 integral_cache_threshold=0.5, dipole_every_n_steps=1):
        super().__init__()
        self.orbital_basis = orbital_basis
        self.remove_atom_density = remove_atom_density
        self.cache_integrals = cache_integrals
        self.integral_cache_threshold = integral_cache_threshold
        self.dipole_every_n_steps = max(1, int(dipole_every_n_steps))
        self._cached_positions = None
        self._cached_int1e_r = []
        self._cached_el_dip = None  # Reused when dipole_every_n_steps > 1
        self._dipole_call_count = 0

    def _can_use_cached_integrals(self, coords):
        """Return True if cached integrals can be reused (small displacement)."""
        if not self.cache_integrals or self._cached_positions is None:
            return False
        if self._cached_positions.shape != coords.shape:
            return False
        displacement = torch.max(torch.abs(coords - self._cached_positions))
        return displacement.item() < self.integral_cache_threshold

    def forward(self, atoms):
        df_coeffs_ml = orbitals.coeffs_dict_to_vector(atoms, self.orbital_basis, atoms['batch_atom_numbers'],
                                                      radial_coeffs=False, convert_to_pyscf=True)['spherical_coeffs']
        charges = atoms['batch_atom_numbers']
        coords = atoms['batch_positions']

        nucl_dip = torch.sum(charges.unsqueeze(-1) * coords, dim=1)
        batch_size = atoms['batch_atom_numbers'].shape[0]

        self._dipole_call_count += 1
        compute_integrals = (
            self.dipole_every_n_steps <= 1
            or self._dipole_call_count % self.dipole_every_n_steps == 1
            or self._cached_el_dip is None
        )

        if not compute_integrals and self._cached_el_dip is not None:
            dens_dip = self._cached_el_dip.to(coords)
        else:
            batch_dens_dip = []
            use_cached = self._can_use_cached_integrals(coords) and len(self._cached_int1e_r) == batch_size

            for i in range(batch_size):
                if use_cached and i < len(self._cached_int1e_r):
                    int1e_r = self._cached_int1e_r[i].to(coords)
                else:
                    auxmol_ml = orbitals.ml_basis_to_auxmol(atoms, i, skip_zero=False)
                    helper_mol = orbitals.build_1c1e_helper_mol(auxmol_ml)
                    intor_idx = [
                        auxmol_ml.bas_atom(ibas)
                        for ibas in range(auxmol_ml.nbas) for _ in range(auxmol_ml.bas_angular(ibas) * 2 + 1)
                    ]
                    # PySCF libcint (CPU) is ~10k× faster than CuPy implementation for small molecules
                    int1e_r_arr = gto.mole.intor_cross("int1e_r", helper_mol, auxmol_ml)
                    int1e_r_arr = int1e_r_arr[:, intor_idx, range(auxmol_ml.nao)]
                    int1e_r = torch.from_numpy(int1e_r_arr).to(coords.dtype).to(coords.device)
                    int1e_r = utils.bohr_to_angstrom(int1e_r)
                    if self.cache_integrals:
                        if i >= len(self._cached_int1e_r):
                            self._cached_int1e_r.append(int1e_r.detach())
                        else:
                            self._cached_int1e_r[i] = int1e_r.detach()

                el_dip = torch.einsum('ji,i->j', int1e_r, df_coeffs_ml[i])
                batch_dens_dip.append(el_dip)

            if self.cache_integrals and not use_cached:
                self._cached_positions = coords.detach().clone()

            dens_dip = torch.stack(batch_dens_dip, dim=0)
            if self.dipole_every_n_steps > 1:
                self._cached_el_dip = dens_dip.detach().clone()

        if self.remove_atom_density:
            atoms = orbitals.intor_dipole_moment_free_atom(atoms)
            dens_dip += atoms['atom_dipole_moment']

        atoms['dipole_moment'] = nucl_dip - dens_dip

        return atoms
