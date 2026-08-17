import numpy as np
import os
from scipy.interpolate import make_interp_spline, BSpline
from pyscf import gto, scf, dft
from pyscf.scf import atom_hf, ADIIS
from pyscf.dft import rks
from pyscf.data import elements
from equiv_dens.utils import base as utils
from pyscf.data.elements import NRSRHFS_CONFIGURATION
import torch
from pyscf.lib import param

def get_atm_nrks(mol, atomic_configuration=NRSRHFS_CONFIGURATION, xc='slater', grid=(120, 770)):
    '''
    # Original file see https://github.com/pyscf/pyscf/blob/master/pyscf/scf/atom_ks.py
    # Slightly modified for hirshfeld charge analysis
    Performing atomic spherically averaged DFT calculation and return mf.object containing results

    Args:
        mol: Mol object
        atomic_configuration: Non-relativistic spin-restricted spherically averaged exchange-only LDA a.k.a.
            Hartree-Fock-Slater configurations for use in atomic SAD
        xc: name of exchange correlation functional
        grid: tuple of grid specifiction, default value should be ok for second to third row elements
    '''
    basis = mol.basis
    elem_list = list([a[0]+str(n) for n,a in enumerate(mol._atom)])
    print(f'Spherically averaged atomic KS electron density for {elem_list}')

    atm_scf_result = {}
    for n,element in enumerate(mol._atom):

        elem_chrg = elements.charge(element[0])
        print('elem_chrg', elem_chrg)
        atm = gto.Mole(atom=element[0], basis=basis, spin=elem_chrg,unit="B").build()

        nao = atm.nao
        # nao == 0 for the case that no basis was assigned to an atom
        if nao == 0 or atm.nelectron == 0:  # GHOST
            raise ValueError("Ghost atom not implemented!")
        else:
            atm_ks = AtomSphericAverageRKS(atm)
            atm_ks.atomic_configuration = atomic_configuration
            atm_ks.xc = xc
            atm_ks.grids.atom_grid = grid
            my_diis_obj = ADIIS()
            my_diis_obj.space = 12
            atm_ks.diis = my_diis_obj
            atm_ks.chkfile = False
            atm_ks.run()
            atm_scf_result[element[0]+str(n)] = atm_ks
    return atm_scf_result


class AtomSphAverageRKS(rks.RKS, atom_hf.AtomSphericAverageRHF):
    def __init__(self, mol, *args, **kwargs):
        atom_hf.AtomSphericAverageRHF.__init__(self, mol)
        rks.RKS.__init__(self, mol, *args, **kwargs)

        # SAP guess is perfect for atoms
        self.init_guess = 'vsap'

    eig = atom_hf.AtomSphericAverageRHF.eig
    get_occ = atom_hf.AtomSphericAverageRHF.get_occ
    get_grad = atom_hf.AtomSphericAverageRHF.get_grad


AtomSphericAverageRKS = AtomSphAverageRKS


def free_atom_spline(mf_atom, ngrid=1000):
    '''
    spherically averaged density
    the electron density is averaged over all possible orientations of the atom. 
    Meaning any directional information is disregarded, and only the radial distance from the nucleus is considered
    Therefore only radial grid needed and rc[:,0] = r
    '''

    radial, dr = dft.radi.becke(ngrid,0)
    # needed to make it strictly monotonic increasing for the spline interpolation
    radial = -np.sort(radial)*-1
    #radial, dr = dft.radi.gauss_chebyshev(ngrid)
    coords = np.zeros((len(radial), 3))
    coords[:, 0] = radial

    ao = dft.numint.eval_ao(mf_atom.mol, coords)
    rho = dft.numint.eval_rho(mf_atom.mol, ao, mf_atom.make_rdm1())

    spl_free = spline_radial(radial, rho)
    return spl_free


def spline_radial(x, y, k=7):
    # log sampled spline interpolation
    spl = make_interp_spline(np.log(x), y, k=k)

    return spl


def _ensure_bspline_compatible(spl):
    """Recreate BSpline from (t, c, k) if loaded from pickle with incompatible scipy version.

    Pickled BSpline objects from older scipy may have t, c, k (no underscore) in __dict__
    instead of _t, _c, causing AttributeError when the current scipy's __call__ runs.
    """
    if not isinstance(spl, BSpline):
        return spl
    d = getattr(spl, '__dict__', {})
    if 't' in d and 'c' in d and 'k' in d:
        return BSpline(
            np.ascontiguousarray(np.asarray(d['t'])),
            np.ascontiguousarray(np.asarray(d['c'])),
            int(d['k']),
        )
    return spl


def eval_spline_density(spl, coords, density_grad=False):
    spl = _ensure_bspline_compatible(spl)
    x_in = torch.norm(coords, dim=-1) / param.BOHR
    x_log = np.ascontiguousarray(np.log(x_in.detach().cpu().numpy()), dtype=np.float64)
    y_out = spl(x_log)

    y_out[y_out < 0] = 0
    y_out[np.isnan(y_out)] = 0
    y_out = torch.from_numpy(y_out)

    if density_grad:
        deriv = _ensure_bspline_compatible(spl.derivative())
        spl_deriv = torch.from_numpy(deriv(x_log))
        y_deriv = spl_deriv.unsqueeze(-1) * (1 / x_in).unsqueeze(-1) * (coords / x_in.unsqueeze(-1)) / param.BOHR
        y_out = torch.cat([y_out.unsqueeze(-1), y_deriv], dim=-1)
    return y_out


class HirshfeldAnalysis:
    """
    Class for computing the weightning function, Hirshfeld-partitioned integrals
    and performing hirshfeld analysis using the molecular electron density
    
    elem -> Element
    V_free_elem -> Volume the free element occupies
    interpolate_fn_rho_free_elem -> function to interpolate the free atomic density
    """
    mol = None
    mf = None
    xc = None

    def __init__(self, mf,mol,xc):

        self.mf = mf
        self.xc = mf.xc
        self.mol = mf.mol

        keys = ["elem","V_free_elem","interpolate_fn_rho_free_elem"]
        self.result = {k : {} for k in keys}

    def run_free_atom_rho_calculation(self):
        mol = self.mol
        result = self.result

        # dict of spherical atomic RKS density 
        # TODO use becke scheme and see if difference is there -> no difference 
        mf_elems = get_atm_nrks(mol, xc=self.xc)
        for elem in mf_elems:
            mf_elem = mf_elems[elem]
            result["elem"][elem] = mf_elem
            result["interpolate_fn_rho_free_elem"][elem] = free_atom_info(mf_elem)

        return self

    def perform_hirshfeld_analysis(self, dm):
        """ 
        TODO description
        """

        result = self.result
        mol = self.mol

        grid = dft.Grids(mol)
        grid.level = 5
        grid.atom_grid = (77, 302)
        grid.build()

        ao = dft.numint.eval_ao(mol, grid.coords)
        rho = dft.numint.eval_rho(mol,ao,dm)
        Ntot = np.einsum("g,g->",rho,grid.weights)
        rho *= mol.nelectron / Ntot
        # gridpoint - r_atom_center, for centering because the atom densities are evaluated at 0 0 0
        gcoords = grid.coords
        mcords = mol.atom_coords()# /  0.52917721092
        coords_atoms = gcoords[None, :, :] - (mcords[:, None, :] )
        #coords_atoms = grid.coords[None, :, :] - (mol.atom_coords()[:, None, :] )
        masses = mol.atom_mass_list()
        #center_of_mass = masses @ mol.atom_coords() / masses.sum()
        #coords_atoms = grid.coords - center_of_mass
        #coords_atoms = coords_atoms[None, :, :]
        # euclidiean distance from each atom center to each grid point
        rad_atoms = np.linalg.norm(coords_atoms, axis=-1)
        # rho_free = rho_A(r)
        rho_free = np.empty((mol.natm, len(grid.coords)))
        for atom in range(mol.natm):
            elem = list(result["elem"].keys())[atom]
            rho_free[atom] = result["interpolate_fn_rho_free_elem"][elem](rad_atoms[atom])


        # W_A(r) = rho_A(r) / sum(rho_A(r))   sum(rho_A(r)) is rho protomol | rho(r) is molecular electron density
        # Q_A = Z_A - \integral(W_A(r) * rho(r) dr)
        # rho_eff = \integral(W_A(r) * rho(r) dr)

        rho_protomol = rho_free.sum(axis=0)
        wA = rho_free / (rho_protomol + ((rho_protomol < 1e-15) * 1e-15))
        # molecular electron density rho(r) partitioned onto every atom
        rho_eff = np.einsum("g,ig -> ig",rho , wA)

        # num electrons per atom
        elec_atm = np.einsum("ig,g->gi",rho_eff,grid.weights).sum(axis=0)
        # net charge (Q_A) on each atom 
        atomic_charges = - elec_atm + mol.atom_charges()

        masses = mol.atom_mass_list()
        center_of_mass = masses @ mol.atom_coords() / masses.sum()
        dipoles = - ( (coords_atoms-center_of_mass) * rho_eff[:, :, None] * grid.weights[:, None]).sum(axis=-2)

        result["rho_free"] = rho_free
        result["wA"] = wA
        result["hirshfeld charges"] = atomic_charges
        result["hirshfeld dipole moment"] = dipoles

        print(atomic_charges)
        return self

    def run(self,dm,fn=None):
        self.run_free_atom_rho_calculation().perform_hirshfeld_analysis(dm)
        return self


def hirshfeld_partitioning(density, free_atom_densities, atom_positions, atom_numbers, coords, coord_weights, to_bohr=True):
    if to_bohr:
        atom_positions = utils.angstrom_to_bohr(atom_positions)
        coords = utils.angstrom_to_bohr(coords)
    sum_charge = torch.sum(atom_numbers, dim=1)
    dens_int = torch.sum(density * coord_weights, dim=-1)
    # print('dens_int', dens_int)
    density *= (sum_charge / dens_int).unsqueeze(1)
    free_atom_density = torch.sum(free_atom_densities, dim=1, keepdim=True)
    # print('atom_dens_int', torch.sum(free_atom_density * coord_weights, dim=-1))
    # print('atom_dens_all_int', torch.sum(free_atom_densities * coord_weights, dim=-1))
    # gridpoint - r_atom_center, for centering because the atom densities are evaluated at 0 0 0
    # coords_atoms = grid.coords[None, :, :] - (mol.atom_coords()[:, None, :] )
    wA = free_atom_densities / (free_atom_density + ((free_atom_density < 1e-15) * 1e-15))
    dens_eff = density.unsqueeze(1) * wA

    # num electrons per atom
    elec_atm = torch.sum(dens_eff * coord_weights.unsqueeze(1), dim=-1)
    # net charge (Q_A) on each atom
    atomic_charges = - elec_atm + atom_numbers
    # print('atomic_charges', atomic_charges)
    dipoles = - torch.sum((dens_eff * coord_weights.unsqueeze(1)).unsqueeze(-1) * (coords.unsqueeze(1) - atom_positions.unsqueeze(2)), dim=-2)                                           
    r3_volume = torch.sum((dens_eff * coord_weights.unsqueeze(1)) *
                          torch.norm(coords.unsqueeze(1) - atom_positions.unsqueeze(2), dim=-1)**3, dim=-1)
    r3_volume_free = torch.sum((free_atom_densities * coord_weights.unsqueeze(1)) *
                               torch.norm(coords.unsqueeze(1) - atom_positions.unsqueeze(2), dim=-1)**3, dim=-1)
    # print('volume free', r3_volume_free)
    # print('volume eff', r3_volume)
    # print('atom_numbers', atom_numbers)
    volume_ratio = r3_volume / r3_volume_free
    #
    # print('r3_volume', r3_volume)
    # print('r3_volume_free', r3_volume_free)
    return wA, atomic_charges, dipoles, volume_ratio, r3_volume, r3_volume_free


def volume_ratios_from_expansion(atoms, expansion_model, free_atom_volumes, to_bohr=True,
                                 removed_free_atom=False):
    atoms_c = {**atoms}
    volumes = torch.zeros_like(atoms["batch_atom_numbers"]).to(torch.float32)
    volume_ratios = torch.zeros_like(atoms["batch_atom_numbers"]).to(torch.float32)
    if to_bohr:
        coords = utils.angstrom_to_bohr(atoms["coords"])
        pos = utils.angstrom_to_bohr(atoms['batch_positions'])
    else:
        coords = atoms["coords"]
        pos = atoms['batch_positions']
    for i in range(len(atoms["spherical_coeffs"])):
        z = torch.max(atoms['batch_atom_numbers'][:, i]).item()
        dens = expansion_model(atoms_c, eval_atoms=[i])["density"]
        if removed_free_atom:
            dens += atoms['atom_density_split'][:, i]
        vol1 = torch.sum((dens * atoms['coord_weights']) *
                         torch.norm(coords - pos[:, [i]], dim=-1)**3, dim=-1)
        volumes[:, i] = vol1
        volume_ratios[:, i] = vol1 / free_atom_volumes[z]

    return volume_ratios, volumes
    # r3_volume = torch.sum((dens_eff * coord_weights.unsqueeze(1)) *
    #                       torch.norm(coords.unsqueeze(1) - atom_positions.unsqueeze(2), dim=-1)**3, dim=-1)
    # print('volume eff', r3_volume)
    # print('atom_numbers', atoms['batch_atom_numbers'])
    # volume_ratio = r3_volume / r3_volume_free
    # #
    # # print('r3_volume', r3_volume)
    # # print('r3_volume_free', r3_volume_free)
    # return wA, atomic_charges, dipoles, volume_ratio
