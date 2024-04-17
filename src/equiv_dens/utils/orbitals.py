import numpy as np
import equiv_dens.utils.base as utils
import torch
import scipy as sp
from pyscf.dft import numint
from pyscf.lib import param
from pyscf import gto, df, lib, dft
from pyscf.scf import hf
import scipy
# import time
from equiv_dens.utils.hirshfeld_analysis import eval_spline_density

hf.MUTE_CHKFILE = True

pyscf_gto_factor = 2*np.sqrt(np.pi)

#
# def combine_orbitals(orbitals, order_max):
#     radial_spec = [None] * len(orbitals)
#     spherical_spec = [None] * len(orbitals)
#     radial_counts = [None] * len(orbitals)
#     for i in range(len(orbitals)):
#         radial_L_count = [0] * (order_max + 2)
#         spherical_L_count = [0] * (order_max + 2)
#         radial_spec[i] = []
#         spherical_spec[i] = []
#         radial_counts[i] = [[] for i in range(order_max + 2)]
#         # print('density L count len', len(density_L_count))
#         z = orbitals[i][0][0]
#         for j in range(len(orbitals[i])):
#             orb = orbitals[i][j]
#             L = orb[2]
#             radial_L_count[L] += orb[1]
#             spherical_L_count[L] += 1
#             radial_counts[i][L].append(orb[1])
#         for L, c in enumerate(radial_L_count):
#             if c == 0:
#                 continue
#             radial_spec[i].append((z, c, L))
#         for L, c in enumerate(spherical_L_count):
#             if c == 0:
#                 continue
#             spherical_spec[i].append((z, c, L))
#     return spherical_spec, radial_spec, radial_counts


def combine_orbital_basis(orbital_basis, order_max):
    radial_spec = {}
    spherical_spec = {}
    radial_counts = {}
    for z in orbital_basis.keys():
        radial_L_count = [0] * (order_max + 2)
        spherical_L_count = [0] * (order_max + 2)
        radial_spec[z] = []
        spherical_spec[z] = []
        radial_counts[z] = [[] for i in range(order_max + 2)]
        # print('density L count len', len(density_L_count))
        for j in range(len(orbital_basis[z])):
            orb = orbital_basis[z][j]
            L = orb[2]
            radial_L_count[L] += orb[1]
            spherical_L_count[L] += 1
            radial_counts[z][L].append(orb[1])
        for L, c in enumerate(radial_L_count):
            if c == 0:
                continue
            radial_spec[z].append((z, c, L))
        for L, c in enumerate(spherical_L_count):
            if c == 0:
                continue
            spherical_spec[z].append((z, c, L))
    return spherical_spec, radial_spec, radial_counts


def get_max_order(orbital_basis, per_atom=False):
    order_max = {}
    for key in orbital_basis.keys():
        for z, _, l in orbital_basis[key]:
            if z in order_max.keys():
                if l > order_max[z]:
                    order_max[z] = l
            else:
                order_max[z] = l
    if per_atom:
        return order_max
    else:
        return max(order_max.values())


def get_n_electrons(atom_numbers):
    return torch.sum(atom_numbers, -1, keepdim=True)


# def get_n_electrons(orbitals):
#     n_electrons = 0
#     for i in range(len(orbitals)):
#         n_electrons += orbitals[i][0][0]
#     return n_electrons
#
#
def gaussian_rbf(r, width, scale, order, normalize=False):
    # print('scale shape', scale.shape)
    # print('scale shape', scale.shape)
    # print('width', width)
    # print('r shape', r.shape)
    # print('L', order)
    if normalize:
        # scale_calc = scale * (width**(3 / 2)) / (np.pi**(3 / 2)) * utils.to_angstrom**3  
        # scale_calc = scale * (width**(3/2)) / (np.pi**(3 / 2))
        scale_calc = scale * gto_norm(order, width)
    else:
        scale_calc = scale / pyscf_gto_factor
    # print('order', order)
    # print('width', width)
    # print('gto_norm', 1/gto_norm(order, width))
    # print('pyscf gto factor', pyscf_gto_factor)
    # print('scale calc', scale_calc)
    # print('scale calc * gto norm', scale_calc/gto_norm(order, width))
    r_bohr = r * utils.to_bohr
    rbf = scale_calc * r_bohr**(order) * torch.exp(-width * (r_bohr)**2)
    # rbf = scale_calc * torch.exp(-width * (r_bohr)**2)
    return torch.sum(rbf, dim=-2, keepdim=True)


def gto_norm(order, width):
    # norm_factor = (width**(3 / 2)) / (np.pi**(3 / 2)) * utils.to_angstrom**3  
    # norm_factor = (width**(3/2)) / (np.pi**(3 / 2))

    n1 = (order + 3) / 2
    n2 = order + 1
    norm_factor = (sp.special.gamma(order/2 + 1) * 2**(order) * (width**(n1))) / ((np.pi**(3 / 2)) * sp.special.gamma(n2 + 1))
    return norm_factor


def gto_norm_pyscf(order, width):
        if np.all(order >= 0):
            #f = 2**(2*l+3) * math.factorial(l+1) * (2*expnt)**(l+1.5) \
            #        / (math.factorial(2*l+2) * math.sqrt(math.pi))
            #return math.sqrt(f)
            n = (order * 2) + 2
            alpha = width * 2
            n1 = (n + 1) * .5
            gaussian_int = sp.special.gamma(n1) / (2. * alpha**n1)
            return 1/np.sqrt(gaussian_int)
        else:
            raise ValueError('l should be >= 0')


def get_invariant_features(coeffs, permutational_invariance=True, keep_dims=False, radial_coeffs=True):
    # coeffs = model(R=R)
    sph_coeffs, rad_scale, rad_width = coeffs_dict_to_tensors(coeffs, radial_coeffs=radial_coeffs)
    for L in range(len(sph_coeffs)):
        # avoid calculating norm for zero vectors
        sph_zeros = (torch.sum(torch.abs(sph_coeffs[L]), dim=-2, keepdim=True) == 0).to(sph_coeffs[L])
        sph_nz = (1 - sph_zeros).squeeze()
        sph_coeffs_mod = sph_coeffs[L] + sph_zeros
        sph_coeffs[L] = torch.norm(sph_coeffs_mod, dim=-2) * sph_nz
        if radial_coeffs:
            scale_zeros = (torch.sum(torch.abs(rad_scale[L]), dim=-2, keepdim=True) == 0).to(sph_coeffs[L])
            scale_nz = (1 - scale_zeros).squeeze()
            width_zeros = (torch.sum(torch.abs(rad_width[L]), dim=-2, keepdim=True) == 0).to(sph_coeffs[L])
            width_nz = (1 - width_zeros).squeeze()
            rad_scale_mod = rad_scale[L] + scale_zeros
            rad_width_mod = rad_width[L] + width_zeros
            rad_scale[L] = torch.norm(rad_scale_mod, dim=-2) * scale_nz
            rad_width[L] = torch.norm(rad_width_mod, dim=-2) * width_nz

    all_sph = torch.cat(sph_coeffs, dim=-1)
    if radial_coeffs:
        all_width = torch.cat(rad_width, dim=-1)
        all_scale = torch.cat(rad_scale, dim=-1)
    else:
        all_width = torch.tensor([])
        all_scale = torch.tensor([])

    invariant_feats = torch.cat([all_sph, all_scale, all_width], dim=-1)

    if permutational_invariance:
        invariant_feats = invariant_feats.sum(dim=1)
    if keep_dims:
        invariant_feats = invariant_feats.squeeze(2)
    else:
        invariant_feats = invariant_feats.squeeze()
    return invariant_feats


def coeffs_dict_to_tensors(coeffs, radial_coeffs=True):
    sph_coeffs = coeffs['spherical_coeffs']
    rad_width = coeffs['radial_width']
    rad_scale = coeffs['radial_scale']
    if 'sph_dict' in coeffs:
        sph_dict = coeffs['sph_dict']
    else:
        sph_dict = coeffs['L_dict']

    max_order = 0
    for i in range(len(sph_coeffs)):
        for key in sph_coeffs[i].keys():
            if key[1] > max_order:
                max_order = key[1]

    batch_size = 0
    first_key = None
    first_idx = 0
    for i in range(len(sph_coeffs)):
        if sph_coeffs[i]:
            first_key = list(sph_coeffs[i].keys())[0]
            batch_size = sph_coeffs[i][first_key].shape[0]
            first_idx = i
            break

    max_num_coeffs = [0] * (max_order + 1)
    max_num_radial = [0] * (max_order + 1)
    # print('max order', model.order_max)
    for i in range(len(sph_coeffs)):
        for key in sph_coeffs[i].keys():
            L = key[1]
            # print(i, L)
            if sph_dict[key][-1] >= max_num_coeffs[L]:
                max_num_coeffs[L] = sph_dict[key][-1] + 1
            if rad_scale[i][key].shape[-2] > max_num_radial[L]:
                max_num_radial[L] = rad_scale[i][key].shape[-2]
    # print('max num coeffs', max_num_coeffs)
    # print('max num radial', max_num_radial)

    all_sph = [[torch.zeros([batch_size, 1, (2 * i) + 1, max_num_coeffs[i]]).to(sph_coeffs[first_idx][first_key])
               for _ in range(len(sph_coeffs))]
               for i in range(max_order + 1)]
    if radial_coeffs:
        all_width = [[torch.zeros([batch_size, 1, max_num_radial[i], max_num_coeffs[i]]).to(sph_coeffs[first_idx][first_key])
                     for _ in range(len(sph_coeffs))]
                     for i in range(max_order + 1)]
        all_scale = [[torch.zeros([batch_size, 1, max_num_radial[i], max_num_coeffs[i]]).to(sph_coeffs[first_idx][first_key])
                     for _ in range(len(sph_coeffs))]
                     for i in range(max_order + 1)]
    else:
        all_scale = []
        all_width = []

    for i in range(len(sph_coeffs)):
        for key in sph_coeffs[i].keys():
            L = key[1]
            inds = sph_dict[key]
            # print('i, ', i, ', key', key)
            # print('padding', padding)
            all_sph[L][i][..., inds] = sph_coeffs[i][key]
            if radial_coeffs:
                all_width[L][i][..., inds] = rad_width[i][key]
                all_scale[L][i][..., inds] = rad_scale[i][key]

    for L in range(max_order + 1):
        all_sph[L] = torch.cat(all_sph[L], dim=1)
        if radial_coeffs:
            all_width[L] = torch.cat(all_width[L], dim=1)
            all_scale[L] = torch.cat(all_scale[L], dim=1)

    return all_sph, all_scale, all_width


def coeffs_dict_to_vector(coeffs, orbital_basis, a_num, radial_coeffs=True,
                          coeff_weighting=False, convert_to_pyscf=True):
    relevant_keys = ['spherical_coeffs']
    # sph_coeffs = coeffs['spherical_coeffs']
    if radial_coeffs:
        relevant_keys.extend(['radial_width', 'radial_scale'])
        # rad_width = coeffs['radial_width']
        # rad_scale = coeffs['radial_scale']
    if coeff_weighting:
        relevant_keys.append('coeff_weights')
        # coeff_weights = coeffs['coeff_weights']

    if convert_to_pyscf:
        new_coeffs = {key: coeffs[key] for key in coeffs.keys()}
        new_coeffs['spherical_coeffs'] = []
        for i in range(len(coeffs['spherical_coeffs'])):
            new_coeffs['spherical_coeffs'].append({})
            for key in coeffs['spherical_coeffs'][i].keys():
                if key[1] == 1:
                    new_coeffs['spherical_coeffs'][i][key] = coeffs['spherical_coeffs'][i][key]
                    new_coeffs['spherical_coeffs'][i][key] = new_coeffs['spherical_coeffs'][i][key][:, :, [2, 0, 1], :]
                else:
                    new_coeffs['spherical_coeffs'][i][key] = coeffs['spherical_coeffs'][i][key]
                if key[1] % 2 == 1:
                    new_coeffs['spherical_coeffs'][i][key] = -1 * new_coeffs['spherical_coeffs'][i][key]
        coeffs = new_coeffs

    all_coeffs = {key: None for key in relevant_keys}
    # all_sph = None
    # all_scale = None
    # all_width = None
    # all_weights = None

    for i, z in enumerate(torch.max(a_num, dim=0)[0]):
        orb_ind = {}
        z = int(z)
        basis = orbital_basis[z]
        for orb in basis:
            L = orb[2]
            key = (z, L)
            if key not in orb_ind.keys():
                orb_ind[key] = 0
            if all_coeffs['spherical_coeffs'] is None:
                for coeff_type in relevant_keys:
                    all_coeffs[coeff_type] = 1 * coeffs[coeff_type][i][key][..., orb_ind[key]]
            else:
                for coeff_type in relevant_keys:
                    all_coeffs[coeff_type] = torch.cat([all_coeffs[coeff_type],
                                                        coeffs[coeff_type][i][key][..., orb_ind[key]]], dim=2)
            orb_ind[key] += 1

    vector_coeffs = {}
    for coeff_type in relevant_keys:
        vector_coeffs[coeff_type] = all_coeffs[coeff_type].squeeze(1)

    return vector_coeffs


# def coeffs_dict_to_vector(coeffs, orbital_basis, a_num, radial_coeffs=True, coeff_weighting=False):
#     # relevant_keys = ['spherical_coeffs']
#     sph_coeffs = coeffs['spherical_coeffs']
#     if radial_coeffs:
#         # relevant_keys.extend(['radial_width', 'radial_scale'])
#         rad_width = coeffs['radial_width']
#         rad_scale = coeffs['radial_scale']
#     if coeff_weighting:
#         coeff_weights = coeffs['coeff_weights']
#         # relevant_keys.append(coeff_weights)
#
#     # all_coeffs = {key: None for key in relevant_keys}
#     all_sph = None
#     all_scale = None
#     all_width = None
#     all_weights = None
#     for i, z in enumerate(torch.max(a_num, dim=0)[0]):
#         orb_ind = {}
#         z = int(z)
#         basis = orbital_basis[z]
#         for orb in basis:
#             L = orb[2]
#             key = (z, L)
#             if key not in orb_ind.keys():
#                 orb_ind[key] = 0
#             if all_sph is None:
#                 all_sph = 1 * sph_coeffs[i][key][..., orb_ind[key]]
#                 if radial_coeffs:
#                     all_scale = 1 * rad_scale[i][key][..., orb_ind[key]]
#                     all_width = 1 * rad_width[i][key][..., orb_ind[key]]
#                 if coeff_weighting:
#                     all_weights = 1 * coeff_weights[i][key][..., orb_ind[key]]
#             else:
#                 all_sph = torch.cat([all_sph, sph_coeffs[i][key][..., orb_ind[key]]], dim=2)
#                 if radial_coeffs:
#                     all_scale = torch.cat([all_scale, rad_scale[i][key][..., orb_ind[key]]], dim=2)
#                     all_width = torch.cat([all_width, rad_width[i][key][..., orb_ind[key]]], dim=2)
#                 if coeff_weighting:
#                     all_weights = torch.cat([all_weights, coeff_weights[i][key][..., orb_ind[key]]], dim=2)
#             orb_ind[key] += 1
#
#     vector_coeffs = {}
#     vector_coeffs['spherical_coeffs'] = all_sph.squeeze(1)
#     if radial_coeffs:
#         vector_coeffs['radial_width'] = all_width.squeeze(1)
#         vector_coeffs['radial_scale'] = all_scale.squeeze(1)
#     if coeff_weighting:
#         vector_coeffs['coeff_weights'] = all_weights.squeeze(1)
#
#     return vector_coeffs

def radial_basis_to_vector(a_num, orbital_basis, radial_basis):
    """
    Uses a radial basis definition to construct two vectors of scale and width radial coefficients for a given set of atomistic systems.

    Args:
        a_num (torch.Tensor): Tensor of atomic numbers.
        orbital_basis (dict): Dictionary describing the orbital basis for a set of atom types.
        radial_basis (dict): Dictionary containing the radial coefficients of the basis for a set of atom types.
    """
    # print('orbital basis', orbital_basis)
    # print('radial basis', radial_basis)

    radial_widths = []
    radial_scales = []

    all_nums = torch.max(a_num, dim=0)[0]
    for z in all_nums:
        z = int(z)
        basis_z = radial_basis[z]
        for i in range(len(basis_z)):
            radial_widths.append(radial_basis[z][i][0])
            radial_scales.append(radial_basis[z][i][1])

    radial_widths = np.concatenate(radial_widths)
    radial_scales = np.concatenate(radial_scales)
    radial_widths = torch.tile(torch.from_numpy(radial_widths), (a_num.shape[0], 1))
    radial_scales = torch.tile(torch.from_numpy(radial_scales), (a_num.shape[0], 1))

    return radial_widths, radial_scales

def vector_to_coeffs_dict(coeffs, orbital_basis, a_num, radial_coeffs=True,
                          convert_to_equiv_dens=True, radial_basis=None):
    vec_sph = coeffs['spherical_coeffs']
    dict_sph = []
    if radial_coeffs:
        if 'radial_width' not in coeffs.keys():
            if radial_basis is None:
                raise ValueError('No radial coefficients or radial basis definition was provided!')
            else:
                vec_width, vec_scale = radial_basis_to_vector(a_num, orbital_basis, radial_basis)
        else:
            vec_width = coeffs['radial_width']
            vec_scale = coeffs['radial_scale']
        dict_width = []
        dict_scale = []
    print("starting vector to coeffs dict")
    sph_count = 0
    rad_count = 0
    for i, z in enumerate(torch.max(a_num, dim=0)[0]):
        dict_sph.append({})
        if radial_coeffs:
            dict_width.append({})
            dict_scale.append({})
        z = int(z)
        basis = orbital_basis[z]
        for orb in basis:
            L = orb[2]
            key = (z, L)
            step = 2*L + 1
            if key not in dict_sph[i].keys():
                dict_sph[i][key] = vec_sph[:, sph_count:sph_count + step].unsqueeze(1).unsqueeze(-1)
                sph_count += step
                if radial_coeffs:
                    dict_width[i][key] = vec_width[:, [rad_count]].unsqueeze(1).unsqueeze(-1)
                    dict_scale[i][key] = vec_scale[:, [rad_count]].unsqueeze(1).unsqueeze(-1)
                    rad_count += 1
            else:
                new_sph = vec_sph[:, sph_count:sph_count + step].unsqueeze(1).unsqueeze(-1)
                dict_sph[i][key] = torch.cat([dict_sph[i][key], new_sph], dim=-1)
                sph_count += step
                if radial_coeffs:
                    new_scale = vec_scale[:, [rad_count]].unsqueeze(1).unsqueeze(-1)
                    new_width = vec_width[:, [rad_count]].unsqueeze(1).unsqueeze(-1)
                    dict_width[i][key] = torch.cat([dict_width[i][key], new_width], dim=-1)
                    dict_scale[i][key] = torch.cat([dict_scale[i][key], new_scale], dim=-1)
                    rad_count += 1
    dict_coeffs = {}
    dict_coeffs['spherical_coeffs'] = dict_sph
    if radial_coeffs:
        dict_coeffs['radial_width'] = dict_width
        dict_coeffs['radial_scale'] = dict_scale

    if convert_to_equiv_dens:
        for i in range(len(dict_coeffs['spherical_coeffs'])):
            for key in dict_coeffs['spherical_coeffs'][i].keys():
                if key[1] == 1:
                    dict_coeffs['spherical_coeffs'][i][key] = dict_coeffs['spherical_coeffs'][i][key][:, :, [1, 2, 0], :]
                else:
                    dict_coeffs['spherical_coeffs'][i][key] = dict_coeffs['spherical_coeffs'][i][key]
                if key[1] % 2 == 1:
                    dict_coeffs['spherical_coeffs'][i][key] = -1 * dict_coeffs['spherical_coeffs'][i][key]

    return dict_coeffs


def orbitals_from_hamiltonian(hamiltonians, overlaps):
    orbital_coeffs = []
    ens = []
    for i in range(hamiltonians.shape[0]):
        en, coefs = sp.linalg.eigh(a=hamiltonians[i], b=overlaps[i])
        orbital_coeffs.append(coefs)
        ens.append(en)
    return orbital_coeffs, ens


def parse_orbitals(orbitals, atom_types, basis_def):
    split_orbitals = []
    count = 0
    for at in atom_types:
        basis_at = basis_def[at]
        for orb in basis_at:
            L = orb[2]
            split_orbitals.append(orbitals[:, count:(count + (2 * L) + 1)])
            count += (2 * L) + 1
    return split_orbitals

def split_df_coeffs(atom, df_coeffs, basis_size):
    atom_numbers = []
    for at in atom:
        if isinstance(at[0], str):
            atom_numbers.append(utils.symbols_to_numbers([at[0]])[0])
        else:
            atom_numbers.append(at[0])
    df_coeffs_split = []
    curr_idx = 0
    for an in atom_numbers:
        df_coeffs_split.append((an, df_coeffs[curr_idx:curr_idx + basis_size[an]]))
        curr_idx += basis_size[an]

    return df_coeffs_split

def calc_dipole_moment(atoms, center_coordinates=True, normalize_density=True, positive_density=True,
                       density=None):
    if density is None:
        density = atoms['density']
    if positive_density:
        density = torch.clamp(density, min=0)
    if normalize_density:
        n_electrons = get_n_electrons(atoms['batch_atom_numbers'])
        scaling_factor = n_electrons / torch.sum(density * atoms['coord_weights'], dim=1, keepdim=True)
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


def sample_density_base(mols, coords, coeffs, scale_coords=False, projected=False):
    if scale_coords:
        coords = coords / param.BOHR
    if coords.shape[0] != len(mols):
        raise ValueError('Batch dimension of coordinates (' + str(coords.shape[0]) +
                         ') must match number of molecules (' + str(len(mols)) + ')')
    dens = torch.zeros((coords.shape[0], coords.shape[1]))
    for i in range(len(mols)):
        mol = mols[i]
        if not mol._built:
            mol.build()
        ao = numint.eval_ao(mol, coords[i])
        if projected:
            rho = _expand_pyscf_projected_density(mol, ao, coeffs[i])
        else:
            rho = _expand_pyscf_density(mol, ao, coeffs[i])
        dens[i, :] = torch.from_numpy(rho)
    return dens


def _expand_pyscf_density(mol, ao, coeffs):
    if coeffs['mo_occ'].ndim > 1:
        rho = 0
        for j in range(coeffs['mo_occ'].shape[0]):
            rho += numint.eval_rho2(mol, ao, mo_occ=coeffs['mo_occ'][j],
                                    mo_coeff=coeffs['mo_coeff'][j])
    else:
        rho = numint.eval_rho2(mol, ao, **coeffs)

    return rho


def _expand_pyscf_projected_density(mol, ao, coeffs):
    if coeffs.ndim > 1:
        rho = 0
        for j in range(coeffs.shape[0]):
            rho += np.einsum('ij,j->i', ao, coeffs[j])
    else:
        rho = np.einsum('ij,j->i', ao, coeffs)

    return rho


def sample_density(atoms, mo_coeff, mo_occ, basis='augccpvdz'):
    scaled_sample_coords = atoms['coords'].detach().cpu().numpy() / param.BOHR  # convert Angstrom grid to Bohr

    mol = utils.npy_to_pyscf(atoms["batch_positions"].detach().cpu().numpy(),
                             atoms["batch_atom_numbers"].detach().cpu().numpy(),
                             basis)
    dens = sample_density_base(mol, scaled_sample_coords,
                               [{'mo_coeff': mo_coeff, 'mo_occ': mo_occ}],
                               projected=False)
    # print('mol_time', time.time() - mol_start)
    return dens


def sample_projected_density(atoms, df_coeffs, auxbasis, auxmol=None):
    df_coeffs = df_coeffs.detach().cpu().numpy()
    scaled_sample_coords = atoms['coords'].detach().cpu().numpy() / param.BOHR  # convert Angstrom grid to Bohr
    if auxmol is None:
        mol = utils.npy_to_pyscf(atoms["batch_positions"].detach().cpu().numpy(),
                                 atoms["batch_atom_numbers"].detach().cpu().numpy(),
                                 auxbasis)
    else:
        if isinstance(auxmol, list):
            mol = auxmol
        else:
            mol = [auxmol]
    dens = sample_density_base(mol, scaled_sample_coords,
                               df_coeffs,
                               projected=True)
    return dens


def expand_df_density_by_degree(samp_df, eval_degrees, orbital_basis, orbital_basis_size, auxbasis, auxmol=None):
    df_coeffs = samp_df['df_coeffs']
    atom = [(int(samp_df['batch_atom_numbers'][0, i].detach().cpu().numpy()),
            samp_df['batch_positions'][0, i].detach().cpu().numpy())
            for i in range(samp_df['batch_positions'].shape[1])]
    df_coeffs_split = split_df_coeffs(atom, df_coeffs.squeeze(), orbital_basis_size)
    orbital_dict = combine_orbital_basis(orbital_basis, 5)[0]

    masks_pos = {key: 0 for key in orbital_basis_size}
    max_z = max(list(orbital_dict.keys()))
    max_orbitals = orbital_dict[max_z]
    masks = {}
    for z, atom_coeffs in df_coeffs_split:
        if z not in masks:
            masks[z] = torch.zeros_like(torch.tensor(atom_coeffs))
    # print('max orbitals', max_orbitals)
    for L in range(max(eval_degrees) + 1):
        nc = 2 * L + 1
        max_coeffs = max_orbitals[L][1]
        for i in range(1, max_coeffs + 1):
            for z in masks_pos.keys():
                if L >= len(orbital_dict[z]) or i >= orbital_dict[z][L][1] + 1:
                    continue
                if L in eval_degrees:
                    masks[z][masks_pos[z]:masks_pos[z] + nc] = 1
                masks_pos[z] += nc
                # print('L', L, 'max coeffs', max_coeffs, 'i', i, 'z', z, 'masks_pos', masks_pos[z])

    mask_coeffs = []
    for z, atom_coeffs in df_coeffs_split:
        atom_coeffs = torch.Tensor(atom_coeffs)
        mask_coeffs.append(atom_coeffs * masks[z])

    coeffs = torch.cat(mask_coeffs).unsqueeze(0)
    dens = sample_projected_density(samp_df, coeffs, auxbasis, auxmol)
    new_samp = {key: samp_df[key] for key in samp_df.keys()}
    new_samp['density'] = dens

    return dens


def atom_basis_descriptors(auxmol):
    atom_bas = []
    order = auxmol._bas[0, 0]
    start_bas = 0
    start_env = auxmol._bas[0, 5]
    atom_count = {}
    for i in range(auxmol._bas.shape[0]):
        row = auxmol._bas[i]
        if row[0] != order:
            order = row[0]
            atom_bas.append([(start_bas, end_bas), (start_env, end_env)])
            if start_env not in atom_count:
                atom_count[start_env] = 0
            start_env = row[5]
            start_bas = i
        end_bas = i
        end_env = row[6]

    atom_bas.append([(start_bas, end_bas), (start_env, end_env)])
    if start_env not in atom_count:
        atom_count[start_env] = 0

    # print('atom_bas', atom_bas)
    # print('atom_count', atom_count)
    return atom_bas, atom_count


def ml_basis_to_pyscf_basis(pred, atom_types, index=0):
    basis_dict = {}
    for i in range(len(pred['radial_scale'])):
        at_symb = atom_types[i]
        for key in pred['radial_scale'][i].keys():
            z, L = key
            radial_widths = pred['radial_width'][i][key][index].squeeze().numpy(force=True)
            radial_scales = pred['radial_scale'][i][key][index].squeeze().numpy(force=True)
            # print('z, L', z, L)
            # print('radial widths shape', radial_widths.shape)
            g_norm = gto_norm_pyscf(L, radial_widths)
            radial_scales = radial_scales / g_norm
            if at_symb in basis_dict:
                for j in range(radial_widths.shape[-1]):
                    basis_dict[at_symb].append([[L, [radial_widths[j].item(), radial_scales[j].item()]]])
            if at_symb not in basis_dict:
                basis_dict[at_symb] = [[[L, [radial_widths[j].item(), radial_scales[j].item()]]
                                       for j in range(radial_widths.shape[-1])]]
    return basis_dict


def ml_basis_to_auxmol(pred, index=0):
    anum = pred['batch_atom_numbers'][index]
    atom_types = []

    num_dict = {}

    for an in anum:
        if an == 0:
            continue
        an = an.item()
        symb = utils.numbers_to_symbols([an])[0]
        if an in num_dict:
            atom_types.append(symb + str(num_dict[an]))
            num_dict[an] += 1
        else:
            atom_types.append(symb + str(0))
            num_dict[an] = 1
    pos_list = [pred['batch_positions'][index, i].numpy(force=True)
                for i in range(anum.shape[0]) if anum[i] != 0]
    atom = list(zip(atom_types, pos_list))

    # Extending _env variable for duplicate atoms
    basis_dict = ml_basis_to_pyscf_basis(pred, atom_types, index)
    auxmol = gto.M(atom=atom, basis=basis_dict)
    auxmol.build()

    return auxmol


def ml_basis_to_df_coeffs(pred, basis, auxbasis, mo_coeff=None, mo_occ=None):
    nbatch = pred['batch_positions'].shape[0]
    df_bases = []
    auxmol_exts = []
    for b in range(nbatch):
        atom = [(int(pred['batch_atom_numbers'][b, i].detach().cpu().numpy()),
                pred['batch_positions'][b, i].detach().cpu().numpy())
                for i in range(pred['batch_positions'].shape[1])]

        mol = gto.M(atom=atom, basis=basis)
        mol.build()
        if mo_coeff is None:
            mf = dft.RKS(mol)
            mf.chkfile = False
            mf.xc = 'pbe'
            mf.kernel()
            dm1 = hf.make_rdm1(mf.mo_coeff, mf.mo_occ)
        else:
            dm1 = hf.make_rdm1(mo_coeff[b], mo_occ[b])

        auxmol_ext = ml_basis_to_auxmol(pred, index=b)
        # print('auxmol_ext env', auxmol_ext._env)

        # Define the auxiliary fitting basis for 3-center integrals. Use the function
        # make_auxmol to construct the auxiliary Mole object (auxmol) which will be
        # used to generate integrals.

        # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
        # indices of AO basis and P is the auxiliary basis
        ints_3c2e = df.incore.aux_e2(mol, auxmol_ext, intor='int3c2e')
        ints_2c2e = auxmol_ext.intor('int2c2e')

        nao = mol.nao
        naux = auxmol_ext.nao

# Compute the DF coefficients (df_coef) and the DF 2-electron (df_eri)
        df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao*nao, naux).T)
        df_coef = df_coef.reshape(naux, nao, nao)
        # print('df_coeff shape', df_coef.shape)
        # print('atoms', auxmol_ext._atm)
        df_basis = lib.einsum('Pij,ij->P', df_coef, dm1)
        df_bases.append(torch.from_numpy(df_basis))
        auxmol_exts.append(auxmol_ext)

    df_bases = torch.stack(df_bases, dim=0)
    return df_bases, auxmol_exts

def get_density_charges(atoms, removed_free_atom=False):
    """
    Calculate the atomwise electron density charges.

    Args:
        atoms (dict): dictionary containing the properties of the atomic system, including positions and density coefficients
    Returns:
        charges (torch.Tensor): Atomwise electron density charges [batch_size, num_atoms]
    """
    charges = torch.zeros_like(atoms['batch_atom_numbers']).to(atoms['positions'])
    for i in range(len(atoms['spherical_coeffs'])):
        for key in atoms['spherical_coeffs'][i].keys():
            z, L = key
            if L > 0:
                continue
            sph = atoms['spherical_coeffs'][i][key]
            width = atoms['radial_width'][i][key]
            scale = atoms['radial_scale'][i][key]

            charges[:, i] += torch.sum((sph * scale) / (gto_norm(0, width) * pyscf_gto_factor), dim=(-3, -2, -1))

    if not removed_free_atom:
        charges = atoms['batch_atom_numbers'] - charges
    else:
        charges = -charges
    return charges

def get_atomic_dipoles(atoms, expansion_model):
    """
    Calculate the atomic dipoles of an atomic system.

    Args:
        atoms (dict): dictionary containing the properties of the atomic system, including positions and density coefficients
    Returns:
        dipoles (torch.Tensor): Atomic dipoles for each atom in the system [batch_size, num_atoms, 3]
    """
    atoms_c = {**atoms}
    dipoles = torch.zeros_like(atoms['batch_positions'])
    for i in range(len(atoms['spherical_coeffs'])):
        dens = expansion_model(atoms_c, eval_atoms=[i], eval_L=[0, 1])['density']
        dpm1 = torch.sum((dens * atoms['coord_weights']).unsqueeze(-1) * (atoms['coords'] - atoms['batch_positions'][:, [i]]), dim=-2)
        dipoles[:, i] = -dpm1

    return dipoles

def sample_single_atom_density_spline(position, atom_number, coords, spline_basis):
    """
    Sample the free atom density of a single atom on a given coordinate grid using a spline basis.

    Args:
        position (torch.Tensor): Position of the atom [batch, 1, 3]
        atom_number (torch.Tensor): Atomic number of the atom [batch]
        coords (torch.Tensor): Coordinates of the grid to sample the density on [num_coords, 3]
        atom_dens_dict (dict): Dictionary containing the atom densities
    Returns:
        density (torch.Tensor): Free atom density of the atom
    """
    atom_coords = coords - position
    anum_nz = atom_number != 0
    dens_spline = eval_spline_density(spline_basis, atom_coords)

    return torch.tensor(dens_spline) * anum_nz.view(-1, 1)

def sample_single_atom_density_mo(position, atom_number, coords, basis, mo_coeffs):
    """
    Sample the free atom density of a single atom on a given coordinate grid using a molecular orbital basis.

    Args:
        position (torch.Tensor): Position of the atom [batch, 1, 3]
        atom_number (torch.Tensor): Atomic number of the atom [batch]
        coords (torch.Tensor): Coordinates of the grid to sample the density on [num_coords, 3]
        atom_dens_dict (dict): Dictionary containing the atom densities
    Returns:
        density (torch.Tensor): Free atom density of the atom
    """
    dens = torch.zeros((coords.shape[0], coords.shape[1]))
    coeffs = [{'mo_coeff': mo_coeffs['mo_coeff'],
               'mo_occ': mo_coeffs['mo_occ']}] * coords.shape[0]
    atom = utils.npy_to_pyscf(position.detach().cpu().numpy(),
                              atom_number.detach().cpu().numpy(),
                              basis)
    dens = sample_density_base(atom, coords, coeffs,
                               scale_coords=True, projected=False)

    return dens

def sample_single_atom_density_df(position, atom_number, coords, basis, df_coeffs):
    """
    Sample the free atom density of a single atom on a given coordinate grid using a density fitting basis.

    Args:
        position (torch.Tensor): Position of the atom [batch, 1, 3]
        atom_number (torch.Tensor): Atomic number of the atom [batch]
        coords (torch.Tensor): Coordinates of the grid to sample the density on [num_coords, 3]
        atom_dens_dict (dict): Dictionary containing the atom densities
    Returns:
        density (torch.Tensor): Free atom density of the atom
    """
    df_coeffs = [df_coeffs] * coords.shape[0]
    atom = utils.npy_to_pyscf(position.detach().cpu().numpy(),
                              atom_number.detach().cpu().numpy(),
                              basis)
    dens = sample_density_base(atom, coords, df_coeffs,
                               scale_coords=True, projected=True)
    return dens

def sample_atom_density(positions, atom_numbers, coords, basis,
                        atom_dens_type, atom_dens_dict, individual_dens=False):
    """
    Sample the free atom density of a molecule on a given coordinate grid.

    Args:
        position (torch.Tensor): Position of the atom [batch, 1, 3]
        atom_number (torch.Tensor): Atomic number of the atom [batch]
        coords (torch.Tensor): Coordinates of the grid to sample the density on [num_coords, 3]
        atom_dens_type (str): Type of the basis used to expand the atom densities
        atom_dens_dict (dict): Dictionary containing the atom densities
    Returns:
        (density, atom_wise_density) (torch.Tensor, torch.Tensor): Tuple containing free atom
        density of the molecule, plus the individual density of each atom in the molecule
    """
    dens = torch.zeros((coords.shape[0], coords.shape[1]))
    atom_densities = []
    for i in range(positions.shape[1]):
        anum = int(torch.max(atom_numbers[:, i]))
        if atom_dens_type == 'mo_coeffs':
            atom_dens = sample_single_atom_density_mo(positions[:, [i]], atom_numbers[:, [i]],
                                                      coords, basis, atom_dens_dict[anum])
            dens += atom_dens
        elif atom_dens_type == 'df_coeffs':
            atom_dens = sample_single_atom_density_df(positions[:, [i]], atom_numbers[:, [i]],
                                                      coords,
                                                      atom_dens_dict[anum]['df_basis'],
                                                      atom_dens_dict[anum]['df_coeffs'])
            dens += atom_dens
        elif atom_dens_type == 'spline':
            atom_dens = sample_single_atom_density_spline(positions[:, [i]],
                                                          atom_numbers[:, [i]], coords,
                                                          atom_dens_dict[anum]['spline_interp'])
            dens += atom_dens
        else:
            raise ValueError('Unknown free atom density type')
        if individual_dens:
            atom_densities.append(atom_dens)
    if individual_dens:
        atom_densities = torch.stack(atom_densities, dim=1)
    return dens, atom_densities

def model_input_from_atoms(atoms, use_gpu=False, density_expansion=False,
                           pyscf_grid=True, grid_spec=None,
                           grid_sampling_fn=None, cutoff=5, dtype=torch.float32,
                           atom_dens_type='spline',
                           free_atom_densities=None,
                           split_atom_densities=False, basis=None):
    """
    Function to extracts neighbor lists, atom_types, positions e.t.c. from the system and generate a properly
    formatted input for the schnetpack model.

    Args:
        system (schnetpack.md.System): System object containing current state of the simulation.

    Returns:
        dict(torch.Tensor): Schnetpack inputs in dictionary format.
    """
    positions = torch.tensor(atoms['positions'])
    if atoms['atom_numbers'].shape[0] != positions.shape[0]:
        atom_types = np.tile(atoms['atom_numbers'], (positions.shape[0], 1))
    atom_types = torch.tensor(atoms['atom_numbers'])
    if use_gpu:
        positions = positions.cuda()
        atom_types = atom_types.cuda()
    natoms = atom_types.shape[-1]
    positions = positions.view(-1, natoms, 3)
    atom_types = atom_types.view(-1, natoms)
    center = torch.sum(positions * atom_types.unsqueeze(-1), 1) / torch.sum(atom_types, 1).unsqueeze(-1)
    # inputs = {'positions': positions + 10,
    props = {'positions': (positions - center.unsqueeze(1)).cpu().numpy()}
    atom_numbers, props = utils.compress_batch_atoms(atom_types.cpu().numpy(), props)
    positions = torch.from_numpy(props['positions']).to(positions)
    anums = torch.from_numpy(atom_numbers).to(positions).type(torch.long)
    inputs = {}
    if density_expansion:
        # print('grid spec', self.grid_spec)
        if pyscf_grid:
            sample_coords, coord_weights = utils.get_pyscf_coords(grid_spec, 10000000000,
                                                                  atom_numbers,
                                                                  positions)
        else:
            sample_coords, coord_weights = utils.grid_sampling_fn(grid_spec, 10000000000,
                                                                  atom_numbers,
                                                                  positions)
        if free_atom_densities is not None:

            inputs['atom_density'], split_dens = sample_atom_density(positions=positions,
                                                                     atom_numbers=anums,
                                                                     coords=sample_coords,
                                                                     basis=basis,
                                                                     atom_dens_type=atom_dens_type,
                                                                     atom_dens_dict=free_atom_densities,
                                                                     individual_dens=split_atom_densities,
                                                                     )
            if split_atom_densities:
                inputs['atom_density_split'] = split_dens
        inputs['coords'] = sample_coords
        inputs['coord_weights'] = coord_weights

    inputs['positions'] = positions.type(dtype)
    inputs['atom_numbers_first_positions'] = utils.get_atom_num_first_positions(atom_numbers)
    inputs['atom_numbers'] = torch.tensor(atom_numbers).to(positions).type(torch.long)
    inputs['atom_mask'] = inputs['atom_numbers'] > 0

    nl = utils.TorchNeighborList(cutoff)
    idx_is, idx_js, _ = nl.get_neighbors(inputs)
    # print('inputs positions shape', inputs['positions'].shape)
    prev_max = 0
    for i in range(len(idx_is)):
        n_atoms = torch.sum(inputs['atom_mask'][i])
        idx_is[i] += prev_max
        idx_js[i] += prev_max
        prev_max += n_atoms

    atom_batch_idx = np.zeros_like(atom_numbers)
    for i in range(len(atom_numbers)):
        atom_batch_idx[i, :] = i
    atom_batch_idx = torch.tensor(atom_batch_idx).to(positions).type(torch.long)

    idx_is = torch.cat(idx_is, dim=0)
    idx_js = torch.cat(idx_js, dim=0)
    inputs['idx_i'] = idx_is
    inputs['idx_j'] = idx_js
    inputs['batch_atom_numbers'] = inputs['atom_numbers'] * 1
    inputs['batch_atom_mask'] = (inputs['atom_mask'] * 1).type(torch.bool)
    inputs['batch_positions'] = inputs['positions'] * 1
    inputs['positions'] = positions.view(1, -1, *inputs['positions'].shape[2:])
    inputs['atom_numbers'] = inputs['batch_atom_numbers'].flatten()
    inputs['atom_mask'] = inputs['batch_atom_mask'].flatten()
    batch_nz = inputs['atom_mask'].to(inputs['positions'])
    batch_idx_pos = batch_nz * torch.arange(len(batch_nz)).to(batch_nz)
    inputs['batch_idx_pos'] = batch_idx_pos[inputs['atom_mask']].to(torch.long)
    inputs['atom_numbers'] = inputs['atom_numbers'][inputs['atom_mask']].view(1, -1)
    inputs['atom_batch_idx'] = atom_batch_idx.flatten()
    inputs['atom_batch_idx'] = inputs['atom_batch_idx'][inputs['atom_mask']].view(1, -1)
    inputs['positions'] = inputs['positions'][:, inputs['atom_mask']]

    return inputs
