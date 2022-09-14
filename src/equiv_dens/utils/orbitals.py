import numpy as np
import equiv_dens.utils.base as utils
import torch
import scipy as sp

pyscf_gto_factor = 3.5449070930480957

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
    # print('gto norm', 1/gto_norm(order, width))
    # print('scale', scale)
    # print('width', width)
    # print('scale calc', scale_calc)
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


def coeffs_dict_to_vector(coeffs, orbital_basis, a_num, radial_coeffs=True):
    sph_coeffs = coeffs['spherical_coeffs']
    if radial_coeffs:
        rad_width = coeffs['radial_width']
        rad_scale = coeffs['radial_scale']

    all_sph = None
    all_scale = None
    all_width = None
    for i, z in enumerate(torch.max(a_num, dim=0)[0]):
        orb_ind = {}
        z = int(z)
        basis = orbital_basis[z]
        for orb in basis:
            L = orb[2]
            key = (z, L)
            if key not in orb_ind.keys():
                orb_ind[key] = 0
            if all_sph is None:
                all_sph = 1 * sph_coeffs[i][key][..., orb_ind[key]]
                if radial_coeffs:
                    all_scale = 1 * rad_scale[i][key][..., orb_ind[key]]
                    all_width = 1 * rad_width[i][key][..., orb_ind[key]]
                orb_ind[key] += 1
            else:
                all_sph = torch.cat([all_sph, sph_coeffs[i][key][..., orb_ind[key]]], dim=2)
                if radial_coeffs:
                    all_scale = torch.cat([all_scale, rad_scale[i][key][..., orb_ind[key]]], dim=2)
                    all_width = torch.cat([all_width, rad_width[i][key][..., orb_ind[key]]], dim=2)
                orb_ind[key] += 1

    vector_coeffs = {}
    vector_coeffs['spherical_coeffs'] = all_sph.squeeze(1)
    if radial_coeffs:
        vector_coeffs['radial_width'] = all_width.squeeze(1)
        vector_coeffs['radial_scale'] = all_scale.squeeze(1)

    return vector_coeffs


def vector_to_coeffs_dict(coeffs, orbital_basis, a_num, radial_coeffs=True):
    vec_sph = coeffs['spherical_coeffs']
    dict_sph = []
    if radial_coeffs:
        vec_width = coeffs['radial_width']
        vec_scale = coeffs['radial_scale']
        dict_width = []
        dict_scale = []
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

    return dict_coeffs


def orbitals_from_hamiltonian(hamiltonians, overlaps):
    orbital_coeffs = []
    for i in range(hamiltonians.shape[0]):
        en, coefs = sp.linalg.eigh(a=hamiltonians[i], b=overlaps[i])
        orbital_coeffs.append(coefs)
    return orbital_coeffs


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

