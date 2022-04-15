import numpy as np
import torch
import scipy as sp


def combine_orbitals(orbitals, order_max):
    radial_spec = [None] * len(orbitals)
    spherical_spec = [None] * len(orbitals)
    radial_counts = [None] * len(orbitals)
    for i in range(len(orbitals)):
        radial_L_count = [0] * (order_max + 2)
        spherical_L_count = [0] * (order_max + 2)
        radial_spec[i] = []
        spherical_spec[i] = []
        radial_counts[i] = [[] for i in range(order_max + 2)]
        # print('density L count len', len(density_L_count))
        z = orbitals[i][0][0]
        for j in range(len(orbitals[i])):
            orb = orbitals[i][j]
            L = orb[2]
            radial_L_count[L] += orb[1]
            spherical_L_count[L] += 1
            radial_counts[i][L].append(orb[1])
        for L, c in enumerate(radial_L_count):
            if c == 0:
                continue
            radial_spec[i].append((z, c, L))
        for L, c in enumerate(spherical_L_count):
            if c == 0:
                continue
            spherical_spec[i].append((z, c, L))
    return spherical_spec, radial_spec, radial_counts


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


def get_max_order(orbitals, per_atom=False):
    order_max = {}
    for i in range(len(orbitals)):
        for z, _, l in orbitals[i]:
            if z in order_max.keys():
                if l > order_max[z]:
                    order_max[z] = l
            else:
                order_max[z] = l
    if per_atom:
        return order_max
    else:
        return max(order_max.values())


def get_n_electrons_transfer(atom_numbers):
    return torch.sum(atom_numbers, -1, keepdim=True)


def get_n_electrons(orbitals):
    n_electrons = 0
    for i in range(len(orbitals)):
        n_electrons += orbitals[i][0][0]
    return n_electrons


def gaussian_rbf(r, width, scale, normalize=True):
    # print('scale shape', scale.shape)
    # print('scale shape', scale.shape)
    # print('width shape', width.shape)
    # print('r shape', r.shape)
    if normalize:
        scale_calc = scale * 8 * (width**(3 / 2)) / (np.pi**(3 / 2) * 53.9866)
    else:
        scale_calc = scale

    print('scale shape', scale_calc.shape)
    print('r', r.shape)
    rbf = scale_calc * torch.exp(-width * (r)**2)
    return torch.sum(rbf, dim=-2, keepdim=True)


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
