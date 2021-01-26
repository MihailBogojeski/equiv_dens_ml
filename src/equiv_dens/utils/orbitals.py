import numpy as np
import torch
from torch.nn.functional import pad


def combine_orbitals(orbitals, order_max):
    orbital_spec = [None] * len(orbitals)
    radial_counts = [None] * len(orbitals)
    for i in range(len(orbitals)):
        orbital_L_count = [0] * (order_max + 2)
        orbital_spec[i] = []
        radial_counts[i] = [[]] * (order_max + 2)
        # print('density L count len', len(density_L_count))
        z = orbitals[i][0][0]
        for j in range(len(orbitals[i])):
            orb = orbitals[i][j]
            L = orb[2]
            orbital_L_count[L] += 1
            radial_counts[i][L].append(orb[1])
        for L, c in enumerate(orbital_L_count):
            if c == 0:
                continue
            orbital_spec[i].append((z, c, L))
    return orbital_spec, radial_counts


def get_max_order(orbitals):
    order_max = 0
    for i in range(len(orbitals)):
        for z, _, l in orbitals[i]:
            if l > order_max:
                order_max = l
    return order_max


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
    rbf = scale_calc * torch.exp(-width * (r)**2)
    return torch.sum(rbf, dim=-2, keepdim=True)


def get_invariant_features(coeffs, permutational_invariance=True, keep_dims=False):
    # coeffs = model(R=R)
    sph_coeffs = coeffs['spherical_coeffs']
    rad_coeffs = coeffs['radial_width']
    rad_scale = coeffs['radial_scale']

    max_order = 0
    for i in range(len(sph_coeffs)):
        for key in sph_coeffs[i].keys():
            if key[1] > max_order:
                max_order = key[1]

    first_key = list(sph_coeffs[0].keys())[0]
    batch_size = sph_coeffs[0][first_key].shape[0]

    max_num_coeffs = [0] * (max_order + 1)
    max_num_radial = [0] * (max_order + 1)
    # print('max order', model.order_max)
    for i in range(len(sph_coeffs)):
        for key in sph_coeffs[i].keys():
            L = key[1]
            # print(i, L)
            if sph_coeffs[i][key].shape[-1] > max_num_coeffs[L]:
                max_num_coeffs[L] = sph_coeffs[i][key].shape[-1]
            if rad_coeffs[i][key].shape[-2] > max_num_radial[L]:
                max_num_radial[L] = rad_coeffs[i][key].shape[-2]
    # print('max num coeffs', max_num_coeffs)
    # print('max num radial', max_num_radial)

    all_sph = [[torch.zeros([batch_size, 1, 1, max_num_coeffs[i]]).to(sph_coeffs[0][first_key])
               for j in range(len(sph_coeffs))]
               for i in range(max_order + 1)]
    all_width = [[torch.zeros([batch_size, 1, max_num_radial[i], max_num_coeffs[i]]).to(sph_coeffs[0][first_key])
                 for j in range(len(sph_coeffs))]
                 for i in range(max_order + 1)]
    all_scale = [[torch.zeros([batch_size, 1, max_num_radial[i], max_num_coeffs[i]]).to(sph_coeffs[0][first_key])
                 for j in range(len(sph_coeffs))]
                 for i in range(max_order + 1)]
    # print('all width shapes', [[d.shape for d in sca] for sca in all_scale])

    for i in range(len(sph_coeffs)):
        for key in sph_coeffs[i].keys():
            L = key[1]
            # print('i, ', i, ', key', key)
            sph_norm = sph_coeffs[i][key].norm(dim=-2, keepdim=True)

            padding = (0, max_num_coeffs[L] - sph_norm.shape[-1])
            # print('padding', padding)
            sph_norm = pad(sph_norm, padding)
            width_pad = pad(rad_coeffs[i][key], padding)
            scale_pad = pad(rad_scale[i][key], padding)
            all_sph[L][i] = sph_norm
            all_width[L][i] = width_pad
            all_scale[L][i] = scale_pad

    for L in range(max_order + 1):
        all_sph[L] = torch.stack(all_sph[L], dim=1)
        all_width[L] = torch.stack(all_width[L], dim=1)
        all_scale[L] = torch.stack(all_scale[L], dim=1)
        if permutational_invariance:
            all_sph[L] = all_sph[L].sum(dim=1)
            all_width[L] = all_width[L].sum(dim=1)
            all_scale[L] = all_scale[L].sum(dim=1)

    all_sph = torch.cat(all_sph, dim=-1)
    all_width = torch.cat(all_width, dim=-1)
    all_scale = torch.cat(all_scale, dim=-1)

    invariant_feats = torch.cat([all_sph, all_width, all_scale], dim=-1)
    if keep_dims:
        invariant_feats = invariant_feats.squeeze(2)
    else:
        invariant_feats = invariant_feats.squeeze()
    return invariant_feats


def compress_coefficients(coeffs):
    sph_coeffs = coeffs['spherical_coeffs']
    rad_coeffs = coeffs['radial_width']
    rad_scale = coeffs['radial_scale']

    max_order = 0
    for i in range(len(sph_coeffs)):
        for key in sph_coeffs[i].keys():
            if key[1] > max_order:
                max_order = key[1]

    first_key = list(sph_coeffs[0].keys())[0]
    batch_size = sph_coeffs[0][first_key].shape[0]

    max_num_coeffs = [0] * (max_order + 1)
    max_num_radial = [0] * (max_order + 1)
    # print('max order', model.order_max)
    for i in range(len(sph_coeffs)):
        for key in sph_coeffs[i].keys():
            L = key[1]
            # print(i, L)
            if sph_coeffs[i][key].shape[-1] > max_num_coeffs[L]:
                max_num_coeffs[L] = sph_coeffs[i][key].shape[-1]
            if rad_coeffs[i][key].shape[-2] > max_num_radial[L]:
                max_num_radial[L] = rad_coeffs[i][key].shape[-2]
    # print('max num coeffs', max_num_coeffs)
    # print('max num radial', max_num_radial)

    max_coeffs_all = max(max_num_coeffs)
    all_sph = [[torch.zeros([batch_size, 1, (2 * i) + 1, max_num_coeffs[i]]).to(sph_coeffs[0][first_key])
               for j in range(len(sph_coeffs))]
               for i in range(max_order + 1)]
    all_width = [[torch.zeros([batch_size, 1, max_num_radial[i], max_num_coeffs[i]]).to(sph_coeffs[0][first_key])
                 for j in range(len(sph_coeffs))]
                 for i in range(max_order + 1)]
    all_scale = [[torch.zeros([batch_size, 1, max_num_radial[i], max_num_coeffs[i]]).to(sph_coeffs[0][first_key])
                 for j in range(len(sph_coeffs))]
                 for i in range(max_order + 1)]

    for i in range(len(sph_coeffs)):
        for key in sph_coeffs[i].keys():
            L = key[1]
            # print('i, ', i, ', key', key)
            sph_tensor = sph_coeffs[i][key]

            padding = (0, max_coeffs_all - sph_tensor.shape[-1])
            # print('padding', padding)
            sph_pad = pad(sph_tensor, padding)
            width_pad = pad(rad_coeffs[i][key], padding)
            scale_pad = pad(rad_scale[i][key], padding)
            all_sph[L][i] = sph_pad
            all_width[L][i] = width_pad
            all_scale[L][i] = scale_pad

    for L in range(max_order + 1):
        all_sph[L] = torch.stack(all_sph[L], dim=1)
        all_width[L] = torch.stack(all_width[L], dim=1)
        all_scale[L] = torch.stack(all_scale[L], dim=1)

    return all_sph, all_scale, all_width

