import torch
from torch.nn.functional import pad
import time
import numpy as np
from torch.autograd import grad


def from_r(model, R):
    start = time.time()
    r_desc, r_d_desc = r_to_desc(model, R, grad=True, parallel=False)
    # r_desc, r_d_desc = r_to_schnet_desc(schnet, properties, grad=True)
    print("r_desc.shape", r_desc.shape)
    print("r_d_desc.shape", r_d_desc.shape)

    duration = time.time() - start
    print('Analytical gradients duration:', duration)

    # dh = 0.025
    # dh = 0.0025
    # dh = 0.00025

    # for dh in [0.25, 0.025, 0.0025, 0.00025]:
    with torch.no_grad():
        # for dh in [0.25, 0.025, 0.0025, 0.00025]:
        for dh in [0.025]:
            start = time.time()
            r_d_num_desc = get_numerical_gradients(model, R, offset=dh)
            print('grad', r_d_desc[0, 0, :])
            print('num grad', r_d_num_desc[0, 0, :])
            duration = time.time() - start
            print('Numerical gradients duration:', duration)

            diff = torch.abs(r_d_num_desc - r_d_desc.to(r_d_num_desc))
            # print("r_d_num_desc-r_d_desc",diff)
            print("Statistics difference: dh=", dh)
            print("mean(diff)", torch.mean(diff))
            print("max(diff)", torch.max(diff))
            print("min(diff)", torch.min(diff))

    return r_desc, r_d_desc, r_d_num_desc


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

    all_sph = [[torch.zeros([batch_size, 1, 1, max_num_coeffs[i]]).to(sph_coeffs[0][first_key]) for j in range(len(sph_coeffs))]
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

            # print('shp norm shape', sph_norm.shape)
            # print('max num coeffs', max_num_coeffs[L])
            padding = (0, max_num_coeffs[L] - sph_norm.shape[-1])
            # print('padding', padding)
            sph_norm = pad(sph_norm, padding)
            width_pad = pad(rad_coeffs[i][key], padding)
            scale_pad = pad(rad_scale[i][key], padding)
            all_sph[L][i] = sph_norm
            all_width[L][i] = width_pad
            all_scale[L][i] = scale_pad
            # print('sph_coeffs', sph_norm.shape)
            # print('rad_coeffs', width_pad.shape)
            # print('rad_scale',  scale_pad.shape)
            #
            # print('sph_coeffs', all_sph[L][i].shape)
            # print('rad_coeffs', all_width[L][i].shape)
            # print('rad_scale', all_scale[L][i].shape)
            # print([[d.shape for d in all_sph[order] if d is not None] for order in range(model.order_max + 1)])

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


def r_to_desc(model, R, grad=False, parallel=False):
    if grad:
        R.requires_grad = True
        if parallel:
            coeffs = model(R=R)
            desc = get_invariant_features(coeffs)
            nfeats = desc.shape[-1]
            natoms = R.shape[1]

            d_R = R.repeat(nfeats * natoms, 1, 1)
            # print('grad positions', d_R.shape)
            # print('grad positions req grad', d_R.requires_grad)

            desc, d_desc = get_gradients_fast(model, d_R, nfeats, natoms, batch_size=120)
        else:
            coeffs = model(R=R)
            desc = get_invariant_features(coeffs)
            d_desc = get_gradients(R, desc)

        R.requires_grad = False
        return desc, d_desc
    else:
        coeffs = model(R=R)
        desc = get_invariant_features(coeffs)
        return desc


def get_gradients_fast(model, d_R, nfeats, natoms, batch_size=100):
    # print('d_prop positions', d_properties['_positions'].shape)
    batch_R = batch_gradients(d_R, batch_size, nfeats * natoms)
    # print('batch props', len(batch_properties))
    d_descs = []
    descs = []
    for b_R, mask in batch_R:
        bs = b_R.shape[0]
        coeffs = model(R=d_R)
        b_desc = get_invariant_features(coeffs)
        flat_desc = b_desc.view(bs, -1)
        grads = grad(flat_desc, b_R, mask.to(b_desc))[0]
        d_descs.append(grads)

    descs = b_desc[:1]
    d_descs = torch.cat(d_descs)
    d_descs = d_descs.reshape(1, natoms, nfeats, -1)
    print('grad descs shape', d_descs.shape)

    return descs, d_descs


def batch_gradients(d_R, batch_size, ngrads=None):
    n = d_R.shape[0]
    if ngrads is None:
        ngrads = n
    n_batches = int(n / batch_size) + 2
    slices = np.linspace(0, n, n_batches)
    slices = np.round(slices)
    batches = []
    gradient_mask = torch.eye(ngrads)
    for i in range(1, len(slices)):
        inds = np.arange(slices[i - 1], slices[i], dtype=int)
        # inds = np.arange(sep[i - 1], sep[i], dtype=int)
        b_R = d_R[inds]

        batches.append((b_R, gradient_mask[inds]))

    return batches


def get_gradients(R, desc):
    d_desc = torch.zeros([*desc.shape] + [R.shape[2] * R.shape[1]]).to(desc)

    for i in range(desc.shape[-1]):
        print('i', i)
        # print('desc shape', desc.shape)
        tmp = grad(desc[:, i], R, torch.ones_like(desc[:, i]), retain_graph=True)[0]
        # print('d_desc shape', d_desc.shape)
        d_desc[:, i, :] = tmp.view(tmp.shape[0], -1)

    print('grad descs shape', d_desc.shape)

    return d_desc


def get_numerical_gradients(model, R, offset=0.001):
    batch_size, n_atoms, dims = R.shape
    r_flat = R.cpu().clone().view(batch_size, -1)
    print('r_flat shape', r_flat.shape)
    displacements = []
    for i in range(r_flat.shape[-1]):
        displacement = r_flat.clone()
        displacement[:, i] += offset
        displacements.append(displacement)

        displacement = r_flat.clone()
        displacement[:, i] -= offset
        displacements.append(displacement)

    print(len(displacements))
    num_displacements = len(displacements)

    displacements = torch.stack(displacements, dim=1)
    print('displacements shape', displacements.shape)
    displacements = displacements.view(batch_size * num_displacements, -1, 3)  # [6 * N_atms, N_atms, 3]
    print('displacements shape', displacements.shape)

    d_R = torch.Tensor(displacements).to(R)
    with torch.no_grad():
        coeffs = model(R=d_R)
        descs = get_invariant_features(coeffs)

    nfeats = descs.shape[1]

    # print('descs', descs.shape)
    d_num_descs = (descs[::2, :] - descs[1::2, :]) / (2 * offset)
    print('d_num_descs', d_num_descs.shape)
    d_num_descs = d_num_descs.view(batch_size, -1, nfeats)
    d_num_descs = d_num_descs.permute(0, 2, 1)
    # d_num_descs = d_num_descs.permute(1, 0)
    # d_num_descs = d_num_descs.unsqueeze(dim=0)
    print('d descs shape', d_num_descs.shape)

    return d_num_descs
