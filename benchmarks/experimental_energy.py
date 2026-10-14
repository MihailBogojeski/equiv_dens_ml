"""Rejected H5-E1 accumulation retained for reproducible benchmarks only."""
import time

import numpy as np
import torch

from equiv_dens.nn.property_output.energy import SphericalHarmonicsEnergyNetwork
from equiv_dens.nn.modules.network_blocks import layer_norm
from equiv_dens.utils.orbitals import coeffs_dict_to_tensors


def _accumulate_energy_features(previous, update, normalize, backend='legacy'):
    """Accumulate with the original mean-square-zero normalization policy."""
    if backend not in ('legacy', 'device'):
        raise ValueError(f'Unknown energy accumulation backend: {backend}')
    if not normalize:
        scale = 1
    elif backend == 'device':
        zero_mean_square = (torch.mean(previous**2) == 0) | (torch.mean(update**2) == 0)
        scale = torch.where(zero_mean_square, previous.new_ones(()),
                            previous.new_full((), np.sqrt(1/2)))
    elif torch.mean(previous**2) == 0 or torch.mean(update**2) == 0:
        scale = 1
    else:
        scale = np.sqrt(1/2)
    return update * scale + previous * scale


class ExperimentalEnergyNetwork(SphericalHarmonicsEnergyNetwork):
    """Original initialized network with the measured benchmark forward."""

    def __init__(self, *args, accumulation_backend="legacy", **kwargs):
        if accumulation_backend not in ("legacy", "device"):
            raise ValueError(f"Unknown energy accumulation backend: {accumulation_backend}")
        super().__init__(*args, **kwargs)
        self.accumulation_backend = accumulation_backend

    def forward(self, atoms):
        start = time.time()
        R = atoms['positions']
        N = R.shape[1]
        batch_size = R.shape[0]
        idx_i = atoms['idx_i']
        idx_j = atoms['idx_j']
        neighbor_mask = 1
        # exclude self - interactions
        # initialize atomic features to embeddings
        # print('atoms sph_dict', atoms['sph_dict'])
        # print('atoms keys', atoms.keys())
        # print('atoms dens', self.atom_dens[16]['df_basis'])
        # print('atoms df widths', self.atom_df_widths)
        # print('atoms df scales', self.atom_df_scales)
        # print('radial_widths 16', [(key, atoms['radial_width'][8][key]) for key in atoms['radial_width'][8].keys() if key[1]==0])
        # print('radial_scales 16', [(key, atoms['radial_scale'][8][key]) for key in atoms['radial_scale'][8].keys() if key[1]==0])
        # print('spherical coeffs 0', [(key, atoms['spherical_coeffs'][0][key]) for key in atoms['spherical_coeffs'][0].keys() if key[1]==0])
        sph_fs, scale_fs, width_fs = coeffs_dict_to_tensors(atoms, radial_coeffs=self.pred_radial_coeffs,
                                                            atom_df_coeffs=self.atom_df_coeffs,
                                                            atom_df_widths=self.atom_df_widths,
                                                            atom_df_scales=self.atom_df_scales)
        # print('sph fs 0', sph_fs[0][:, 0, 0, :10])
        # print('width fs 0', width_fs[0][:, 8, 0, 32:])
        # print('scale fs 0', scale_fs[0][:, 8, 0, 32:])
        # print('atoms.atom_dens', self.atom_dens[1]['df_coeffs'])
        for i in range(len(sph_fs)):
            sph_fs[i] = sph_fs[i].view(1, -1, *sph_fs[i].shape[2:])
            sph_fs[i] = sph_fs[i][:, atoms['atom_mask']]
            scale_fs[i] = scale_fs[i].view(1, -1, *scale_fs[i].shape[2:])
            scale_fs[i] = scale_fs[i][:, atoms['atom_mask']]
            width_fs[i] = width_fs[i].view(1, -1, *width_fs[i].shape[2:])
            width_fs[i] = width_fs[i][:, atoms['atom_mask']]
        dij = atoms['distances']
        sph = atoms['sph']
        rbf = self.radial_basis_functions(dij).unsqueeze(-2)  # unsqueeze for broadcasting
        xs = []
        radial_all = 0
        # print('dens features', self.dens_features)
        if self.pred_radial_coeffs:
            for L in range(len(scale_fs)):
                # print('L', L)
                # print('self.radial_scale_filters[L]', self.radial_scale_filters(L).shape)
                scale_fs[L] = scale_fs[L] * self.radial_scale_filters(L)
                width_fs[L] = width_fs[L] * self.radial_width_filters(L)
                radial_comb = self.coeff_activation[L](scale_fs[L] * width_fs[L])
                radial_comb = radial_comb / torch.sqrt(torch.abs(radial_comb) + 1e-8)
                radial_comb = radial_comb.sum(-2, keepdim=True)
                if self.L0_start:
                    radial_all = radial_all + self.radial_L0_map[L](radial_comb)
                    xs.append(sph_fs[L])
                else:
                    xs.append(sph_fs[L] * radial_comb)
            if self.L0_start:
                xs[0] = xs[0] * radial_all
                xs[0] = xs[0] / torch.sqrt(torch.abs(xs[0]) + 1e-8)

        else:
            xs = sph_fs

        for L in range(len(xs)):
            xs[L] = self.input_layer[L](xs[L])

        if self.L0_start:
            xs = self.L0_change(xs)

        if self.normalize > 1 or self.L0_start:
            for L in range(len(xs)):
                xs[L] = layer_norm(xs[L], dims=(-3, -2, -1))
        # print('xs energy norm after input layer:', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
        # print('xs.shape', xs[0].shape)
        # print('positions.shape', atoms['positions'].shape)
        # perform iterations over modular building blocks to get environment - dependent features
        fs = [torch.tensor(0, device=xs[0].device, dtype=xs[0].dtype) for _ in range(max(self.order_max, self.orbitals_max_order) + 1)]  # output features
        # print('self normalize en', self.normalize)
        # print('fs norm start :', [float(torch.mean(fs[L]**2)) for L in range(len(fs))])
        # fs = [torch.zeros_like(x) for x in xs]  # output features
        accumulation_backend = getattr(self, 'accumulation_backend', 'legacy')
        for i, module in enumerate(self.module):
            xs = self.order_change[i](xs)
            # print('xs norm module ', i, ':', [float(torch.mean(xs[L]**2)) for L in range(len(xs))])
            # print('xs nan', [torch.any(torch.isnan(xs[L])) for L in range(len(xs))])
            xs, ys = module(xs, rbf, sph, idx_i, idx_j, neighbor_mask=neighbor_mask)
            # print('ys nan', [torch.any(torch.isnan(ys[L])) for L in range(len(ys))])
            for L in range(self._order[i] + 1):
                if accumulation_backend == 'device' and self.normalize:
                    fs[L] = _accumulate_energy_features(fs[L], ys[L], self.normalize, 'device')
                    continue
                if not self.normalize or torch.mean(fs[L]**2) == 0 or torch.mean(ys[L]**2) == 0:
                    scale = 1
                else:
                    scale = np.sqrt(1/2)
                    # print('self normalize en', self.normalize)
                fs[L] = ys[L] * scale + fs[L] * scale
                # print('module', i, 'fs[0]', fs[0])
            # print('fs norm ', i, ':', [float(torch.mean(fs[L]**2)) for L in range(len(fs))])
            # print('fs nan', [torch.any(torch.isnan(fs[L])) for L in range(len(fs))])
        fs[0] = self.out_activation(fs[0])

        atom_en = self.energy_output(fs)[0].squeeze(-1).squeeze(-1)
        # print('energy', atom_en)

        energy = torch.zeros(1, atoms['batch_atom_numbers'].shape[0], device=atoms['positions'].device, dtype=atoms['positions'].dtype)
        energy = energy.scatter_add(1, atoms['atom_batch_idx'], atom_en)
        energy = torch.t(energy)

        atoms['energy'] = energy
        if self.calculate_forces:
            forces = -torch.autograd.grad(torch.sum(energy), atoms['positions'], create_graph=self.training)[0]
            atoms['forces'] = forces

        # raise Exception('Random exception')

        if self.timing:
            print('simple energy time', time.time() - start)
        return atoms


def install_energy_accumulation_experiment(model, backend="device"):
    """Select the exact rejected forward without rebuilding parameters/state."""
    if backend not in ("legacy", "device"):
        raise ValueError(f"Unknown energy accumulation backend: {backend}")
    energy = model.property_models["energy"]
    if not isinstance(energy, SphericalHarmonicsEnergyNetwork):
        raise TypeError("Energy experiment requires SphericalHarmonicsEnergyNetwork")
    energy.__class__ = ExperimentalEnergyNetwork
    energy.accumulation_backend = backend
    return model
