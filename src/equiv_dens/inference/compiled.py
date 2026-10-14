"""Strict functional compilation of the original, fixed-topology DenSNet.

Topology preparation is a host operation. All geometry, density coefficients,
energy, forces, losses and SGD updates remain in the full compiled graph. The
source model is copied, retaining its parameter names and tensor submodules.
This initial adapter supports normalize=0, no core head and supplied SAD grid
values only; unsupported scientific policies fail explicitly.
"""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from equiv_dens.nn.property_output.density import DensityCoeffsNetwork, DensityExpansion
from equiv_dens.nn.property_output.energy import SphericalHarmonicsEnergyNetwork
from equiv_dens.utils.base import batch_compressed_atoms, calculate_distances_and_directions, bohr_to_angstrom
from equiv_dens.utils.orbitals import coeffs_dict_to_tensors, eval_gto, eval_gto_grad
from equiv_dens.utils.spherical_harmonics import spherical_harmonics
from equiv_dens.utils.spherical_harmonics_deriv import spherical_harmonics_deriv


class _StaticCoefficients(DensityCoeffsNetwork):
    def extract_coefficients(self, sph_fs, rad_width, rad_scale, atoms):
        spherical, widths, scales, weights = [], [], [], []
        for i, z in enumerate(self._species):
            spherical.append({}); widths.append({}); scales.append({}); weights.append({})
            if z == 0:
                continue
            mask = atoms['batch_atom_mask'][:, i].to(sph_fs[0]).reshape(-1, 1, 1, 1)
            for orb in self.spherical_spec[z]:
                L, key = orb[2], (z, orb[2])
                features = sph_fs[L][:, i:i + 1]
                spherical[i][key] = features[..., self.sph_dict[key]]
                if self.coeff_weights is not None:
                    weights[i][key] = self.coeff_weight(key).sum(-2, keepdim=True).expand(
                        -1, -1, spherical[i][key].shape[-2], -1).clone()
                shape = (*features.shape[:2], self.r_max[key], orb[1])
                width, scale = features.new_zeros(shape), features.new_zeros(shape)
                if self.pred_radial_coeffs:
                    inds = self.rad_dict[key]
                    dest = getattr(self, '_radial_pack_{}_{}'.format(z, L))
                    width = width.flatten(-2).index_copy(-1, dest, rad_width[L][:, i:i + 1, 0, inds]).view(shape)
                    scale = scale.flatten(-2).index_copy(-1, dest, rad_scale[L][:, i:i + 1, 0, inds]).view(shape)
                if self.init_coeffs is not None and L == 0:
                    spherical[i][key] = spherical[i][key] + self.init_sph(key)
                    if self.pred_radial_coeffs:
                        width = (width + self.init_width(key)).clamp(-0.999999, 0.99999)
                        scale = scale + self.init_scale(key)
                if self.radial_coeffs is not None:
                    width = ((width + 1) * self.ml_width_scale + self.ml_width_offset) * self.init_radial_width(key)
                    if self.positive_coeffs:
                        scale = ((scale + 1) * self.ml_width_scale + self.ml_width_offset) * self.init_radial_scale(key) * mask
                    else:
                        scale = (scale + self.init_radial_scale(key)) * mask
                widths[i][key], scales[i][key] = width, scale
        return spherical, widths, scales, weights

    def forward(self, atoms):
        atoms['sph_repr_batch'] = [x * 1 for x in atoms['sph_repr']]
        atoms = batch_compressed_atoms(atoms, ['sph_repr_batch'])
        features = atoms['sph_repr_batch']
        spherical = self.spherical_output(features)
        if self.linear_out:
            spherical = self.linear_output(spherical)
        if self.scale_sph_order:
            spherical = [x * 10 ** (-L) for L, x in enumerate(spherical)]
        if self.positive_coeffs:
            spherical[0] = F.softplus(spherical[0])
        widths, scales = [], []
        if self.pred_radial_coeffs:
            for L in range(len(self.radial_width)):
                widths.append(torch.tanh(self.radial_width[L](features[0])))
                scale = self.radial_scale[L](features[0])
                scales.append(torch.tanh(scale) if self.positive_coeffs else scale)
        atoms['spherical_coeffs'], atoms['radial_width'], atoms['radial_scale'], atoms['coeff_weights'] = self.extract_coefficients(spherical, widths, scales, atoms)
        if self.integral_constraint is not None:
            atoms = self.integral_constraint(atoms)
        atoms['sph_dict'] = self.sph_dict
        atoms['rad_dict'] = self.rad_dict
        # df_coeffs is an auxiliary PySCF export, unused by these applications.
        return atoms


def _layer_norm(x):
    """Original unbiased variance and global zero-variance branch."""
    dims = (-3, -2, -1)
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, keepdim=True)
    zero = var.sum() == 0
    std = torch.where(zero, torch.zeros_like(var), torch.sqrt(torch.where(zero, torch.ones_like(var), var)))
    return (x - mean) / (std + 1e-8)


class _StaticEnergy(SphericalHarmonicsEnergyNetwork):
    def forward(self, atoms):
        spherical, scales, widths = coeffs_dict_to_tensors(
            atoms, radial_coeffs=self.pred_radial_coeffs,
            atom_df_coeffs=self.atom_df_coeffs, atom_df_widths=self.atom_df_widths,
            atom_df_scales=self.atom_df_scales)
        for values in (spherical, scales, widths):
            for L, value in enumerate(values):
                values[L] = value.view(1, -1, *value.shape[2:]).index_select(1, self._active)
        rbf = self.radial_basis_functions(atoms['distances']).unsqueeze(-2)
        xs, radial_all = [], 0
        if self.pred_radial_coeffs:
            for L in range(len(scales)):
                scales[L] = scales[L] * self.radial_scale_filters(L)
                widths[L] = widths[L] * self.radial_width_filters(L)
                radial = self.coeff_activation[L](scales[L] * widths[L])
                radial = radial / torch.sqrt(radial.abs() + 1e-8)
                radial = radial.sum(-2, keepdim=True)
                if self.L0_start:
                    radial_all = radial_all + self.radial_L0_map[L](radial)
                    xs.append(spherical[L])
                else:
                    xs.append(spherical[L] * radial)
            if self.L0_start:
                xs[0] = xs[0] * radial_all
                xs[0] = xs[0] / torch.sqrt(xs[0].abs() + 1e-8)
        else:
            xs = spherical
        xs = [self.input_layer[L](x) for L, x in enumerate(xs)]
        if self.L0_start:
            xs = self.L0_change(xs)
            xs = [_layer_norm(x) for x in xs]
        outputs = [xs[0].new_zeros(()) for _ in range(max(self.order_max, self.orbitals_max_order) + 1)]
        for i, module in enumerate(self.module):
            xs = self.order_change[i](xs)
            xs, ys = module(xs, rbf, atoms['sph'], atoms['idx_i'], atoms['idx_j'], neighbor_mask=1)
            for L in range(self._order[i] + 1):
                outputs[L] = ys[L] + outputs[L]
        outputs[0] = self.out_activation(outputs[0])
        atomic = self.energy_output(outputs)[0].squeeze(-1).squeeze(-1)
        energy = atomic.new_zeros(1, atoms['batch_atom_numbers'].shape[0])
        atoms['energy'] = energy.scatter_add(1, atoms['atom_batch_idx'], atomic).t()
        return atoms


class _StaticExpansion(DensityExpansion):
    def forward(self, atoms):
        atoms['density'], atoms['density_grad'] = 0, 0
        for i, z in enumerate(self._species):
            if z == 0:
                continue
            pos = atoms['batch_positions'][:, [i]]
            distance, direction = calculate_distances_and_directions(atoms['coords'], center=pos)
            sph = spherical_harmonics(self.orbitals_max_order_dict[z], direction)
            deriv = spherical_harmonics_deriv(self.orbitals_max_order_dict[z], direction)
            sph = [torch.where(torch.isnan(x), torch.zeros_like(x), x) for x in sph]
            for orb in self.spherical_spec[z]:
                L, key = orb[2], (z, orb[2])
                coeff = atoms['spherical_coeffs'][i][key]
                width, scale = atoms['radial_width'][i][key], atoms['radial_scale'][i][key]
                extra = sph[L].dim() - coeff.dim()
                if extra >= 0:
                    def expand(value):
                        return value.reshape(value.shape[:2] + (1,) * (extra + 1) + value.shape[2:])
                    coeff, width, scale = expand(coeff), expand(width), expand(scale)
                part, angular, radial = eval_gto(sph[L], distance, L, coeff, width, scale)
                atoms['density'] = atoms['density'] + part
                grad = eval_gto_grad(deriv[L], distance, L, atoms['coords'] - pos, coeff, width, scale, angular, radial)
                atoms['density_grad'] = atoms['density_grad'] + bohr_to_angstrom(grad)
        # Preserve the historical behavior: constraints change density only.
        if self.expansion_constraint == 'sq':
            atoms['density'] = atoms['density'] ** 2
        elif self.expansion_constraint == 'abs':
            atoms['density'] = torch.sqrt(atoms['density'] ** 2)
        elif self.expansion_constraint == 'sp':
            atoms['density'] = F.softplus(atoms['density'], beta=100000000) + 1e-30
        if self.integral_constraint == 'grid':
            factor = atoms['batch_atom_numbers'].sum(-1, keepdim=True) / (atoms['density'] * atoms['coord_weights']).sum(1, keepdim=True)
            factor = factor * (1 - self.grid_scaling_factor) + self.grid_scaling_factor
            atoms['density'] = atoms['density'] * factor
        return atoms


class _Application(nn.Module):
    def __init__(self, model, example):
        super().__init__()
        copied = copy.deepcopy(model)
        self.density_repr_model = copied.density_repr_model
        self.property_models = copied.property_models
        self.remove_atom_density = copied.remove_atom_density
        species = tuple(example['batch_atom_numbers'].amax(0).cpu().tolist())
        heads = [m for m in self.density_repr_model.modules() if isinstance(m, DensityCoeffsNetwork)]
        if len(heads) != 1 or heads[0].core_basis_ratio:
            raise ValueError('Compilation currently requires one density coefficient head without core basis')
        for module in self.modules():
            if getattr(module, 'normalize', 0):
                raise ValueError('Compilation currently supports normalize=0 only')
            if getattr(module, 'verbose', 0) or getattr(module, 'memory', False) or getattr(module, 'timing', False):
                raise ValueError('Disable verbose, memory and timing instrumentation before compilation')
        head = heads[0]
        if head.integral_constraint is not None:
            raise ValueError('Compilation currently excludes coefficient integral constraints')
        head.__class__ = _StaticCoefficients
        head._species = species
        head.sph_dict = {key: value.to(example['positions'].device) for key, value in head.sph_dict.items()}
        head.rad_dict = {key: value.to(example['positions'].device) for key, value in head.rad_dict.items()}
        energy, density = self.property_models['energy'], self.property_models['density']
        if type(energy) is not SphericalHarmonicsEnergyNetwork or type(density) is not DensityExpansion:
            raise ValueError('Compilation requires the original spherical energy and density expansion modules')
        energy.__class__ = _StaticEnergy
        energy.register_buffer('_active', example['atom_mask'].nonzero().flatten(), persistent=False)
        density.__class__ = _StaticExpansion
        density._species = species
        if density.expansion_constraint not in (None, 'sq', 'abs', 'sp') or density.integral_constraint not in (None, 'grid'):
            raise ValueError('Unsupported density expansion constraint')

    def forward(self, inputs, mode):
        atoms = dict(inputs)
        atoms = self.density_repr_model(atoms)
        if mode == 'density':
            atoms = self.property_models['density'](atoms)
            if self.remove_atom_density:
                atoms['density'] = atoms['density'] + atoms['atom_density']
                atoms['density_grad'] = atoms['density_grad'] + atoms['atom_density_grad']
            return {key: atoms[key] for key in ('density', 'density_grad')}
        atoms = self.property_models['energy'](atoms)
        return {'energy': atoms['energy']}


class FunctionalDenSNet:
    """Compile fixed-topology applications with explicit, replaceable parameters.

    Inputs use the original prepared model units. Returned E/F are raw training
    units; callers apply evaluation unit/scaling conversions outside this API.
    Targets use matching units. Update modes return all active gradients and a
    complete parameter mapping; frozen leaves retain their original tensor.
    """
    MODES = ('ef', 'density', 'density_update', 'energy_force_update')

    def __init__(self, model, example_inputs):
        self.module = _Application(model, example_inputs)
        self.buffers = dict(self.module.named_buffers())
        self._params = {name: value.detach().clone() for name, value in self.module.named_parameters()}
        self.density_names = tuple(name for name in self._params if name.startswith('density_repr_model.'))
        self.energy_names = tuple(name for name in self._params if name.startswith('property_models.energy.'))
        self.trainable_names = tuple(name for name, value in self.module.named_parameters() if value.requires_grad)
        self._topology = {key: example_inputs[key].detach().clone() for key in (
            'atom_numbers', 'batch_atom_numbers', 'atom_mask', 'batch_atom_mask',
            'batch_idx_pos', 'atom_batch_idx', 'idx_i', 'idx_j')}
        self._input_shapes = {key: value.shape for key, value in example_inputs.items() if isinstance(value, torch.Tensor)}

    def validate_inputs(self, inputs):
        """Host boundary check; topology changes require a new adapter.

        Call before invoking predict directly or timing an operation's unchecked
        kernel. Coordinates, grid values and target values may change freely.
        """
        for key, expected in self._topology.items():
            if key not in inputs or not torch.equal(inputs[key], expected):
                raise ValueError(f'Fixed topology changed: {key}; construct a new adapter')
        for key in ('positions', 'batch_positions', 'coords', 'coord_weights'):
            if key in self._input_shapes and (key not in inputs or inputs[key].shape != self._input_shapes[key]):
                raise ValueError(f'Fixed input shape changed: {key}; construct a new adapter')

    def parameters(self):
        return {name: value.clone() for name, value in self._params.items()}

    def predict(self, params, inputs, mode='ef'):
        if mode == 'density':
            return torch.func.functional_call(self.module, (params, self.buffers), (inputs, 'density'), strict=True)
        if mode != 'ef':
            raise ValueError(mode)
        def total_energy(positions):
            current = dict(inputs, positions=positions)
            output = torch.func.functional_call(self.module, (params, self.buffers), (current, 'energy'), strict=True)
            return output['energy'].sum(), output
        derivative, output = torch.func.grad_and_value(total_energy, has_aux=True)(inputs['positions'])
        return dict(output[1], forces=-derivative)

    def operation(self, mode, *, backend='inductor', learning_rate=1e-5, loss_weights=None, compile=True):
        """Return a checked application callable; ``.unchecked`` is its kernel.

        Use unchecked only after validate_inputs for every supplied layout/frame.
        Fullgraph failures propagate; compile=False is an explicit eager adapter
        diagnostic and is never a fallback from a compiled application.
        """
        if mode not in self.MODES:
            raise ValueError(mode)
        prediction_mode = 'density' if mode.startswith('density') else 'ef'
        loss_weights = dict(loss_weights or {})
        output_keys = ('density', 'density_grad') if prediction_mode == 'density' else ('energy', 'forces')
        if any(key not in output_keys for key in loss_weights):
            raise ValueError('Loss weights contain a field not predicted by this mode')
        stage_names = self.density_names if prediction_mode == 'density' else self.energy_names
        active_names = tuple(name for name in stage_names if name in self.trainable_names)
        if mode.endswith('update') and not active_names:
            raise ValueError(f'No trainable parameters in the {prediction_mode} stage')
        def operation(params, inputs, targets=None):
            if not mode.endswith('update'):
                return self.predict(params, inputs, prediction_mode)
            active = {name: params[name] for name in active_names}
            fixed = {name: value for name, value in params.items() if name not in active_names}
            def objective(current):
                output = self.predict(dict(fixed, **current), inputs, prediction_mode)
                loss = sum(loss_weights.get(key, 1.0) * (output[key] - targets[key]).square().mean() for key in output)
                return loss, output
            grads, (loss, outputs) = torch.func.grad_and_value(objective, has_aux=True)(active)
            updated = {name: value - learning_rate * grads[name] if name in grads else value for name, value in params.items()}
            return dict(outputs, loss=loss, gradients=grads, parameters=updated)
        kernel = operation
        if compile:
            if torch._dynamo.config.suppress_errors:
                raise RuntimeError('Strict compilation requires torch._dynamo.config.suppress_errors=False')
            kernel = torch.compile(operation, backend=backend, fullgraph=True, dynamic=False)
        def validated(params, inputs, targets=None):
            self.validate_inputs(inputs)
            return kernel(params, inputs, targets)
        validated.unchecked = kernel
        return validated
