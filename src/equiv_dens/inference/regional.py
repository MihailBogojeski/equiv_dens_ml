"""Two strict compiled regions for original DenSNet density-stage SGD.

This application is regional, not a single fullgraph application. The first
region predicts coefficients, including learned radial widths and scales. The
second evaluates the original density, analytical spatial gradient and loss.
Ordinary first-order autograd joins their AOT backward graphs; explicit SGD and
parameter preparation run outside compilation. Energy/force training is excluded
because it would require differentiating an AOT backward graph again.
"""
from __future__ import annotations

import torch
from torch import nn

from equiv_dens.inference.compiled import FunctionalDenSNet


class _Coefficients(nn.Module):
    def __init__(self, application):
        super().__init__()
        self.density_repr_model = application.density_repr_model

    def forward(self, inputs):
        atoms = self.density_repr_model(dict(inputs))
        return tuple(atoms[key] for key in ('spherical_coeffs', 'radial_width', 'radial_scale'))


class _DensityLoss(nn.Module):
    def __init__(self, application, weights):
        super().__init__()
        self.property_models = nn.ModuleDict({'density': application.property_models['density']})
        self.remove_atom_density = application.remove_atom_density
        self.weights = weights

    def forward(self, inputs, coefficients, targets):
        atoms = dict(inputs)
        atoms.update(zip(('spherical_coeffs', 'radial_width', 'radial_scale'), coefficients))
        atoms = self.property_models['density'](atoms)
        if self.remove_atom_density:
            atoms['density'] = atoms['density'] + atoms['atom_density']
            atoms['density_grad'] = atoms['density_grad'] + atoms['atom_density_grad']
        output = {key: atoms[key] for key in ('density', 'density_grad')}
        loss = sum(weight * (output[key] - targets[key]).square().mean()
                   for key, weight in zip(output, self.weights))
        return dict(output, loss=loss)


class RegionalDensityDenSNet(FunctionalDenSNet):
    """Fixed-topology density training with two fullgraph compiled regions.

    Uses the same parameter names, buffers and raw units as FunctionalDenSNet.
    Only density_update is supported. Inputs and parameters may change values
    between calls; topology and shapes must be validated before unchecked use.
    Outputs and active SGD parameters are detached, and source-frozen leaves are kept
    bitwise unchanged. This API does not support higher-order differentiation of
    its update, whole-application fullgraph claims, or energy/force training.
    """
    MODES = ('density_update',)
    compilation = {'kind': 'regional', 'regions': 2, 'region_fullgraph': True,
                   'full_application_fullgraph': False, 'dynamic': False}

    def operation(self, mode='density_update', *, backend='inductor', learning_rate=1e-5,
                  loss_weights=None, compile=True):
        if mode not in self.MODES:
            raise ValueError('Regional compilation supports density_update only')
        loss_weights = dict(loss_weights or {})
        if any(key not in ('density', 'density_grad') for key in loss_weights):
            raise ValueError('Loss weights contain a field not predicted by this mode')
        active_names = tuple(name for name in self.density_names if name in self.trainable_names)
        if not active_names:
            raise ValueError('No trainable parameters in the density stage')
        active_name_set = set(active_names)
        coefficients = _Coefficients(self.module)
        density_loss = _DensityLoss(self.module, tuple(loss_weights.get(key, 1.0)
                                                      for key in ('density', 'density_grad')))
        coefficient_buffers = {name: self.buffers[name] for name, _ in coefficients.named_buffers()}
        loss_buffers = {name: self.buffers[name] for name, _ in density_loss.named_buffers()}

        def coefficient_region(params, inputs):
            return torch.func.functional_call(coefficients, (params, coefficient_buffers),
                                              (inputs,), strict=True)

        def loss_region(inputs, values, targets):
            return torch.func.functional_call(density_loss, ({}, loss_buffers),
                                              (inputs, values, targets), strict=True)

        if compile:
            if torch._dynamo.config.suppress_errors:
                raise RuntimeError('Strict compilation requires torch._dynamo.config.suppress_errors=False')
            coefficient_region = torch.compile(coefficient_region, backend=backend, fullgraph=True, dynamic=False)
            loss_region = torch.compile(loss_region, backend=backend, fullgraph=True, dynamic=False)

        def operation(params, inputs, targets=None):
            if targets is None:
                raise ValueError('Density update requires density and density_grad targets')
            with torch.enable_grad():
                # Separate leaves avoid accumulating gradients in caller tensors
                # and keep requires_grad guards identical after every SGD call.
                leaves = {name: params[name].detach().requires_grad_(name in active_name_set)
                          for name in self.density_names}
                values = coefficient_region(leaves, inputs)
                output = loss_region(inputs, values, targets)
                active = tuple(leaves[name] for name in active_names)
                if output['loss'].requires_grad:
                    torch.autograd.backward(output['loss'], inputs=active)
                gradients = {name: (torch.zeros_like(leaves[name]) if leaves[name].grad is None
                                    else leaves[name].grad.detach()) for name in active_names}
            with torch.no_grad():
                updated = {name: value - learning_rate * gradients[name] if name in gradients else value
                           for name, value in params.items()}
            return dict({key: value.detach() for key, value in output.items()},
                        gradients=gradients, parameters=updated)

        def validated(params, inputs, targets=None):
            self.validate_inputs(inputs)
            return operation(params, inputs, targets)
        validated.unchecked = operation
        validated.compilation = dict(self.compilation, enabled=compile, backend=backend)
        return validated
