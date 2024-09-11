# !/usr/bin/env python3
import os
import torch
import torch.nn as nn
from equiv_dens.nn.dft_network import DFTNetwork
from equiv_dens.nn.representation.spherical_harmonic import EquivariantSphericalHarmonics
from equiv_dens.nn.property_output.energy import SphericalHarmonicsEnergyNetwork,\
    SphericalLinearEnergyNetwork, RepresentationEnergyNetwork
from equiv_dens.nn.property_output.density import DensityCoeffsNetwork, DFDensityCoeffs,\
    FreeAtomDensityCoeffs, DensityExpansion
from equiv_dens.nn.property_output.dipole_moment import DipoleMomentCalc, DipoleMomentIntorCalc
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.utils.scaling import UnitConversion, VarianceScaling
import equiv_dens.utils.base as utils
from equiv_dens.utils.grids import CubicalGrid
from equiv_dens.density_functionals.LDA import LDAFunctional

import numpy as np
import torch._dynamo

torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')


def load_model(args, dataset, train=False):
    use_gpu = args.use_gpu and torch.cuda.is_available()
    z_vals = dataset.atoms['atom_numbers']
    clebsch_gordan = ClebschGordanMatrix()
    print('args energy_unit_in', args.energy_unit_in)
    print('args energy_unit_out', args.energy_unit_out)
    conversions_in = UnitConversion(
        en_conversion_func=getattr(utils, args.energy_unit_in + '_to_kcal'),
        dist_conversion_func=getattr(utils, args.distance_unit_in + '_to_angstrom'))
    conversions_out = UnitConversion(
        en_conversion_func=getattr(utils, 'kcal_to_' + args.energy_unit_out),
        dist_conversion_func=getattr(utils, 'angstrom_to_' + args.distance_unit_out))
    force_scaling = VarianceScaling()
    if args.output_scaling and 'forces' in dataset.required_properties:
        force_scaling = VarianceScaling(conversions_in.en_conversion_func(dataset.forces)/conversions_in.dist_conversion_func(1))
    print('conversions in', conversions_in.en_conversion_func)
    print('conversions out', conversions_out.en_conversion_func)

    if args.density_from_df or args.density_from_free_atoms:
        repr_model = nn.Identity()

        if args.density_from_df:
            dens_model = DFDensityCoeffs(
                orbital_basis=dataset.orbital_basis_num,
                radial_coeffs=dataset.radial_coeffs,
                dtype=args.dtype,
            )
        else:
            dens_model = FreeAtomDensityCoeffs(
                orbital_basis=dataset.orbital_basis_num,
                radial_coeffs=dataset.radial_coeffs,
                atom_dens=dataset.atom_dens,
                dtype=args.dtype,
            )
        if args.density_weight + args.dipole_moment_weight > 0:
            density_expansion = DensityExpansion
            expansion_model = density_expansion(dataset.orbital_basis_num,
                                                expansion_constraint=args.expansion_constraint,
                                                integral_constraint=args.integral_constraint,
                                                verbose=args.verbose,
                                                timing=args.timing,
                                                memory=args.memory,
                                                grid_scaling_factor=args.grid_scaling_factor,
                                                density_grad=args.density_grad,
                                                )
    # core density expansion not applicable here
    else:
        repr_class = EquivariantSphericalHarmonics

        repr_model = repr_class(
            orbital_basis=dataset.orbital_basis_num,
            order=args.order,
            num_features=args.num_features,
            num_basis_functions=args.num_basis_functions,
            num_modules=args.num_modules,
            num_residual_pre_x=args.num_residual_pre_x,
            num_residual_post_x=args.num_residual_post_x,
            num_residual_pre_vi=args.num_residual_pre_vi,
            num_residual_pre_vj=args.num_residual_pre_vj,
            num_residual_post_v=args.num_residual_post_v,
            num_residual_output=args.num_residual_output,
            num_radial_components=args.num_radial_components,
            num_neighbours=args.num_neighbours,
            basis_functions=args.basis_functions,
            cutoff=args.cutoff,
            activation=args.activation,
            clebsch_gordan=clebsch_gordan,
            verbose=args.verbose,
            timing=args.timing,
            memory=args.memory,
            normalize=args.normalize,
            parity=args.parity_dens,
            nonmixing_interaction=args.nonmixing_interaction,
            nonmixing_interaction_residual=args.nonmixing_interaction_residual,
        )

        density_coeffs_network = DensityCoeffsNetwork
        density_expansion = DensityExpansion


        if args.density_coeffs:
            dens_model = density_coeffs_network(
                orbital_basis=dataset.orbital_basis_num,
                order=args.order[-1],
                num_features=args.num_features,
                positive_coeffs=args.positive_coeffs,
                integral_constraint=args.integral_constraint,
                clebsch_gordan=clebsch_gordan,
                verbose=args.verbose,
                timing=args.timing,
                memory=args.memory,
                init_coeffs=dataset.L0_coeffs,
                coeff_weights=dataset.coeff_weights,
                pred_radial_coeffs=args.pred_radial_coeffs,
                init_radial_coeffs=dataset.radial_coeffs,
                ml_width_min=args.ml_width_min,
                ml_width_max=args.ml_width_max,
                scale_sph_order=args.scale_sph_order,
                normalize=args.normalize,
                parity=args.parity_dens,
                remove_atom_density=args.remove_atom_density,
                nonmixing=args.nonmixing_interaction,
                nonmixing_bias=args.nonmixing_interaction_residual,
                linear_out=args.remove_atom_density,
            )

        if args.density_weight + args.dipole_moment_weight > 0:
            expansion_model = density_expansion(dataset.orbital_basis_num,
                                                expansion_constraint=args.expansion_constraint,
                                                integral_constraint=args.integral_constraint,
                                                verbose=args.verbose,
                                                timing=args.timing,
                                                memory=args.memory,
                                                grid_scaling_factor=args.grid_scaling_factor,
                                                density_grad=args.density_grad,
                                                )
            if args.core_density_basis > 0:
                core_coeffs_model = density_coeffs_network(
                    orbital_basis=dataset.orbital_basis_num,
                    order=args.order[-1],
                    num_features=args.num_features,
                    positive_coeffs=args.positive_coeffs,
                    clebsch_gordan=clebsch_gordan,
                    verbose=args.verbose,
                    timing=args.timing,
                    memory=args.memory,
                    init_coeffs=dataset.L0_coeffs,
                    init_radial_coeffs=dataset.radial_coeffs,
                    ml_width_min=args.ml_width_min,
                    ml_width_max=args.ml_width_max,
                    coeff_weights=dataset.coeff_weights,
                    pred_radial_coeffs=args.pred_radial_coeffs,
                    scale_sph_order=args.scale_sph_order,
                    normalize=args.normalize,
                    parity=args.parity_dens,
                    core_basis_ratio=args.core_density_basis,
                    remove_atom_density=args.remove_atom_density,
                    linear_out=args.remove_atom_density,
                )
        else:
            expansion_model = None

    calculate_forces = args.forces_weight > 0

    if args.num_energy_features is None:
        args.num_energy_features = args.num_features
    if args.num_en_basis_functions is None:
        args.num_en_basis_functions = args.num_basis_functions
    if args.num_en_modules is None:
        args.num_en_modules = args.num_modules

    if args.energy_weight + args.forces_weight > 0:
        if args.energy_model == 'spherical':
            en_class = SphericalHarmonicsEnergyNetwork
            print('building spherical harmonic energy model')
            en_model = en_class(
                orbital_basis=dataset.orbital_basis_num,
                order=args.order_en,
                mixing_order=args.mixing_order_en,
                num_features=args.num_energy_features,
                num_basis_functions=args.num_en_basis_functions,
                num_modules=args.num_en_modules,
                num_residual_pre_x=args.num_residual_pre_x,
                num_residual_post_x=args.num_residual_post_x,
                num_residual_pre_vi=args.num_residual_pre_vi,
                num_residual_pre_vj=args.num_residual_pre_vj,
                num_residual_post_v=args.num_residual_post_v,
                num_residual_output=args.num_residual_output,
                num_radial_components=args.num_radial_components,
                num_neighbours=args.num_neighbours,
                basis_functions=args.basis_functions,
                cutoff=args.cutoff,
                activation=args.activation,
                clebsch_gordan=clebsch_gordan,
                calculate_forces=calculate_forces,
                verbose=args.verbose,
                timing=args.timing,
                normalize=args.normalize_en,
                parity=args.parity_en,
            )
        elif args.energy_model == 'spherical_linear':
            print('building spherical linear energy model')
            en_model = SphericalLinearEnergyNetwork(
                orbital_basis=dataset.orbital_basis_num,
                order=args.order_en,
                num_features=args.num_energy_features,
                # how many modules are stacked for calculating atomic features (iterations)
                num_modules=args.num_en_modules,
                activation=args.activation,
                clebsch_gordan=clebsch_gordan,
                calculate_forces=calculate_forces,
                compressed_extraction=args.compressed_extraction,
                verbose=args.verbose,
                timing=args.timing,
                pred_radial_coeffs=args.pred_radial_coeffs,
                normalize=args.normalize_en,
                parity=args.parity_en,
            )
        elif args.energy_model == 'representation':
            print('building representation energy model')
            en_model = RepresentationEnergyNetwork(
                order=args.order_en,
                num_features=args.num_energy_features,
                activation=args.activation,
                clebsch_gordan=clebsch_gordan,
                calculate_forces=calculate_forces,
                verbose=args.verbose,
                timing=args.timing,
                parity=args.parity_en,
            )
        else:
            args.energy_model = None

    if args.energy_min_weight > 0:
        functional = LDAFunctional(z_vals, verbose=args.verbose,
                                   energy_offset=args.energy_offset,
                                   store_energy=(args.energy_model is None))
        functional_en_model = nn.Sequential(expansion_model, functional)

    if args.density_coeffs:
        density_model = nn.Sequential(repr_model, dens_model)
    else:
        density_model = repr_model

    property_models = {}
    calculate_forces_dict = {}
    print('density_weight', args.density_weight)
    print('dipole_moment_weight', args.dipole_moment_weight)
    if args.density_weight + args.dipole_moment_weight > 0:
        if args.core_density_basis > 0:
            property_models['core_density'] = core_coeffs_model
            calculate_forces_dict['core_density'] = False
        property_models['density'] = expansion_model
        calculate_forces_dict['density'] = False
    if args.energy_min_weight > 0:
        property_models['energy_min'] = functional_en_model
        calculate_forces_dict['energy_min'] = False
    if args.energy_weight + args.forces_weight > 0 and args.energy_model is not None:
        property_models['energy'] = en_model
        calculate_forces_dict['energy'] = calculate_forces
    if args.dipole_moment_weight:
        if args.dpm_intor:
            property_models['dipole_moment'] = DipoleMomentIntorCalc(orbital_basis=dataset.orbital_basis_num)
        else:
            property_models['dipole_moment'] = DipoleMomentCalc()
        calculate_forces_dict['dipole_moment'] = False

    model = DFTNetwork(density_model, property_models,
                       calculate_forces_dict=calculate_forces_dict,
                       verbose=args.verbose,
                       memory=args.memory,
                       conversions_in=conversions_in,
                       conversions_out=conversions_out,
                       scaling=force_scaling,
                       remove_atom_density=args.remove_atom_density,
                       )
    # print('dft network', model)
    if args.restart is not None:
        directory = args.restart  # load directory name
    # load latest checkpoint
        checkpoint_path = os.path.join(directory, 'checkpoints')  # checkpoint directory
        checkpoint = torch.load(os.path.join(
            checkpoint_path, 'latest_checkpoint.pth'), map_location='cpu')
        model_code = checkpoint['ID']
        best_model_path = 'best_' + model_code + '.pth'
        # print('best_model_path', best_model_path)
        # print('args restart', args.restart)
        # print('best_model_path', best_model_path)
        state_dict_path = os.path.join(args.restart, best_model_path)
        # print('state_dict_path', state_dict_path)
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if not train and args.load_from is not None and args.density_weight > 0:
            # print('loading from', args.load_from)
            load_code = args.load_from.split('_')[-1]
            model_dict = torch.load(os.path.join(args.load_from, 'best_' + load_code + '.pth'), map_location='cpu')
            for key in model_dict.keys():
                if 'property_models.density' in key:
                    print('key', key)
                    state_dict[key] = model_dict[key]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            for key in unexpected:
                if args.density_weight + args.df_weight > 0 \
                        and 'property_models.energy' not in key \
                        and 'property_models.density.init_' not in key \
                        and 'property_models.density.integral_scale' not in key \
                        and 'property_models.density.softmax_norm' not in key \
                        and 'property_models.core_density' not in key:
                    print('Unexpected keywords', key)
                    raise Exception('Unexpected keywords in density model state dict')
                elif args.energy_weight + args.forces_weight > 0 and 'density' not in key:
                    print('Unexpected keywords', key)
                    raise Exception('Unexpected keywords in energy model state dict')
        if len(missing) > 0 and not args.ignore_missing_keywords:
            for key in missing:
                if 'init_' not in key:
                    if args.df_weight > 0 and 'property_models.density' not in key:
                        print('Missing keywords', key)
                        raise Exception('Missing keywords in df model state dict')
                    elif args.density_weight > 0:
                        if 'property_models.density' not in key and not (args.core_density_basis > 0) \
                                and 'energy' not in key:
                            print('Missing keywords', key)
                            raise Exception('Missing keywords in density coeffs model state dict')
    if not train:
        # print('dtype type', type(args.dtype))
        model.to(args.dtype)
        if args.use_gpu:
            print('using GPU')
            model.cuda()
        # if there are multiple GPUs, wrap the model in DataParallel
        # "module" is used whenever direct access is needed, e.g. for parameters,
        # whereas "model" may be DataParallel and is used for inference only

        if args.use_gpu and args.multiple_gpus and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    if args.compile:
        model = torch.compile(model)

    return model
