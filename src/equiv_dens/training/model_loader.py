# !/usr/bin/env python3
import os
import torch
import torch.nn as nn
from equiv_dens.nn.dft_network import DFTNetwork
from equiv_dens.nn.representation.spherical_harmonic import EquivariantSphericalHarmonics
from equiv_dens.nn.property_output.energy import ComplexEnergyNetwork, SimpleEnergyNetwork,\
    SphericalHarmonicsEnergyNetwork, SimpleEnergyNetworkv2
from equiv_dens.nn.property_output.density import DensityCoeffsNetwork, DensityExpansion
from equiv_dens.nn.property_output.dipole_moment import DipoleMomentCalc
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix


def load_model(args, dataset):
    z_vals = dataset.atoms['atom_numbers']
    # define model
    clebsch_gordan = ClebschGordanMatrix()
    repr_model = EquivariantSphericalHarmonics(
        orbitals=dataset.orbitals,
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
        basis_functions=args.basis_functions,
        cutoff=args.cutoff,
        activation=args.activation,
        clebsch_gordan=clebsch_gordan,
        verbose=args.verbose,
        timing=args.timing,
    )
    dens_model = DensityCoeffsNetwork(
        orbitals=dataset.orbitals,
        order=args.order[-1],
        num_features=args.num_features,
        positive_coeffs=args.positive_coeffs,
        clebsch_gordan=clebsch_gordan,
        verbose=args.verbose,
        timing=args.timing,
    )

    expansion_model = DensityExpansion(dataset.orbitals, radial_coeffs=dataset.radial_coeffs,
                                       expansion_constraint=args.expansion_constraint,
                                       integral_constraint=args.integral_constraint,
                                       integral_scale=args.integral_scale,
                                       softmax_norm=args.softmax_norm, n_electrons=sum(z_vals),
                                       verbose=args.verbose,
                                       timing=args.timing,
                                       )

    calculate_forces = True

    if args.num_energy_features is None:
        args.num_energy_features = args.num_features

    if args.energy_model == 'spherical':
        print('building spherical harmonic energy model')
        en_model = SphericalHarmonicsEnergyNetwork(
            orbitals=dataset.orbitals,
            order=args.order_en,
            mixing_order=args.mixing_order_en,
            num_features=args.num_energy_features,
            num_basis_functions=args.num_basis_functions,
            num_modules=args.num_modules,
            num_residual_pre_x=args.num_residual_pre_x,
            num_residual_post_x=args.num_residual_post_x,
            num_residual_pre_vi=args.num_residual_pre_vi,
            num_residual_pre_vj=args.num_residual_pre_vj,
            num_residual_post_v=args.num_residual_post_v,
            num_residual_output=args.num_residual_output,
            num_radial_components=args.num_radial_components,
            basis_functions=args.basis_functions,
            cutoff=args.cutoff,
            activation=args.activation,
            clebsch_gordan=clebsch_gordan,
            calculate_forces=calculate_forces,
            verbose=args.verbose,
            timing=args.timing,
        )
    elif args.energy_model == 'complex':
        print('building complex energy model')
        en_model = ComplexEnergyNetwork(
            orbitals=dataset.orbitals,
            num_features=args.num_energy_features,
            num_basis_functions=args.num_basis_functions,
            num_modules=args.num_modules,
            num_residual_pre_x=args.num_residual_pre_x,
            num_residual_post_x=args.num_residual_post_x,
            num_residual_pre_vi=args.num_residual_pre_vi,
            num_residual_pre_vj=args.num_residual_pre_vj,
            num_residual_post_v=args.num_residual_post_v,
            num_residual_output=args.num_residual_output,
            num_radial_components=args.num_radial_components,
            basis_functions=args.basis_functions,
            cutoff=args.cutoff,
            activation=args.activation,
            calculate_forces=calculate_forces,
            verbose=args.verbose,
            timing=args.timing,
        )
    elif args.energy_model == 'simple':
        print('building simple energy model')
        en_model = SimpleEnergyNetwork(
            orbitals=dataset.orbitals,
            num_features=args.num_energy_features,
            num_layers=args.num_energy_output,
            activation=args.activation,
            calculate_forces=calculate_forces,
            verbose=args.verbose,
            timing=args.timing,
        )
    elif args.energy_model == 'simple2':
        print('building simple energy model')
        en_model = SimpleEnergyNetworkv2(
            order=args.order[-1],
            orbitals=dataset.orbitals,
            num_features=args.num_energy_features,
            activation=args.activation,
            calculate_forces=calculate_forces,
            verbose=args.verbose,
            clebsch_gordan=clebsch_gordan,
            timing=args.timing,
        )
    else:
        args.energy_model = None

    # if loss_weights['energy_min'] > 0:
    #     functional = LDAFunctional(z_vals, verbose=args.verbose,
    #                                energy_offset=args.energy_offset,
    #                                store_energy=(args.energy_model is None))
    #     functional_en_model = nn.Sequential(expansion_model, functional)
    density_model = nn.Sequential(repr_model, dens_model)

    property_models = {}
    calculate_forces_dict = {}
    if args.density_weight > 0:
        property_models['density'] = expansion_model
        calculate_forces_dict['density'] = False
    # if loss_weights['energy_min'] > 0:
    #     property_models['energy_min'] = functional_en_model
    #     calculate_forces_dict['energy_min'] = False
    if args.energy_model is not None:
        property_models['energy'] = en_model
        calculate_forces_dict['energy'] = calculate_forces
    if args.dipole_moment_weight:
        property_models['dipole_moment'] = DipoleMomentCalc()
        calculate_forces_dict['dipole_moment'] = False

    model = DFTNetwork(density_model, property_models, calculate_forces_dict=calculate_forces_dict, verbose=args.verbose)
    # print('dft network', model)

    print('args restart', args.restart)
    print('best_model_path', args.best_model_path)
    state_dict_path = os.path.join(args.restart, args.best_model_path)
    print('state_dict_path', state_dict_path)
    state_dict = torch.load(state_dict_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print('dtype type', type(args.dtype))
    model.to(args.dtype)
    if args.use_gpu:
        print('using GPU')
        model.cuda()
    # if there are multiple GPUs, wrap the model in DataParallel
    # "module" is used whenever direct access is needed, e.g. for parameters,
    # whereas "model" may be DataParallel and is used for inference only

    if args.use_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model
