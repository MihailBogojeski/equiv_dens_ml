# !/usr/bin/env python3
import os
import torch
import torch.nn as nn
from equiv_dens.nn.dft_network import DFTNetwork
from equiv_dens.nn.representation.spherical_harmonic import EquivariantSphericalHarmonics,\
    TransferableEquivariantSphericalHarmonics
from equiv_dens.nn.property_output.energy import ComplexEnergyNetwork, SimpleEnergyNetwork,\
    SphericalHarmonicsEnergyNetwork, SimpleEnergyNetworkv2, SimpleRepresentationEnergyNetwork,\
    RepresentationEnergyNetwork, TransferableSphericalHarmonicsEnergyNetwork
from equiv_dens.nn.property_output.density import DensityCoeffsNetwork, DensityExpansion,\
    DummyCoeffsNetwork, TransferableDensityCoeffsNetwork, TransferableDensityExpansion
from equiv_dens.nn.property_output.density_legacy import DensityCoeffsNetwork as LegacyDensityCoeffsNetwork
from equiv_dens.nn.property_output.density_legacy import DensityExpansion as LegacyDensityExpansion
from equiv_dens.nn.property_output.dipole_moment import DipoleMomentCalc
from equiv_dens.nn.modules.clebsch_gordan import ClebschGordanMatrix
from equiv_dens.utils.scaling import UnitConversion, VarianceScaling
import equiv_dens.utils.base as utils
from equiv_dens.utils.grids import dftpy_grid, CubicalGrid
from dftpy.pseudo import LocalPseudo
from equiv_dens.density_functionals.LDA import LDAFunctional

import numpy as np


def load_model(args, dataset, train=False):
    use_gpu = args.use_gpu and torch.cuda.is_available()
    z_vals = dataset.atoms['atom_numbers']
    if args.energy_min_weight > 0:
        grid_extent = np.array([args.cube_extent] * 3)
        grid_cl = CubicalGrid(dataset.atoms, nx=args.cube_size, ny=args.cube_size, nz=args.cube_size,
                              origin=[0, 0, 0], extent=utils.angstrom_to_bohr(grid_extent),
                              use_gpu=use_gpu, dtype=args.dtype)

        cube_gap = utils.angstrom_to_bohr(args.cube_extent) / args.cube_size
        print('cube_extent', utils.angstrom_to_bohr(args.cube_extent))
        print('cube_size', args.cube_size)
        print('cube_gap', cube_gap)
        grid = dftpy_grid(np.diag(utils.angstrom_to_bohr(grid_extent)), cube_gap)
        # print('grid.lattice', grid.lattice)
        # print('grid size', grid.r.shape)
        # print('ions lattice', dataset.ions[0].pos.cell.lattice)

        file_names = {'H': 'H.pbe-kjpaw_psl.0.1.UPF', 'C': 'C.pbe-kjpaw_psl.0.1.UPF',
                      'O': 'O.pbe-n-kjpaw_psl.0.1.UPF'}
        PP_list = {key: os.path.join(args.pseudo_pot_path, file_names[key]) for key in file_names.keys()}
        # print('pseudo potentials', PP_list)
        pseudo_pot = LocalPseudo(grid=grid, ions=None, PP_list=PP_list, PME=True)
        pseudo_pot.restart(grid=grid, ions=dataset.ions[0])

        dataset.add_fixed_properties({'grid': grid_cl, 'dftpy_grid': grid, 'pseudo_pot': pseudo_pot})

        z_vals = []
        print('ions0', dataset.ions[0])
        for t in dataset.atoms['atom_types']:
            z_vals.append(dataset.ions[0].Zval[t])
        z_vals = np.array(z_vals)
        print(dataset.atoms['atom_numbers'])
        print(z_vals)
    # define model
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
    if args.output_scaling:
        force_scaling = VarianceScaling(conversions_in.en_conversion_func(dataset.forces)/conversions_in.dist_conversion_func(1))
    print('conversions in', conversions_in.en_conversion_func)
    print('conversions out', conversions_out.en_conversion_func)

    if args.transferable_model:
        repr_class = TransferableEquivariantSphericalHarmonics
    else:
        repr_class = EquivariantSphericalHarmonics

    repr_model = repr_class(
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

    if args.legacy:
        density_coeffs_network = LegacyDensityCoeffsNetwork
        density_expansion = LegacyDensityExpansion
    else:
        if args.transferable_model:
            density_coeffs_network = TransferableDensityCoeffsNetwork
            density_expansion = TransferableDensityExpansion
        else:
            density_coeffs_network = DensityCoeffsNetwork
            density_expansion = DensityExpansion

    dens_model = density_coeffs_network(
        orbitals=dataset.orbitals,
        order=args.order[-1],
        num_features=args.num_features,
        positive_coeffs=args.positive_coeffs,
        clebsch_gordan=clebsch_gordan,
        verbose=args.verbose,
        timing=args.timing,
        init_coeffs=dataset.L0_coeffs,
        pred_radial_coeffs=args.pred_radial_coeffs,
    )

    expansion_model = density_expansion(dataset.orbitals, radial_coeffs=dataset.radial_coeffs,
                                        expansion_constraint=args.expansion_constraint,
                                        integral_constraint=args.integral_constraint,
                                        integral_scale=args.integral_scale,
                                        softmax_norm=args.softmax_norm, n_electrons=sum(z_vals),
                                        verbose=args.verbose,
                                        timing=args.timing,
                                        grid_scaling_factor=args.grid_scaling_factor,
                                        )

    calculate_forces = args.forces_weight > 0

    if args.num_energy_features is None:
        args.num_energy_features = args.num_features

    if args.energy_model == 'spherical':
        if args.transferable_model:
            en_class = TransferableSphericalHarmonicsEnergyNetwork
        else:
            en_class = SphericalHarmonicsEnergyNetwork
        print('building spherical harmonic energy model')
        en_model = en_class(
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
    elif args.energy_model == 'repr':
        print('building representation energy model')
        en_model = RepresentationEnergyNetwork(
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
    elif args.energy_model == 'simple_repr':
        print('building simple representation energy model')
        en_model = SimpleRepresentationEnergyNetwork(
            orbitals=dataset.orbitals,
            order=args.order,
            num_features=args.num_energy_features,
            activation=args.activation,
            clebsch_gordan=clebsch_gordan,
            calculate_forces=calculate_forces,
            verbose=args.verbose,
            timing=args.timing,
        )
    else:
        args.energy_model = None

    if args.energy_min_weight > 0:
        functional = LDAFunctional(z_vals, verbose=args.verbose,
                                   energy_offset=args.energy_offset,
                                   store_energy=(args.energy_model is None))
        functional_en_model = nn.Sequential(expansion_model, functional)

    density_model = nn.Sequential(repr_model, dens_model)
    if args.dummy_coeff_model:
        density_model = DummyCoeffsNetwork(orbitals=dataset.orbitals,
                                           order=args.order[-1],
                                           num_features=args.num_features,
                                           positive_coeffs=args.positive_coeffs,
                                           clebsch_gordan=clebsch_gordan,
                                           verbose=args.verbose,
                                           timing=args.timing,
                                           init_coeffs=dataset.L0_coeffs,
                                           pred_radial_coeffs=args.pred_radial_coeffs,
                                           )

    property_models = {}
    calculate_forces_dict = {}
    if args.density_weight > 0:
        property_models['density'] = expansion_model
        calculate_forces_dict['density'] = False
    if args.energy_min_weight > 0:
        property_models['energy_min'] = functional_en_model
        calculate_forces_dict['energy_min'] = False
    if args.energy_model is not None:
        property_models['energy'] = en_model
        calculate_forces_dict['energy'] = calculate_forces
    if args.dipole_moment_weight:
        property_models['dipole_moment'] = DipoleMomentCalc()
        calculate_forces_dict['dipole_moment'] = False

    model = DFTNetwork(density_model, property_models,
                       calculate_forces_dict=calculate_forces_dict,
                       verbose=args.verbose,
                       conversions_in=conversions_in,
                       conversions_out=conversions_out,
                       scaling=force_scaling,
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
        print('best_model_path', best_model_path)
        print('args restart', args.restart)
        print('best_model_path', best_model_path)
        state_dict_path = os.path.join(args.restart, best_model_path)
        print('state_dict_path', state_dict_path)
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if not train and args.load_from is not None:
            print('loading from', args.load_from)
            load_code = args.load_from.split('_')[-1]
            model_dict = torch.load(os.path.join(args.load_from, 'best_' + load_code + '.pth'), map_location='cpu')

            for key in model_dict.keys():
                if 'property_models.density' in key:
                    state_dict[key] = model_dict[key]
        model.load_state_dict(state_dict)
    if not train:
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
