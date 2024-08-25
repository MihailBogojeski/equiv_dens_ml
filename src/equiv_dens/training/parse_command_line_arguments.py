import sys
import argparse
import torch


# helper function used in order to support boolean command line arguments
def str2bool(s):
    if s.lower() in ('true', 't', '1'):
        return True
    elif s.lower() in ('false', 'f', '0'):
        return False
    else:
        return s


def parse_command_line_arguments(arg_file=None):
    # declare parser
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser._action_groups.pop()

    # argument for restarting runs
    args_restart = parser.add_argument_group("specification of a restart folder")
    args_restart.add_argument("--restart", metavar='FOLDER', type=str, default=None,
                              help="restart training from the given folder (all other arguments are ignored)")
    args_restart.add_argument("--load_from", metavar='STR', type=str, default=None,
                              help="initialize model from given pth file (other architecture hyperparameters are ignored)")
    args_restart.add_argument("--no_restore", metavar='True|False', type=str2bool, default=False,
                              choices=[True, False], help="Do not restore checkpoint.")
    args_restart.add_argument("--fix_arguments", metavar='True|False', type=str2bool, default=False,
                              choices=[True, False],
                              help="Do not change arguments after loading checkpoint (except hyperparams).")
    args_restart.add_argument("--args_file_name", metavar='STR', type=str, default=None,
                              help="Save filename of .txt file containing arguments for easier bookkeeping.")
    args_restart.add_argument("--ignore_missing_keywords", metavar='True|False', type=str2bool, default=False,
                              choices=[True, False], help="Ignore missing keywords when loading the model.")

    # arguments for neural network architecture hyperparameters
    args_hyperparams = parser.add_argument_group("neural network architecture hyperparameters")
    args_hyperparams.add_argument("--activation", metavar='STR', type=str, default='swish',
                                  choices=['ssp', 'swish'], help="which activation function to use (shifted softplus (ssp) or swish))")
    args_hyperparams.add_argument("--order", metavar='INT', type=int, default=[2], nargs='+', help="angular order of the feature vectors")
    args_hyperparams.add_argument("--mixing_order", metavar='INT', type=int, default=None, nargs='+', help="angular order for interactions")
    args_hyperparams.add_argument("--order_en", metavar='INT', type=int, default=[2], nargs='+',
                                  help="angular order of the feature vectors for energy prediction")
    args_hyperparams.add_argument("--mixing_order_en", metavar='INT', type=int, default=None, nargs='+',
                                  help="angular order for interactions for energy prediction")
    args_hyperparams.add_argument("--num_features", metavar='INT', type=int, default=32, help="dimensionality of feature vectors")
    args_hyperparams.add_argument("--num_basis_functions", metavar='INT', type=int, default=32, help="number of radial basis functions")
    args_hyperparams.add_argument("--num_en_basis_functions", metavar='INT', type=int, default=None, help="number of radial basis functions for energy prediction")
    args_hyperparams.add_argument("--num_radial_components", metavar='INT', type=int, default=32,
                                  help="number of radial basis components used for the density radial functions")
    args_hyperparams.add_argument("--num_energy_features", metavar='INT', type=int, default=None,
                                  help="dimensionality of energy feature vectors")
    args_hyperparams.add_argument("--num_modules", metavar='INT', type=int, default=3,
                                  help="number of modules used in the neural network (interaction iterations)")
    args_hyperparams.add_argument("--num_en_modules", metavar='INT', type=int, default=None,
                                  help="number of modules used in the neural network (interaction iterations)")
    args_hyperparams.add_argument("--num_residual_pre_x", metavar='INT', type=int, default=1,
                                  help="number of residual blocks for refining atomic feature vectors pre interaction")
    args_hyperparams.add_argument("--num_residual_post_x", metavar='INT', type=int, default=1,
                                  help="number of residual blocks for refining atomic feature vectors post interaction")
    args_hyperparams.add_argument("--num_residual_pre_vi", metavar='INT', type=int, default=1,
                                  help="number of residual blocks for refining interaction feature vectors pre interaction (channel i)")
    args_hyperparams.add_argument("--num_residual_pre_vj", metavar='INT', type=int, default=1,
                                  help="number of residual blocks for refining interaction feature vectors pre interaction (channel j)")
    args_hyperparams.add_argument("--num_residual_post_v", metavar='INT', type=int, default=1,
                                  help="number of residual blocks for refining interaction feature vectors post interaction")
    args_hyperparams.add_argument("--num_residual_output", metavar='INT', type=int, default=1,
                                  help="number of residual blocks for refining output feature vectors")
    args_hyperparams.add_argument("--num_residual_pc",      metavar='INT',        type=int,      default=1,
                                  help="number of residual blocks for refining atomic feature vectors before constructing pair features (central atoms)")
    args_hyperparams.add_argument("--num_residual_pn",      metavar='INT',        type=int,      default=1,
                                  help="number of residual blocks for refining atomic feature vectors before constructing pair features (neighboring atoms)")
    args_hyperparams.add_argument("--num_residual_ii",      metavar='INT',        type=int,      default=1,
                                  help="number of residual blocks for predicting diagonal blocks (shared)")
    args_hyperparams.add_argument("--num_residual_ij",      metavar='INT',        type=int,      default=1,
                                  help="number of residual blocks for predicting off-diagonal blocks (shared)")
    args_hyperparams.add_argument("--num_residual_ao_ii", metavar='INT',        type=int,      default=1,
                                  help="number of residual blocks for predicting diagonal blocks (atomic orbital matrix)")
    args_hyperparams.add_argument("--num_residual_ao_ij", metavar='INT',        type=int,      default=1,
                                  help="number of residual blocks for predicting off-diagonal blocks (atomic orbital matrix)")
    args_hyperparams.add_argument("--num_energy_output", metavar='INT', type=int, default=2,
                                  help="number of layers for the simple energy output network.")
    args_hyperparams.add_argument("--num_neighbours", metavar='INT', type=int, default=1,
                                  help="Average number of neighbours.")
    args_hyperparams.add_argument("--basis_functions", metavar='STR', type=str, default='exp-bernstein',
                                  choices=['exp-bernstein', 'exp-gaussian', 'bernstein', 'gaussian'],
                                  help="which type of basis functions to use")
    args_hyperparams.add_argument("--cutoff", metavar='FLOAT', type=float, default=7.937658158457616,
                                  help="cutoff radius for interactions (default corresponds to 15 Bohr)")
    args_hyperparams.add_argument("--orthonormal_basis", metavar='True|False', type=str2bool, default=False,
                                  choices=[True, False],
                                  help="use orthonormal basis (overlap matrix is identity)" +
                                  " (will only work with appropriate reference data)")
    args_hyperparams.add_argument("--expansion_constraint", metavar='STR', type=str, default=None,
                                  help="type of constraint used on density to ensure positivity")
    args_hyperparams.add_argument("--integral_constraint", metavar='STR', type=str, default=None,
                                  choices=['None', 'grid', 'coeffs', 'coeffs_in_coeffs_net', 'True'], help="constrain density integral to number of electrons")
    args_hyperparams.add_argument("--integral_scale", metavar='True|False', type=str2bool, default=False,
                                  choices=[True, False], help="scale density integral by a limited amount")
    args_hyperparams.add_argument("--integral_min", metavar='FLOAT', type=float, default=None,
                                  help="Constrain density integral to a minimum value")
    args_hyperparams.add_argument("--energy_offset", metavar='True|False', type=str2bool, default=False,
                                  choices=[True, False],
                                  help="Whether to use an constant offset to adjust energy levels for different functionals")
    args_hyperparams.add_argument("--positive_coeffs", metavar='True|False', type=str2bool, default=True,
                                  choices=[True, False], help="Make the order 0 coefficients always positive.")
    args_hyperparams.add_argument("--compressed_extraction", metavar='True|False', type=str2bool, default=False,
                                  choices=[True, False],
                                  help="Extract the spherical harmonic coefficients from the features in a more compresesd way.")
    args_hyperparams.add_argument("--energy_model", metavar='STR', type=str, default=None,
                                  help="Use a neural network for energy prediction instead of functional.")
    args_hyperparams.add_argument("--pred_radial_coeffs", metavar='True|False', type=str2bool, default=True,
                                  choices=[True, False], help="Output predictions for the radial coefficients as well.")
    args_hyperparams.add_argument("--dummy_coeff_model", metavar='True|False', type=str2bool, default=False,
                                  choices=[True, False], help="Optimize coefficients directly without a neural network.")
    args_hyperparams.add_argument("--scale_sph_order", metavar='True|False', type=str2bool, default=False,
                                  choices=[True, False], help="Rescale predicted density coeffs based on spherical harmonic order.")
    args_hyperparams.add_argument("--normalize", metavar='INT', type=int, default=0,
                                  help="Internal normalization in network.")
    args_hyperparams.add_argument("--normalize_en", metavar='INT', type=int, default=0,
                                  help="Internal normalization in energy network.")
    args_hyperparams.add_argument("--parity_dens", metavar='True|False', type=str2bool, default=False,
                                  choices=[True, False], help="Include parity equivariance for density prediction.")
    args_hyperparams.add_argument("--parity_en", metavar='True|False', type=str2bool, default=False,
                                  choices=[True, False], help="Include parity equivariance for energy prediction.")
    args_hyperparams.add_argument("--ml_width_min", metavar='FLOAT', type=float, default=0,
                                  help="Minimum value of the learned factor that is multiplied to the initial width")
    args_hyperparams.add_argument("--ml_width_max", metavar='FLOAT', type=float, default=2,
                                  help="Maximum value of the learned factor that is multiplied to the initial width")
    args_hyperparams.add_argument("--nonmixing_interaction", metavar='True|False', type=str2bool, default=False,
                                  help="Use a non-mixing interaction as a final representation layer.")
    args_hyperparams.add_argument("--nonmixing_interaction_residual", metavar='True|False', type=str2bool, default=True,
                                  help="Whether the final nonmixing layer is residual or not.")
    args_hyperparams.add_argument("--density_coeffs", metavar='True|False', type=str2bool, default=True,
                                  help="Use density coefficients as part of representation.")
    args_hyperparams.add_argument("--ao_matrix_from_pair_features", metavar='True|False', type=str2bool, default=False,
                                  help="Predict atomic orbital matrix from pair features instead of atom features.")
    hyperparam_args = [act.dest for act in args_hyperparams._group_actions]

    # arguments for training
    args_training = parser.add_argument_group("training hyperparameters")
    args_training.add_argument("--max_steps", metavar='INT', type=int, help="maximum number of training steps")
    args_training.add_argument("--np_dataset", metavar='STR', type=str, help="filepath to atoms dataset")
    args_training.add_argument("--dens_dataset", metavar='STR', type=str, help="filepath to density dataset")
    args_training.add_argument("--np_dataset_test", metavar='STR', type=str, help="filepath to atoms test dataset")
    args_training.add_argument("--dens_dataset_test", metavar='STR', type=str, help="filepath to density test dataset")
    args_training.add_argument("--np_dataset_valid", metavar='STR', type=str, help="filepath to atoms validation dataset")
    args_training.add_argument("--dens_dataset_valid", metavar='STR', type=str, help="filepath to density validation dataset")
    args_training.add_argument("--pseudo_pot_path", metavar='STR', type=str, help="filepath to pseudo potentials")
    args_training.add_argument("--orbitals_file", metavar='STR', type=str, help="filepath to orbital basis")
    args_training.add_argument("--calc_basis_file", metavar='STR', type=str, help="filepath to calculation basis")
    args_training.add_argument("--radial_coeffs_file", metavar='STR', type=str, default=None,
                               help="filepath to initial radial coefficients")
    args_training.add_argument("--L0_coeffs_file", metavar='STR', type=str, default=None,
                               help="filepath to initial s-orbital coefficients")
    args_training.add_argument("--num_train", metavar='INT', type=int, help="size of training set")
    args_training.add_argument("--num_valid", metavar='INT', type=int, help="size of validation set")
    args_training.add_argument("--num_test", metavar='INT', default=None, type=int, help="size of validation set")
    args_training.add_argument("--ignore_split_indices", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Ignore data splitting indices if they are present.")
    args_training.add_argument("--density_subsamples", metavar='INT', type=int, default=10000,
                               help="number of grid samples used for evaluating density")
    args_training.add_argument("--train_batch_size", metavar='INT', type=int, default=1, help="batch size for training")
    args_training.add_argument("--valid_batch_size", metavar='INT', type=int, default=1, help="batch size for validation")
    args_training.add_argument("--test_batch_size", metavar='INT', type=int, default=1, help="batch size for validation")
    args_training.add_argument("--electron_num_batching", metavar='True|False', type=str2bool, default=False,
                               help="Use adaptive batching based on number of electrons.")
    args_training.add_argument("--batch_efficiency", metavar='FLOAT', type=float, default=0.7,
                               help="Efficiency of batching based on number of electrons, expressed as the fraction of non-padded entries.")
    args_training.add_argument("--num_workers", metavar='INT', type=int, default=0, help="number of worker threads for preparing batches")
    args_training.add_argument("--split_seed", metavar='INT', type=int, default=42,
                               help="seed for splitting the dataset in training, validation and test sets")
    args_training.add_argument("--optimizer", metavar='adam|amsgrad|sgd', type=str, default='sgd',
                               choices=['adam', 'amsgrad', 'sgd'], help="optimizer used for training")
    args_training.add_argument("--lookahead_k", metavar='INT', type=int, default=5,
                               help="Lookahead uses k steps (-1 -> no Lookahead is used)")
    args_training.add_argument("--learning_rate", metavar='FLOAT', type=float, default=1e-3, help="learning rate for the optimizer")
    args_training.add_argument("--decay_factor", metavar='FLOAT', type=float, default=0.5,
                               help="learning rate is decayed by this factor whenever the " +
                               "validation loss does not improve after decay_patience intervals")
    args_training.add_argument("--decay_patience", metavar='INT', type=int, default=10,
                               help="how many validation intervals have to be seen without improvement before the learning rate is decayed")
    args_training.add_argument("--stop_at_learning_rate", metavar='FLOAT', type=float, default=1e-1,
                               help="when the learning rate gets lower than this value, training is stopped")
    args_training.add_argument("--stop_at_learning_rate_patience", metavar='FLOAT', type=float, default=0,
                               help="when the learning rate gets lower than this value, training is stopped")
    args_training.add_argument("--epsilon", metavar='FLOAT', type=float, default=1e-8,
                               help="epsilon for the optimizer (only relevant for Adam/AMSGrad)")
    args_training.add_argument("--beta1", metavar='FLOAT', type=float, default=0.9,
                               help="beta1 for the optimizer (only relevant for Adam/AMSGrad)")
    args_training.add_argument("--beta2", metavar='FLOAT', type=float, default=0.999,
                               help="beta2 for the optimizer (only relevant for Adam/AMSGrad)")
    args_training.add_argument("--momentum", metavar='FLOAT', type=float, default=0.0,
                               help="momentum for the optimizer (only relevant for SGD)")
    args_training.add_argument("--density_grad", metavar='True|False', type=str2bool, default=False,
                               help="Calculate the density gradient")
    args_training.add_argument("--density_weight", metavar='FLOAT', type=float, default=1.0,
                               help="weight of the density in the loss function")
    args_training.add_argument("--density_grad_weight", metavar='FLOAT', type=float, default=0.0,
                               help="weight of the density gradient in the loss function")
    args_training.add_argument("--df_weight", metavar='FLOAT', type=float, default=0.0,
                               help="weight of the density fitting coeffs in the loss function")
    args_training.add_argument("--dipole_moment_weight", metavar='FLOAT', type=float, default=0.0,
                               help="weight of the dipole moment in the loss function")
    args_training.add_argument("--energy_weight", metavar='FLOAT', type=float, default=0.0,
                               help="weight of the energy in the loss function")
    args_training.add_argument("--forces_weight", metavar='FLOAT', type=float, default=0.0,
                               help="weight of the forces in the loss function")
    args_training.add_argument("--hamiltonian_matrix_weight", metavar='FLOAT', type=float, default=1.0,
                               help="weight of the full hamiltonian in the loss function")
    args_training.add_argument("--density_matrix_weight", metavar='FLOAT', type=float, default=1.0,
                               help="weight of the density matrix in the loss function")
    args_training.add_argument("--energy_min_weight", metavar='FLOAT', type=float, default=0.0,
                               help="weight of the energy minimization loss")
    args_training.add_argument("--density_loss_comp", metavar='STR', type=str, default=['mae'], nargs='+',
                               choices=['mae', 'mse', 'rmse', 'lda_mae', 'lda_rmse', 'hartree_mae', 'hartree_rmse',
                                        'coulomb', 'perc_mae', 'perc_rmse', 'mixed_dist_err',
                                        'perc_mixed_dist_err', 'kl_loss', 'dpm_loss', 'dpm_abs_loss'],
                               help="Composition of the density loss.")
    args_training.add_argument("--density_grad_loss_comp", metavar='STR', type=str, default=['norm_int'], nargs='+',
                               choices=['norm_int', 'perc_norm_int', 'kinetic_vw'],
                               help="Composition of the density loss.")
    args_training.add_argument("--dipole_moment_loss_comp", metavar='STR', type=str, default=['mae'], nargs='+',
                               choices=['mae', 'rmse'], help="composition of the dipole moment loss")
    args_training.add_argument("--df_loss_comp", metavar='STR', type=str, default=['mae'], nargs='+',
                               choices=['mae', 'rmse'], help="composition of the density fitting loss")
    args_training.add_argument("--energy_loss_comp", metavar='STR', type=str, default=['mae'], nargs='+',
                               choices=['mae', 'rmse'], help="composition of the energy loss")
    args_training.add_argument("--forces_loss_comp", metavar='STR', type=str, default=['mae'], nargs='+',
                               choices=['mae', 'rmse'], help="composition of the forces loss")
    args_training.add_argument("--density_loss_comp_weights", metavar='FLOAT', type=float, default=[1.0], nargs='+',
                               help="weights of the composition of the density loss")
    args_training.add_argument("--density_grad_loss_comp_weights", metavar='FLOAT', type=float, default=[1.0], nargs='+',
                               help="weights of the composition of the density loss")
    args_training.add_argument("--dipole_moment_loss_comp_weights", metavar='FLOAT', type=float, default=[1.0], nargs='+',
                               help="weights of the composition of the dipole moment loss")
    args_training.add_argument("--df_loss_comp_weights", metavar='FLOAT', type=float, default=[1.0], nargs='+',
                               help="weights of the composition of the density fitting loss")
    args_training.add_argument("--energy_loss_comp_weights", metavar='FLOAT', type=float, default=[1.0], nargs='+',
                               help="weights of the composition of the energy loss")
    args_training.add_argument("--forces_loss_comp_weights", metavar='FLOAT', type=float, default=[1.0], nargs='+',
                               help="weights of the composition of the forces loss")
    args_training.add_argument("--density_weight_min", metavar='FLOAT', type=float, default=0.0,
                               help="minimum weight of the density in the loss function")
    args_training.add_argument("--density_grad_weight_min", metavar='FLOAT', type=float, default=0.0,
                               help="minimum weight of the density gradient in the loss function")
    args_training.add_argument("--dipole_moment_weight_min", metavar='FLOAT', type=float, default=0.0,
                               help="minimum weight of the dipole moment in the loss function")
    args_training.add_argument("--df_weight_min", metavar='FLOAT', type=float, default=0.0,
                               help="minimum weight of the density fitting in the loss function")
    args_training.add_argument("--energy_weight_min", metavar='FLOAT', type=float, default=0.0,
                               help="minimum weight of the energy in the loss function")
    args_training.add_argument("--forces_weight_min", metavar='FLOAT', type=float, default=0.0,
                               help="minimum weight of the forces in the loss function")
    args_training.add_argument("--energy_min_weight_min", metavar='FLOAT', type=float, default=0.0,
                               help="minimum weight of the energy minimization loss")
    args_training.add_argument("--density_weight_decay", metavar='FLOAT', type=float, default=1.0,
                               help="decay of the weight of the density in the loss function")
    args_training.add_argument("--density_grad_weight_decay", metavar='FLOAT', type=float, default=1.0,
                               help="decay of the weight of the density gradient in the loss function")
    args_training.add_argument("--dipole_moment_weight_decay", metavar='FLOAT', type=float, default=1.0,
                               help="decay of the weight of the dipole moment in the loss function")
    args_training.add_argument("--df_weight_decay", metavar='FLOAT', type=float, default=1.0,
                               help="decay of the weight of the density fitting in the loss function")
    args_training.add_argument("--energy_weight_decay", metavar='FLOAT', type=float, default=1.0,
                               help="decay of the weight of the energy in the loss function")
    args_training.add_argument("--forces_weight_decay", metavar='FLOAT', type=float, default=1.0,
                               help="decay of the weight of the forces in the loss function")
    args_training.add_argument("--energy_min_weight_decay", metavar='FLOAT', type=float, default=1.0,
                               help="decay of the weight of the energy minimization loss")
    args_training.add_argument("--max_energy_error", metavar='FLOAT', type=float, default=0.1,
                               help="for better stability at beginning of training: maximum allowed MAE " +
                               "in energy (higher errors are clamped)")
    args_training.add_argument("--max_forces_error", metavar='FLOAT', type=float, default=0.1,
                               help="for better stability at beginning of training: maximum allowed MAE " +
                               "in forces (higher errors are clamped)")
    args_training.add_argument("--df_loss_weights",  metavar='True|False', type=str2bool, default=False, choices=[True, False],
                               help="Incorporate weighting based on radial coefficients for density fitting loss.")
    args_training.add_argument("--center_energy", metavar='True|False', type=str2bool, default=True,
                               choices=[True, False], help="set energy mean to zero.")
    args_training.add_argument("--atomic_energies", metavar='STR', type=str, default=None, help='File containing ' +
                               'atomic energies that will be used for normalization of total_energy')
    args_training.add_argument("--clip_norm", metavar='FLOAT', type=float, default=0.0,
                               help="gradient clip norm (only when --use_gradient_clipping is active)")
    args_training.add_argument("--use_parameter_averaging", metavar='True|False', type=str2bool, default=True,
                               choices=[True, False], help="keep exponential moving average of" +
                               " model parameters (might boost convergence speed)")
    args_training.add_argument("--ema_decay", metavar='FLOAT', type=float, default=0.999,
                               help="decay rate used for exponential moving average of parameters")
    args_training.add_argument("--ema_start_epoch", metavar='INT', type=int, default=0,
                               help="starts exponential moving average of parameters only after the specified epoch is reached")
    args_training.add_argument("--weight_decay", metavar='FLOAT', type=float, default=0.0, help="regularization term for weights")
    args_training.add_argument("--use_gpu", metavar='True|False', type=str2bool, default=True,
                               choices=[True, False], help="use GPU(s) for training (if available)")
    args_training.add_argument("--multiple_gpus", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="use multiple GPUs for training (if available)")
    args_training.add_argument("--coord_weights", metavar='True|False', type=str2bool, default=True,
                               choices=[True, False], help="weight grid coordinates based on grid density")
    args_training.add_argument("--weights_balance", metavar='FLOAT', type=float, default=1.0,
                               help="Term for balancing the coordinate weights of the density grid.")
    args_training.add_argument("--softmax_norm", metavar='True|False', type=str2bool, default=True,
                               choices=[True, False], help="Normalize the coefficients using softmax.")
    args_training.add_argument("--percentage_error", metavar='True|False', type=str2bool, default=True,
                               choices=[True, False], help="Measure error as a percentage of the density integral.")
    args_training.add_argument("--pyscf_grid", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Use pyscf for density grid generation.")
    args_training.add_argument("--cube_grid", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Use cubical densty grid for training.")
    args_training.add_argument("--cube_grid_valid", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="also use cube densty grid for validation.")
    args_training.add_argument("--cube_size", metavar='INT', type=int, default=50,
                               help="Size of the cubical grid")
    args_training.add_argument("--cube_extent", metavar='FLOAT', type=float, default=4.1483,
                               help="Extent of the cubical grid.")
    args_training.add_argument("--cube_origin", metavar='FLOAT', type=float, default=-2.0318,
                               help="Origin of the cubical grid.")
    args_training.add_argument("--spherical_grid_level", metavar='INT', type=int, default=2,
                               help="Size of the spherical grid")
    args_training.add_argument("--verbose", metavar='INT', type=int, default=0, help="Verbosity level.")
    args_training.add_argument("--timing", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Timing runtime.")
    args_training.add_argument("--memory", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Memory statistics.")
    args_training.add_argument("--energy_unit_in", metavar='STR', type=str, default='kcal/mol',
                               choices=['kcal/mol', 'hartree', 'eV', 'kelvin', 'millihartree'],
                               help="energy input unit")
    args_training.add_argument("--energy_unit_out", metavar='STR', type=str, default='kcal/mol',
                               choices=['kcal/mol', 'hartree', 'eV', 'kelvin', 'millihartree'],
                               help="energy output unit")
    args_training.add_argument("--distance_unit_in", metavar='STR', type=str, default='angstrom',
                               choices=['angstrom', 'bohr'], help="distance input unit")
    args_training.add_argument("--distance_unit_out", metavar='STR', type=str, default='angstrom',
                               choices=['angstrom', 'bohr'], help="distance output unit")
    args_training.add_argument("--output_scaling", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Scale the output forces to unit variance.")
    args_training.add_argument("--grid_scaling_factor", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Factor to control scaling of density to correct integral on grid.")
    args_training.add_argument("--grid_scaling_annealing", metavar='FLOAT', type=float, default=1.0,
                               help="Anneal the grid scaling factor over time.")
    args_training.add_argument("--grid_scaling_start", metavar='INT', type=int, default=10000,
                               help="The step at which grid scaling starts.")
    args_training.add_argument("--projected_density", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Use density fitting basis for labels.")
    args_training.add_argument("--fast_df", metavar='True|False', type=str2bool, default=True,
                               choices=[True, False], help="Do a fast/shorter density fitting training procedure.")
    args_training.add_argument("--core_density_basis", metavar='FLOAT', type=float, default=0.0,
                               help="Choose a fraction of the s orbitals to use as an additional"
                               + "basis for the core density, and perform the fitting if > 0.")
    args_training.add_argument("--density_fine_tuning", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Fine tune the density with possibly a different loss.")
    args_training.add_argument("--density_loss_comp_ft", metavar='STR', type=str, default=['mae'], nargs='+',
                               choices=['mae', 'mse', 'rmse', 'lda_mae', 'lda_rmse', 'hartree_mae', 'hartree_rmse',
                                        'coulomb', 'perc_mae', 'perc_rmse', 'mixed_dist_err',
                                        'perc_mixed_dist_err', 'kl_loss', 'dpm_loss', 'dpm_abs_loss'],
                               help="Composition of the density loss for fine tuning")
    args_training.add_argument("--fine_tuning_lr_factor", metavar='FLOAT', type=float, default=0.1,
                               help="Fine tuning learning rate reduction factor.")
    args_training.add_argument("--fine_tuning_stop_lr_factor", metavar='FLOAT', type=float, default=0.1,
                               help="Fine tuning stop at learning rate reduction factor.")
    args_training.add_argument("--density_loss_comp_ft_weights", metavar='FLOAT', type=float, default=[1.0], nargs='+',
                               help="weights of the composition of the density loss")
    args_training.add_argument('--wandb_mode', metavar='STR', type=str, default='offline',
                               choices=['online', 'offline', 'disabled'], help="Wandb mode.")
    args_training.add_argument("--remove_atom_density", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Remove the free atom density from the total density.")
    args_training.add_argument('--atom_dens_path', metavar='STR', type=str,
                               help="Path to free atom densities file.")
    args_training.add_argument('--atom_dens_type', metavar='STR', type=str,
                               default='spline', choices=['spline', 'df_coeffs', 'mo_coeffs'],
                               help="Type of functions to expand the free atom densities.")
    args_training.add_argument("--density_from_df", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Set density coeffs to DF coeffs instead of predicting them")
    args_training.add_argument("--density_from_free_atoms", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Set density coeffs to free atom coeffs instead of predicting them")
    args_training.add_argument("--compile", metavar='True|False', type=str2bool, default=False,
                               choices=[True, False], help="Compile the model for higher efficiency.")
    # arguments for simulations
    args_simulation = parser.add_argument_group("simulation hyperparameters")
    args_simulation.add_argument("--temperature", metavar='INT', type=int, default=300,
                                 help="Temperature in Kelvin for the simulation.")
    args_simulation.add_argument("--new_run", metavar='True|False', type=str2bool, default=True,
                                 choices=[True, False],
                                 help="If true start new simulation, otherwise continue previous one.")
    args_simulation.add_argument("--log_dir", metavar='STR', default='.', type=str, help="Path to simulation and logs directory.")
    args_simulation.add_argument("--log_suffix", metavar='STR', default='', type=str, help="Suffix for the log file.")
    args_simulation.add_argument("--md_steps", metavar='INT', type=int, default=100,
                                 help="Number of molecular dynamic steps.")
    args_simulation.add_argument("--langevin", metavar='True|False', type=str2bool, default=True,
                                 choices=[True, False], help="If true use Langevin dynamics, else use velocity Verlet.")
    args_simulation.add_argument("--warm_up", metavar="True|False", type=str2bool, default=True,
                                 choices=[True, False], help="If true use Langevin as warm up for 5%% of all simulation steps, and then switch to Verlet.")
    args_simulation.add_argument("--simulation_type", metavar='STR', type=str, default='md',
                                 choices=['md', 'opt'], help="type of simulation to run.")
    args_simulation.add_argument("--port_num", metavar='INT', type=int, default=50007,
                                 help="Port number for communication.")
    args_simulation.add_argument("--force_conversion", metavar='STR', default='kcal/mol/Ang', type=str, help="Force conversion unit.")
    args_simulation.add_argument("--position_conversion", metavar='STR', default='Ang', type=str, help="Position conversion unit.")
    args_simulation.add_argument("--energy_conversion", metavar='STR', default='kcal/mol', type=str, help="Energy conversion unit.")
    args_simulation.add_argument("--start_idx", metavar='INT', type=int, default=[-1], nargs='+', help="Start indices for the simulation.")

    # arguments for logging and checkpoints
    args_logging = parser.add_argument_group("logging and checkpoints")
    args_logging.add_argument("--save_dir", metavar='STR', default='.', type=str, help="Path to model and logs directory.")
    args_logging.add_argument("--write_parameter_summaries", metavar='True|False', type=str2bool,
                              default=False, choices=[True, False], help="write summaries for parameters")
    args_logging.add_argument("--validation_interval", metavar='INT', type=int, default=1,
                              help="perform model validation after every INT steps")
    args_logging.add_argument("--summary_interval", metavar='INT', type=int, default=1, help="log summaries after every INT steps")
    args_logging.add_argument("--checkpoint_interval", metavar='INT', type=int, default=1, help="write checkpoints after every INT steps")
    args_logging.add_argument("--keep_checkpoints", metavar='INT', type=int, default=0,
                              help="keep X checkpoints older than the latest checkpoint (-1 keeps all checkpoints)")

    # misc arguments
    args_misc = parser.add_argument_group("miscelleaneous")
    args_misc.add_argument("--dtype", metavar='torch.float32|torch.float64', type=str, default='torch.float32',
                           choices=['torch.float16', 'torch.float32', 'torch.float64'], help="floating point type used during training")
    args_misc.add_argument('--legacy', metavar='True|False', type=str2bool, default=False,
                           choices=[True, False], help="If true use old density network code, else use the most recent version.")
    args_misc.add_argument('--test_save', metavar='True|False', type=str2bool, default=False,
                           choices=[True, False], help="Save output from tests as a list.")
    args_misc.add_argument('--test_save_name', metavar='STR', type=str, default="test_save_results.pt",
                           help="Filename for saved test output.")
    args_misc.add_argument('--no_compare', metavar='True|False', type=str2bool, default=False,
                           choices=[True, False], help="Don't compare accuracy of test samples, just compute predictions.")

    # actually parse command line arguments
    if arg_file is None and len(sys.argv) == 1:  # no arguments were specified, print help message
        args = parser.parse_args(["--help"])
    else:
        if arg_file is not None:
            with open(arg_file, 'r') as f:
                args_str = f.read()
                args = parser.parse_args(args_str.split())
        else:
            args = parser.parse_args()
        # convert dtype argument to the proper torch type
        if args.dtype == 'torch.float32':
            args.dtype = torch.float32
        if args.dtype == 'torch.float16':
            args.dtype = torch.float16
        elif args.dtype == 'torch.float64':
            args.dtype = torch.float64

        # necessary because None is not properly by argparse (special case)
        if args.restart == 'None':
            args.restart = None
        if args.dens_dataset == 'None':
            args.dens_dataset = None
        if args.dens_dataset_test == 'None':
            args.dens_dataset_test = None
        if args.np_dataset_test == 'None':
            args.np_dataset_test = None
        if args.dens_dataset_valid == 'None':
            args.dens_dataset_valid = None
        if args.np_dataset_valid == 'None':
            args.np_dataset_valid = None
        if args.radial_coeffs_file == 'None':
            args.args.radial_coeffs_file = None
        if args.load_from == 'None':
            args.load_from = None
        if args.expansion_constraint == 'None':
            args.expansion_constraint = None
        if args.integral_constraint == 'None':
            args.integral_constraint = None
        if args.integral_constraint == 'True':
            args.integral_constraint = True
        if args.energy_model == 'None':
            args.energy_model = None
        if args.start_idx[0] == -1:
            args.start_idx = None
        if args.atom_dens_path == 'None':
            args.atom_dens_path = None

        args.energy_unit_in = args.energy_unit_in.split('/')[0]
        args.energy_unit_out = args.energy_unit_out.split('/')[0]

    return args, hyperparam_args
