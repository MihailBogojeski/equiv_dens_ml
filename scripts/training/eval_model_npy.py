#!/usr/bin/env python3
import os
import torch
from datetime import datetime
import time
import wandb
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.utils.misc import generate_id
from equiv_dens.training.errors import ErrorDict
from equiv_dens.training.trainer import Trainer
from equiv_dens.data.density_dataset import AtomsDensityData
from equiv_dens.data.hamiltonian_dataset import seeded_random_split
from equiv_dens.training.lookahead import Lookahead
from equiv_dens.training.model_loader import load_model
from equiv_dens.data.custom_samplers import set_up_data_loader
from equiv_dens.utils.grids import cubical_grid, cubical_sampling,\
    spherical_grid, spherical_radial_sampling
from equiv_dens.utils import orbitals
from equiv_dens.utils import base as utils
from equiv_dens.training import utils as train_utils
import argparse
from pathlib import Path

import numpy as np
from functools import partial
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('args_file', type=str)
parser.add_argument('target', type=str)
parser.add_argument('--dpm_intor', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--density_path', type=str, default=None,
                    help='Path to DFT density data (*_pyscf.npy). When set, uses dataset.get_properties (dipole_accuracy flow) instead of model_input_from_atoms.')

main_args = parser.parse_args()

# Preprocess args file to remove =None values which cause parsing errors
import tempfile
with open(main_args.args_file, 'r') as f:
    args_content = f.read()

args_lines = []
for line in args_content.strip().split('\n'):
    line = line.strip()
    if line and not line.startswith('#') and '=None' not in line:
        # Fix common typos
        if '--energy_unit_out=kcal' in line and '--energy_unit_out=kcal/mol' not in line:
            line = line.replace('--energy_unit_out=kcal', '--energy_unit_out=kcal/mol')
        if '--energy_unit_in=kcal' in line and '--energy_unit_in=kcal/mol' not in line:
            line = line.replace('--energy_unit_in=kcal', '--energy_unit_in=kcal/mol')
        args_lines.append(line)

# Create temporary preprocessed args file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
    tmp_file.write('\n'.join(args_lines))
    tmp_args_file = tmp_file.name

args, hyperparam_args = parse_command_line_arguments(arg_file=tmp_args_file)

# Clean up temp file
import os as _os
_os.unlink(tmp_args_file)

# Repo root: scripts/training -> scripts -> equiv_dens_ml
base_dir = str(Path(__file__).resolve().parent.parent.parent)

# Fix dataset paths
if hasattr(args, 'np_dataset') and args.np_dataset and '/home/ml-dft/equiv_dens/' in args.np_dataset:
    args.np_dataset = args.np_dataset.replace('/home/ml-dft/equiv_dens/', base_dir + '/')
if hasattr(args, 'dens_dataset') and args.dens_dataset and '/home/ml-dft/equiv_dens/' in str(args.dens_dataset):
    args.dens_dataset = str(args.dens_dataset).replace('/home/ml-dft/equiv_dens/', base_dir + '/')

# Fix orbital/coefficient paths
for attr in ['orbitals_file', 'radial_coeffs_file', 'L0_coeffs_file', 'atom_dens_path', 'pseudo_pot_path']:
    if hasattr(args, attr) and getattr(args, attr) and '/home/ml-dft/equiv_dens/' in str(getattr(args, attr)):
        setattr(args, attr, str(getattr(args, attr)).replace('/home/ml-dft/equiv_dens/', base_dir + '/'))

# Fall back to the _pyscf prior only when the one the model was trained with is
# genuinely absent. This used to substitute unconditionally, which quietly
# evaluated the model against a different delta-learning reference than it was
# fitted to -- the substitution existed because the original would not unpickle
# under current SciPy, and that is now handled in equiv_dens.utils.scipy_compat.
if hasattr(args, 'atom_dens_path') and args.atom_dens_path:
    if 'free_atom_densities_augccpvdz_augccpvqzjkfit.npy' in args.atom_dens_path \
            and not Path(args.atom_dens_path).exists():
        args.atom_dens_path = args.atom_dens_path.replace(
            'free_atom_densities_augccpvdz_augccpvqzjkfit.npy',
            'free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy'
        )
        print('original SAD prior missing; falling back to', args.atom_dens_path)

# dpm-intor with remove_atom_density requires MO or DF coefficients (not spline).
# Override atom_dens_type and atom_dens_path when the model was trained with spline.
if main_args.dpm_intor and getattr(args, 'remove_atom_density', False):
    if getattr(args, 'atom_dens_type', 'spline') == 'spline':
        args.atom_dens_type = 'mo_coeffs'
        pyscf_minimized = Path(base_dir) / 'datasets' / 'free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy'
        if pyscf_minimized.exists():
            args.atom_dens_path = str(pyscf_minimized)

# Fix checkpoint/log paths
if hasattr(args, 'restart') and args.restart and '/home/ml-dft/equiv_dens/' in args.restart:
    args.restart = args.restart.replace('/home/ml-dft/equiv_dens/', base_dir + '/')
if hasattr(args, 'log_dir') and args.log_dir and '/home/ml-dft/equiv_dens/' in args.log_dir:
    args.log_dir = args.log_dir.replace('/home/ml-dft/equiv_dens/', base_dir + '/')

# Set log_dir to local results directory to avoid permission issues
args.log_dir = os.path.join(base_dir, 'paper', 'results', 'eval_logs')
os.makedirs(args.log_dir, exist_ok=True)

# The args file usually sits inside the run directory it describes, so its
# parent is the checkpoint directory. That is not true of a config under
# config/training/, and overriding unconditionally sent the loader looking for
# config/training/checkpoints/latest_checkpoint.pth. Only take the parent when
# it actually holds checkpoints; otherwise honour the config's own --restart.
args_file_dir = Path(main_args.args_file).resolve().parent
if (args_file_dir / 'checkpoints').is_dir():
    args.restart = str(args_file_dir)
elif not (args.restart and (Path(args.restart) / 'checkpoints').is_dir()):
    raise SystemExit(
        f"no checkpoints/ beside {main_args.args_file} and --restart="
        f"{args.restart!r} does not hold one either"
    )
args.np_dataset = str(Path(main_args.target).resolve())

# Use density_path when provided or auto-detect: target.npy -> target_pyscf.npy in same dir
use_density_path = False
if main_args.density_path and Path(main_args.density_path).exists():
    args.dens_dataset = str(Path(main_args.density_path).resolve())
    use_density_path = True
    print('Using density_path (get_properties flow):', args.dens_dataset)
elif Path(main_args.target).is_file():
    candidate = Path(main_args.target).with_stem(Path(main_args.target).stem + '_pyscf')
    if candidate.exists():
        args.dens_dataset = str(candidate.resolve())
        use_density_path = True
        print('Auto-detected density_path (get_properties flow):', args.dens_dataset)

print('args use gpu', args.use_gpu)
print('Using model checkpoint:', args.restart)

args.timing = True
args.energy_weight = 0
args.forces_weight = 0
args.density_weight = 0
args.dipole_moment_weight = 1
args.dpm_intor = main_args.dpm_intor
if args.dpm_intor:
    args.integral_constraint = 'coeffs_in_coeffs_net'
# args.integral_constraint = None 
args, hyperparam_args, test_vars = train_utils.init_training_vars(args, hyperparam_args)
checkpoint = test_vars['checkpoint']
args_dict = vars(args)

print('model code:', test_vars['model_code'])

# determine whether GPU is used for training
print('args use gpu', args.use_gpu)

# load dataset(s)
print("loading density from" + str(args.dens_dataset) + "...")
print("loading atoms from" + args.np_dataset + "...")

grid_vars = train_utils.init_grid_vars(args, test=True)
print('grid vars', grid_vars)

required_properties = ['density', 'dipole_moment'] if use_density_path else ['dipole_moment']
if not use_density_path:
    args.dens_dataset = None
# args.np_dataset = "datasets/ethanethiol_md_traj_every1000_dft_augccpvdz.npy"
# args.dens_dataset = "datasets/ethanethiol_md_traj_every1000_dft_augccpvdz_df_augccpvqzjkfit.npy"
# args.np_dataset = '/home/ml-dft/equiv_dens/datasets/8mer_all-every20_pyscf_d4_augccpvdz_npy.npy'
# args.dens_dataset = '/home/ml-dft/equiv_dens/datasets/8mer_all-every20_pyscf_d4_augccpvdz.npy'
print('pyscf grid', args.pyscf_grid)
dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=10000000000000,
                           required_properties=required_properties,
                           center_positions=True,
                           radial_coeffs_file=args.radial_coeffs_file,
                           L0_coeffs_file=args.L0_coeffs_file,
                           dtype=args.dtype,
                           grid_fn=grid_vars['grid_fn'],
                           pyscf_grid=args.pyscf_grid,
                           pyscf_rotate=grid_vars['rotate'],
                           sampling_fn=grid_vars['sampling_fn'],
                           grid_extent=grid_vars['grid_extent'],
                           grid_origin=grid_vars['grid_origin'],
                           verbose=args.verbose,
                           timing=args.timing,
                           cutoff=args.cutoff,
                           df_loss_weights=args.df_loss_weights,
                           atom_dens_path=args.atom_dens_path,
                           atom_dens_type=args.atom_dens_type,
                           projected_density=False,
                           density_grad=False,
                           dpm_intor=args.dpm_intor,
                           )

model = load_model(args, dataset, train=False)

model.eval()
if args.use_gpu:
    model.cuda()
print('use gpu', args.use_gpu)
model.to(args.dtype)
for param in model.parameters():
    param.requires_grad = False

is_dir = os.path.isdir(main_args.target)

if is_dir:
    files = os.listdir(main_args.target)
    print(files)
    for i in range(len(files)):
        files[i] = os.path.join(main_args.target, files[i])
    print(files)
else:
    files = [main_args.target]

for file in files:
    dpm_errors = []
    dir = '/'.join(file.split('/')[:-1])
    fname = file.split('/')[-1]
    if args.dpm_intor:
        suffix = '_dpm_intor.npy'
    else:
        suffix = '_dpm.npy'

    ref_file_name = fname[:-4] + '_dm.txt'
    fname = fname[:-4] + suffix
    print('fname', fname)
    print('filetype', file[-3:])
    ref_file_path = os.path.join(dir, ref_file_name)
    ref_file_lines = []
    if os.path.exists(ref_file_path):
        with open(ref_file_path, 'r') as f:
            ref_file_lines = f.readlines()
        print(f"Loaded reference dipoles from {ref_file_path}")
    else:
        print(f"No reference file found at {ref_file_path}, skipping error calculation")
    out_exists = os.path.exists(os.path.join('results', fname))
    if 'npy' != file[-3:] or out_exists:
        print('skipping file')
        continue
    data_npy = np.load(file, allow_pickle=True).item()
    if data_npy['atom_numbers'].ndim == 1:
        data_npy['atom_numbers'] = np.tile(data_npy['atom_numbers'][None, :],
                                           (data_npy['positions'].shape[0], 1))
    data_pos = 0
    data_npy['dipole_moment'] = None
    total_frames = data_npy['positions'].shape[0]
    print(f'Processing {total_frames} frames with batch size {main_args.batch_size}...')

    # Create progress bar
    pbar = tqdm(total=total_frames, desc=f"Computing dipoles for {fname}", unit="frames")
    
    count = 0
    while data_pos < total_frames:
        count += 1
        # if count > 10:
        #     break
        if data_pos + main_args.batch_size > data_npy['positions'].shape[0]:
            max_pos = data_npy['positions'].shape[0]
        else:
            max_pos = data_pos + main_args.batch_size

        batch_npy = {'positions': data_npy['positions'][data_pos:max_pos],
                     'atom_numbers': data_npy['atom_numbers'][data_pos:max_pos]}
        start = time.time()
        if use_density_path and not is_dir:
            # dipole_accuracy flow: dataset loads atom coeffs from density_path
            batch_idx = np.arange(data_pos, max_pos)
            data = dataset.get_properties(batch_idx)
        else:
            data = orbitals.model_input_from_atoms(batch_npy,
                                                   density_expansion=True,  # Always need coords for model
                                                   pyscf_grid=True,
                                                   grid_spec=dataset.grid_spec,
                                                   atom_dens_type=args.atom_dens_type,
                                                   cutoff=args.cutoff,
                                                   grid_sampling_fn=dataset.sampling_fn,
                                                   dtype=args.dtype,
                                                   free_atom_densities=dataset.atom_dens,
                                                   )
        for key in data.keys():
            if isinstance(data[key], torch.Tensor) and args.use_gpu:
                data[key] = data[key].cuda()
        if args.timing:
            print('data from npy time', time.time() - start)
        res = model(data)
        # print('res density integral', torch.sum(res['density'] * res['coord_weights'], dim=1))
        # print('center positions', torch.mean(res['positions'],))
        # print('num electrons', orbitals.get_n_electrons(res['atom_numbers']))
        # print('dipole_moment', res['dipole_moment'])
        # print('dipole magnitude', torch.norm(res['dipole_moment'], dim=-1))
        np_dpm = utils.internal_to_debye(res['dipole_moment'].numpy(force=True))
        
        # Update progress bar with current dipole magnitude
        avg_dipole_mag = np.mean(np.linalg.norm(np_dpm, axis=-1))
        pbar.set_postfix({'avg_dipole': f'{avg_dipole_mag:.3f} D', 
                         'batch_time': f'{time.time()-start:.2f}s'})
        
        print('dipole_moment converted', np_dpm)
        print('data pos', data_pos, 'max pos', max_pos)
        if ref_file_lines:  # Only compare if reference file exists
            try:
                dipole_ref = [ref_file_lines[i].split(' ') for i in range(data_pos, max_pos)]
                dipole_ref = np.array(dipole_ref).astype(float)
                print('dipole ref', dipole_ref)
                print('dipole error', np.linalg.norm(dipole_ref - np_dpm, axis=-1))
                dpm_errors.append(np.linalg.norm(dipole_ref - np_dpm, axis=-1))
            except Exception as e:
                print(f"Error comparing dipoles: {e}")

        if data_npy['dipole_moment'] is None:
            data_npy['dipole_moment'] = np_dpm
        else:
            data_npy['dipole_moment'] = np.concatenate([data_npy['dipole_moment'], np_dpm], axis=0)

        # samp = dataset.get_properties(np.arange(data_pos, max_pos))
        # for key in samp.keys():
        #     if isinstance(samp[key], torch.Tensor) and args.use_gpu:
        #         samp[key] = samp[key].cuda()
        # print('samp dipole_moment', samp['dipole_moment'])
        # dpm_err = utils.internal_to_debye(torch.norm(samp['dipole_moment'] - res['dipole_moment'], dim=-1)).numpy(force=True)
        # print('dpm_errors', dpm_err)
        # if dpm_errors is None:
        #     dpm_errors = dpm_err
        # else:
        #     dpm_errors = np.concatenate([dpm_errors, dpm_err], axis=0)

        batch_processed = max_pos - data_pos
        pbar.update(batch_processed)
        data_pos += main_args.batch_size
        allocated_memory = torch.cuda.memory_allocated()
        print(f"Memory allocated: {allocated_memory / (1024**2):.2f} MB")
        res = None
    
    pbar.close()
    np.save(os.path.join('results', fname), data_npy, allow_pickle=True)
    print('average dpm error', np.mean(np.concatenate(dpm_errors, axis=-1)))

    # np.save(os.path.join('results', 'dpm_errors_' + fname), dpm_errors, allow_pickle=True)
    # print('mean dpm error', np.mean(dpm_errors))
