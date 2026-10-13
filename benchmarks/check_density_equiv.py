"""Equivalence check of the density path (coeff extraction, df_coeffs, density, density_grad, param grads).

python benchmarks/check_density_equiv.py save <dir> [variants...]     # run with the current code, store batch/weights/outputs
python benchmarks/check_density_equiv.py compare <dir> [variants...]  # run again and compare against the stored reference
Run from the repo root. Datasets are read from /workspace/datasets.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training import utils as train_utils
from equiv_dens.training.model_loader import load_model
from equiv_dens.data.custom_samplers import set_up_data_loader

CONFIG = '@config/training/h2o_small_all_001.txt'
DATA = [
    ('np_dataset', 'h2o_small_train_augccpvdz.npy'),
    ('dens_dataset', 'h2o_small_train_augccpvdz_df_augccpvqzjkfit.npy'),
    ('np_dataset_valid', 'h2o_small_valid_augccpvdz.npy'),
    ('dens_dataset_valid', 'h2o_small_valid_augccpvdz_df_augccpvqzjkfit.npy'),
    ('np_dataset_test', 'h2o_small_test_augccpvdz.npy'),
    ('dens_dataset_test', 'h2o_small_test_augccpvdz_df_augccpvqzjkfit.npy'),
    ('orbitals_file', 'augccpvqzjkfit_orbital_basis_df.npy'),
    ('radial_coeffs_file', 'augccpvqzjkfit_radial_coeffs_libcint_df.npy'),
    ('atom_dens_path', 'free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy'),
]
BASE = ['--%s=/workspace/datasets/%s' % kv for kv in DATA] + [
    '--use_gpu=False', '--dtype=torch.float64', '--density_grad=True', '--density_grad_weight=0.1',
    '--wandb_mode=disabled']
VARIANTS = {
    'default': [],
    'no_pred_radial': ['--pred_radial_coeffs=False'],
    'positive': ['--positive_coeffs=True'],
    'int_constraint': ['--integral_constraint=coeffs_in_coeffs_net'],
    'int_constraint_scale': ['--integral_constraint=coeffs_in_coeffs_net', '--integral_scale=True'],
    'from_df': ['--density_from_df=True'],
    'from_free_atoms': ['--density_from_free_atoms=True'],
}


def run(name, mode, d):
    sys.argv = ['x', CONFIG] + BASE + VARIANTS[name]
    args, _ = parse_command_line_arguments()
    grid_vars = train_utils.init_grid_vars(args)
    required = train_utils.get_required_properties_from_args(args)
    torch.manual_seed(0)
    np.random.seed(0)
    dataset, train_ds, _, _, _, _ = train_utils.prepare_datasets(args, required, grid_vars, None)
    torch.manual_seed(0)
    model = load_model(args, dataset, train=True).to(torch.float64)
    bpath = os.path.join(d, name + '_batch.pt')
    wpath = os.path.join(d, name + '_weights.pt')
    if mode == 'save':
        loader = set_up_data_loader(train_ds, args.train_batch_size, args.electron_num_batching,
                                    args.batch_efficiency, False, False)
        data = next(iter(loader))
        torch.save(data, bpath)
        torch.save(model.state_dict(), wpath)
    else:
        data = torch.load(bpath)
        model.load_state_dict(torch.load(wpath))
    model.train()
    data = model.transform_input(data)
    pred = model(data)
    out = {}
    for k in ['density', 'density_grad', 'df_coeffs', 'energy', 'forces']:
        if k in pred:
            out[k] = pred[k].detach()
    for k in ['spherical_coeffs', 'radial_width', 'radial_scale']:
        if k in pred:
            for i, dct in enumerate(pred[k]):
                for key, v in dct.items():
                    out['%s_%d_%d_%d' % (k, i, key[0], key[1])] = v.detach()
    loss = pred['density'].pow(2).mean()
    if 'density_grad' in pred:
        loss = loss + pred['density_grad'].pow(2).mean()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True, retain_graph=True)
    for (n, _), g in zip(params, grads):
        if g is not None:
            out['grad:' + n] = g
    # gradients of the density-only loss and of the density_grad-only loss
    for tag, l in (('gd:', pred['density'].pow(2).mean()), ('gg:', pred['density_grad'].pow(2).mean())):
        gs = torch.autograd.grad(l, [p for _, p in params], allow_unused=True, retain_graph=True)
        for (n, _), g in zip(params, gs):
            if g is not None:
                out[tag + n] = g
    return out


if __name__ == '__main__':
    mode, d = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or list(VARIANTS)
    os.makedirs(d, exist_ok=True)
    for name in names:
        out = run(name, mode, d)
        opath = os.path.join(d, name + '_out.pt')
        if mode == 'save':
            torch.save(out, opath)
            print(name, 'saved', len(out), 'tensors')
        else:
            ref = torch.load(opath)
            worst = 0
            for k in ref:
                if k not in out:
                    print(name, 'MISSING', k)
                    continue
                a, b = out[k], ref[k]
                err = ((a - b).abs().max() / (b.abs().max() + 1e-300)).item()
                worst = max(worst, err)
                if k in ('density', 'density_grad', 'df_coeffs', 'energy', 'forces'):
                    print(name, k, 'rel err %.3e' % err)
                if err > 1e-10:
                    print(name, 'MISMATCH', k, 'rel err %.3e' % err)
            print(name, 'worst rel err %.3e' % worst)
