#!/usr/bin/env python3
"""
Compiled vs eager training steps (energy + forces + density loss) on batches that mix different molecules, elements
and sizes: water, water clusters, ethanol, ethanethiol, resorcinol, random element mixes (H C N O S), a single atom
and batch size 1. The model basis covers all elements of the basis files of the config. Reports per batch the
max deviation of the outputs and parameter gradients (relative to max |eager|) and the number of compiled graphs.

  python benchmarks/compile_mixed.py [--device cuda] [--density_grad=True --density_grad_weight=1.0] [--key=value ...]
Unknown flags are training-arg overrides on top of the h2o_small config.
"""
import argparse
import time

import numpy as np
import torch
import torch._dynamo
from ase.data import atomic_numbers

import common as C
from equiv_dens.training import utils as train_utils
from equiv_dens.training.model_loader import load_model
from equiv_dens.utils.orbitals import model_input_from_atoms

p = argparse.ArgumentParser()
p.add_argument('--device', default='cuda')
p.add_argument('--grid', type=int, default=400)
p.add_argument('--seed', type=int, default=0)
cli, overrides = p.parse_known_args()
dev = torch.device(cli.device)
D = '/workspace/datasets/'

args = C.parse_args([C.DEFAULT_CONFIG], [f'--use_gpu={cli.device == "cuda"}', '--num_train=10', '--num_valid=0',
                                         '--num_test=0', '--np_dataset_valid=None', '--dens_dataset_valid=None',
                                         '--np_dataset_test=None', '--dens_dataset_test=None'] + overrides)
dataset, train_ds, _, _, _, _ = train_utils.prepare_datasets(args, train_utils.get_required_properties_from_args(args),
                                                             train_utils.init_grid_vars(args), None)
# extend the basis of the model to all elements of the basis files
basis = np.load(args.orbitals_file, allow_pickle=True).item()
radial = np.load(args.radial_coeffs_file, allow_pickle=True).item()
els = sorted(set(atomic_numbers[k] for k in basis) & set(atomic_numbers[k] for k in radial))
dataset.orbital_basis_num = {atomic_numbers[k]: v for k, v in basis.items() if atomic_numbers[k] in els}
dataset.radial_coeffs = {atomic_numbers[k]: v for k, v in radial.items() if atomic_numbers[k] in els}
model = load_model(args, dataset, train=True).to(dev).to(args.dtype)
error_dict = train_utils.init_error_dict(args)
model.train()
torch._dynamo.config.suppress_errors = False  # fail loudly instead of falling back to eager
print('basis elements', els)

# ---- molecules: (Z, R [n, 3] Angstrom) ----
rng = np.random.default_rng(cli.seed)


def load(path, i):
    d = np.load(path, allow_pickle=True).item()
    z = np.asarray(d['atom_numbers'])
    return (z[i] if z.ndim > 1 else z), np.asarray(d['positions'][i])


def random_mol(zs):
    """Random geometry with interatomic distances >= 1 A."""
    R = []
    while len(R) < len(zs):
        x = rng.uniform(-1.6, 1.6, 3) * len(zs) ** (1 / 3)
        if all(np.linalg.norm(x - r) > 1.0 for r in R):
            R.append(x)
    return np.array(zs), np.array(R)


eth = np.load(D + 'ethanol.npz')
mols = {
    'h2o': lambda: load(D + 'h2o_small_train_augccpvdz.npy', rng.integers(100)),
    'w2': lambda: load(D + 'water_clusters/water2_augccpvdz.npy', 0),
    'w4': lambda: load(D + 'water_clusters/water4_augccpvdz.npy', rng.integers(4)),
    'w8': lambda: load(D + 'water_clusters/water8_augccpvdz.npy', 0),
    'ethanol': lambda: (eth['z'], eth['R'][rng.integers(len(eth['R']))]),
    'ethanethiol': lambda: load(D + 'ethanethiol_augccpvdz_train.npy', rng.integers(1000)),
    'resorcinol': lambda: load(D + 'resorcinol_augccpvdz_train.npy', rng.integers(100)),
    'rand': lambda n=None: random_mol(rng.choice(els, size=n or rng.integers(2, 12))),
}
compositions = [
    ['h2o'] * 5,
    ['h2o', 'ethanol', 'w2'],
    ['ethanethiol', 'h2o'],
    ['resorcinol'],
    ['ethanol', 'ethanethiol', 'resorcinol', 'w4'],
    ['rand', 'rand', 'rand'],
    ['w8', 'h2o', 'rand'],
    ['h2o'],
    ['rand'],
    ['ethanethiol'] * 3,
    ['rand', 'ethanol'],
    ['w4', 'w2', 'h2o', 'h2o'],
    ['single_N'],
    ['single_S', 'h2o'],
    ['rand', 'rand', 'rand', 'rand', 'rand', 'rand'],
    ['resorcinol', 'ethanethiol'],
    ['NH3_CS2', 'h2o'],
    ['h2o'] * 5,
]


def make_batch(names):
    ms = []
    for n in names:
        if n.startswith('single_'):
            ms.append((np.array([atomic_numbers[n[7:]]]), np.zeros((1, 3))))
        elif n == 'NH3_CS2':
            ms.append(random_mol([7, 1, 1, 1, 6, 16, 16]))
        else:
            ms.append(mols[n]())
    A = max(len(z) for z, _ in ms)
    Z = np.zeros((len(ms), A), dtype=np.int64)
    R = np.zeros((len(ms), A, 3))
    for i, (z, r) in enumerate(ms):
        Z[i, :len(z)], R[i, :len(z)] = z, r
    b = model_input_from_atoms({'atom_numbers': Z, 'positions': R}, cutoff=args.cutoff, dtype=args.dtype)
    B, N = b['batch_atom_numbers'].shape[0], b['positions'].shape[1]
    g = torch.Generator().manual_seed(len(names))
    # synthetic labels: grid points around the atoms, random density / energy / forces
    centers = b['batch_positions'][:, torch.randint(0, b['batch_positions'].shape[1], (cli.grid,), generator=g)]
    b['coords'] = centers + torch.randn(B, cli.grid, 3, generator=g, dtype=args.dtype)
    b['coord_weights'] = torch.rand(B, cli.grid, generator=g, dtype=args.dtype)
    b['density'] = torch.rand(B, cli.grid, generator=g, dtype=args.dtype)
    b['density_grad'] = torch.randn(B, cli.grid, 3, generator=g, dtype=args.dtype)
    b['energy'] = torch.randn(B, 1, generator=g, dtype=args.dtype)
    b['forces'] = torch.randn(1, N, 3, generator=g, dtype=args.dtype)
    return model.transform_input(C.to_device(b, dev))


batches = [make_batch(c) for c in compositions]
KEYS = ['energy', 'forces', 'df_coeffs', 'density', 'density_grad']


def sync():
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def step(data):
    model.zero_grad()
    pred = model(dict(data))
    loss = error_dict.compute(pred, data)['loss']
    loss.backward()
    sync()
    out = {k: pred[k].detach().clone() for k in KEYS if k in pred}
    out['loss'] = loss.detach()
    out['grads'] = {n: q.grad.clone() if q.grad is not None else torch.zeros_like(q) for n, q in model.named_parameters()}
    return out


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


eager = [step(b) for b in batches]
model.enable_compile()
counters = torch._dynamo.utils.counters
worst = {}
graphs = []
for i, (names, b) in enumerate(zip(compositions, batches)):
    t = time.perf_counter()
    out = step(b)
    dt = time.perf_counter() - t
    dev_ = {k: rel(out[k], eager[i][k]) for k in out if k != 'grads'}
    g_dev = {n: ((g - eager[i]['grads'][n]).abs().max() / eager[i]['grads'][n].abs().max().clamp_min(1e-10)).item()
             for n, g in out['grads'].items()}
    dev_['grads'] = max(g_dev.values())
    for k, v in dev_.items():
        worst[k] = max(worst.get(k, 0), v)
    graphs.append(counters['stats']['unique_graphs'])
    z = b['batch_atom_numbers']
    print(f'{i:2d} B={z.shape[0]:2d} cols={z.shape[1]:3d} atoms={int((z > 0).sum()):4d} '
          f'elements={sorted(set(z.unique().tolist()) - {0})} {"+".join(names):<40} {dt * 1e3:9.1f} ms '
          f'graphs={graphs[-1]} ' + ' '.join(f'{k}={v:.1e}' for k, v in dev_.items()), flush=True)
print('\nworst rel deviation compiled vs eager: ' + ' '.join(f'{k}={v:.1e}' for k, v in worst.items()))
print({k: dict(v) for k, v in counters.items()})
print(f'compiled graphs after the first batch: {graphs[0]}, after all {len(batches)} batches: {graphs[-1]} '
      f'(recompiles after the first: {graphs[-1] - graphs[0]}), graph breaks: {sum(counters["graph_break"].values())}')
