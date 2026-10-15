#!/usr/bin/env python3
"""
Time the model components on real batches.

Usage (from anywhere):
  python benchmarks/bench_model.py --dataset h2o_small --device cuda --batch_size 5
  python benchmarks/bench_model.py --dataset water8 --device cuda --batch_size 2 --tag baseline_
  python benchmarks/bench_model.py --dataset h2o_small --device cpu --density_grad=True --density_grad_weight=1.0
Unknown flags (--key=value) are passed through as training-arg overrides on top of --config.
--dataset: h2o_small or water{n} (from /workspace/datasets/water_clusters, see make_water_clusters.py).
Writes benchmarks/results/<tag><dataset>_<device>_bs<bs>.json and prints a table.
Training: components and train_step (forward + loss + backward). Inference (model.eval(), no parameter grads):
infer_forces (energy + forces, gradient w.r.t. positions only), infer_coeffs (representation + density coefficient
head under inference_mode) and infer_density (coefficients + density on the grid). With --compile=True the training
step and infer_forces use the compiled force region, infer_coeffs / infer_density a separately compiled
representation + coefficient head.
"""
import argparse
import json
import os
import statistics
import time

import torch

import common as C
from equiv_dens.data.custom_samplers import set_up_data_loader

p = argparse.ArgumentParser()
p.add_argument('--config', nargs='+', default=[C.DEFAULT_CONFIG])
p.add_argument('--dataset', default='h2o_small')
p.add_argument('--device', default='cuda')
p.add_argument('--batch_size', type=int, default=2)
p.add_argument('--reps', type=int, default=5)
p.add_argument('--warmup', type=int, default=2)
p.add_argument('--tag', default='')
p.add_argument('--mem_fraction', type=float, default=0.3)
p.add_argument('--out_dir', default=os.path.join(C.ROOT, 'benchmarks/results'))
cli, overrides = p.parse_known_args()

cuda = cli.device == 'cuda'
if cuda:
    torch.cuda.set_per_process_memory_fraction(cli.mem_fraction)
overrides = C.dataset_overrides(cli.dataset, cli.batch_size) + [f'--use_gpu={cuda}'] + overrides
args, dataset, train_ds, model, error_dict = C.build(cli.config, overrides)
device = torch.device('cuda' if cuda else 'cpu')
model.train()


def sync():
    if cuda:
        torch.cuda.synchronize()


def timeit(fn):
    """Median wall time in ms and peak GPU memory in MB over cli.reps calls (after warmup)."""
    t = time.perf_counter()
    fn()
    sync()
    first = (time.perf_counter() - t) * 1e3  # includes the compile
    for _ in range(cli.warmup - 1):
        fn()
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(cli.reps):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t) * 1e3)
    return {'ms': statistics.median(ts), 'first_ms': first, 'peak_mb': torch.cuda.max_memory_allocated() / 2**20 if cuda else None}


# ---- (f) data loading: dataset __getitem__ + collate (get_properties), grid / density sampling ----
loader = set_up_data_loader(train_ds, cli.batch_size, False, 0.7, cuda, True)
it = iter(loader)


def load_batch():
    global it
    try:
        return next(it)
    except StopIteration:
        it = iter(loader)
        return next(it)


results = {'data_loading': timeit(load_batch)}

# ---- fixed batch on device, transformed once (transform_input rescales in place) ----
data = model.transform_input(C.to_device(load_batch(), device))
n_atoms = int(data['batch_atom_numbers'].gt(0).sum())
n_grid = data['coords'].shape[1] if 'coords' in data else 0


def fresh():
    """Same copy that DFTNetwork.forward makes; positions require grad when forces are computed."""
    atoms = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
    if model.calculate_forces:
        atoms['positions'].requires_grad_(True)
    return atoms


repr_model, dens_head = model.density_repr_model[0], model.density_repr_model[1]
atoms_a = repr_model(fresh())
atoms_b = dens_head(dict(atoms_a))

# (a) representation, (b) density coefficient head
results['repr_forward'] = timeit(lambda: repr_model(fresh()))


results['density_head'] = timeit(lambda: dens_head(dict(atoms_a)))

# (c) density expansion on the grid
expansion = model.property_models['density']
results['expansion'] = timeit(lambda: expansion(dict(atoms_b), density_grad=False))
results['expansion_grad'] = timeit(lambda: expansion(dict(atoms_b), density_grad=True))

# (d) energy + forces via autograd (forces computed inside the energy model)
if 'energy' in model.property_models:
    energy_model = model.property_models['energy']
    results['energy_forces'] = timeit(lambda: energy_model(dict(atoms_b)))

# (e) full training step: forward + loss, backward
step_t = {'fwd_loss': [], 'bwd': []}


def train_step():
    model.zero_grad()
    sync()
    t0 = time.perf_counter()
    pred = model(data)
    errors = error_dict.compute(pred, data)
    sync()
    t1 = time.perf_counter()
    errors['loss'].backward()
    sync()
    t2 = time.perf_counter()
    step_t['fwd_loss'].append((t1 - t0) * 1e3)
    step_t['bwd'].append((t2 - t1) * 1e3)


results['train_step'] = timeit(train_step)
reps = cli.reps
results['train_step']['fwd_loss_ms'] = statistics.median(step_t['fwd_loss'][-reps:])
results['train_step']['bwd_ms'] = statistics.median(step_t['bwd'][-reps:])

# ---- inference (model.eval()) ----
model.eval()
for q in model.parameters():
    q.requires_grad_(False)
compiled = getattr(model, 'compiled', False)

# (g) energy + forces: gradient w.r.t. the positions only (no create_graph, no parameter grads, no density on the grid)
if 'energy' in model.property_models:
    no_force_props, model.no_force_props = model.no_force_props, []
    results['infer_forces'] = timeit(lambda: model(data))
    model.no_force_props = no_force_props

# (h) density only under inference_mode: representation + coefficient head, then also the density on the grid
labels = ('coords', 'coord_weights', 'density', 'density_grad', 'atom_density', 'atom_density_grad', 'energy', 'forces',
          'batch_forces', 'dipole_moment')
inputs = {k: v for k, v in data.items() if isinstance(v, torch.Tensor) and k not in labels}


def coeffs_fn(atoms):
    return dens_head(repr_model(atoms))


if compiled:
    inputs['coeff_layout'] = dens_head.batch_layout(data)[1]
    coeffs_fn = torch.compile(coeffs_fn, dynamic=True)


def infer_coeffs():
    with torch.inference_mode():
        return coeffs_fn(dict(inputs))


def infer_density():
    with torch.inference_mode():
        atoms = dict(data)
        atoms.update(coeffs_fn(dict(inputs)))
        return expansion(atoms, density_grad=False)['density']


results['infer_coeffs'] = timeit(infer_coeffs)
results['infer_density'] = timeit(infer_density)

meta = {'dataset': cli.dataset, 'device': cli.device, 'batch_size': cli.batch_size, 'reps': cli.reps,
        'config': cli.config, 'overrides': overrides, 'n_atoms_batch': n_atoms, 'n_grid_points': n_grid,
        'n_params': sum(p.numel() for p in model.parameters()), 'dtype': str(args.dtype),
        'gpu': torch.cuda.get_device_name() if cuda else None}
print(f"\n{cli.dataset} {cli.device} bs={cli.batch_size} atoms/batch={n_atoms} grid={n_grid}")
print(f"{'component':<16}{'ms':>12}{'peak MB':>12}{'first ms':>12}")
for k, v in results.items():
    print(f"{k:<16}{v['ms']:>12.1f}{('-' if v['peak_mb'] is None else format(v['peak_mb'], '.0f')):>12}{v['first_ms']:>12.0f}")
print(f"  train_step: fwd+loss {results['train_step']['fwd_loss_ms']:.1f} ms, bwd {results['train_step']['bwd_ms']:.1f} ms")

os.makedirs(cli.out_dir, exist_ok=True)
path = os.path.join(cli.out_dir, f'{cli.tag}{cli.dataset}_{cli.device}_bs{cli.batch_size}.json')
json.dump({'meta': meta, 'results': results}, open(path, 'w'), indent=1)
print('wrote', path)
