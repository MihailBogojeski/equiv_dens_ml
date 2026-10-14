"""Timing of coefficient extraction and density expansion (+ density_grad), and of a full training step.

python benchmarks/bench_density.py {cpu,cuda} [h2o|big|all]
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import torch
from equiv_dens.training.parse_command_line_arguments import parse_command_line_arguments
from equiv_dens.training import utils as train_utils
from equiv_dens.training.model_loader import load_model
from equiv_dens.data.custom_samplers import set_up_data_loader
from equiv_dens.utils.orbitals import model_input_from_atoms
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_density_equiv import CONFIG, BASE

dev = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else 'all'
if dev == 'cuda':
    torch.cuda.set_per_process_memory_fraction(0.3)


def sync():
    if dev == 'cuda':
        torch.cuda.synchronize()


def timeit(fn, n=12):
    fn()
    sync()
    ts = []
    if dev == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for _ in range(n):
        t = time.time()
        fn()
        sync()
        ts.append(time.time() - t)
    mem = torch.cuda.max_memory_allocated() / 1024**2 if dev == 'cuda' else 0
    return '%8.1f ms (min %8.1f) %7.0f MB' % (1000 * np.median(ts), 1000 * np.min(ts), mem)


def report(name, fn):
    try:
        print('%-34s %s' % (name, timeit(fn)), flush=True)
    except torch.OutOfMemoryError:
        print('%-34s OOM' % name, flush=True)
        torch.cuda.empty_cache()


def bench_atoms(model, atoms, label, repr_cpu=False):
    """atoms: model input already on device (training-mode batch or inference input)."""
    repr_model, coeffs_net = model.density_repr_model[0], model.density_repr_model[1]
    expansion = model.property_models['density']
    with torch.no_grad():
        if repr_cpu:  # the representation of the big system does not fit the shared GPU
            import copy
            atoms = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in atoms.items()}
            atoms_r = copy.deepcopy(repr_model).cpu()({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in atoms.items()})
            atoms_r = {k: ([x.to(dev) for x in v] if isinstance(v, list) else v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in atoms_r.items()}
        else:
            atoms_r = repr_model({k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in atoms.items()})
    shallow = lambda: {k: v for k, v in atoms_r.items()}
    sph0 = atoms_r['sph_repr']

    def fresh():
        a = shallow()
        a['sph_repr'] = [x for x in sph0] if isinstance(sph0, list) else sph0
        return a
    atoms_c = coeffs_net(fresh())

    def exp_in():
        a = {k: v for k, v in atoms_c.items()}
        return a

    def extract_fb():
        a = coeffs_net(fresh())
        a['df_coeffs'].pow(2).mean().backward()

    def make(gradflag, bwd):
        def f():
            a = expansion(exp_in(), density_grad=gradflag)
            if bwd:
                loss = a['density'].pow(2).mean()
                if gradflag:
                    loss = loss + a['density_grad'].pow(2).mean()
                loss.backward(retain_graph=True)
        return f
    print('----', label, '  (median over 5 runs, peak GPU mem)')
    report('extract coeffs fwd', lambda: coeffs_net(fresh()))
    report('extract coeffs fwd+bwd', extract_fb)
    model.eval()
    with torch.no_grad():
        report('expansion fwd (eval, no_grad)', make(False, False))
        report('expansion fwd+grad (eval, no_grad)', make(True, False))
    model.train()
    report('expansion fwd (train)', make(False, False))
    report('expansion fwd+bwd', make(False, True))
    report('expansion fwd+density_grad', make(True, False))
    report('expansion fwd+density_grad+bwd', make(True, True))


def build(extra=()):
    sys.argv = ['x', CONFIG] + BASE + ['--dtype=torch.float32', '--density_weight=1.0'] + list(extra)
    args, _ = parse_command_line_arguments()
    args.use_gpu = dev == 'cuda'
    grid_vars = train_utils.init_grid_vars(args)
    required = train_utils.get_required_properties_from_args(args)
    torch.manual_seed(0)
    dataset, train_ds, _, _, _, _ = train_utils.prepare_datasets(args, required, grid_vars, None)
    model = load_model(args, dataset, train=True).to(args.dtype).to(dev)
    return args, dataset, train_ds, model


if which in ('h2o', 'all'):
    args, dataset, train_ds, model = build(['--density_grad=True', '--density_grad_weight=0.1'])
    loader = set_up_data_loader(train_ds, args.train_batch_size, args.electron_num_batching,
                                args.batch_efficiency, False, False)
    data = next(iter(loader))
    data = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
    data = model.transform_input(data)
    bench_atoms(model, data, 'h2o_small batch 5 x 5000 points')
    error_dict = train_utils.init_error_dict(args)

    def step():
        model.train()
        pred = model(data)
        loss = error_dict.compute(pred, data)['loss']
        loss.backward()
        model.zero_grad()
    report('FULL train step (dens+grad+en+F)', step)
    if os.environ.get('PROFILE'):
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step()
            sync()
        print(prof.key_averages().table(sort_by='cuda_time_total' if dev == 'cuda' else 'cpu_time_total', row_limit=15))
    args.density_grad = False

if which in ('big', 'all'):
    args, dataset, train_ds, model = build(['--density_grad=True'])
    h2o = np.load('/workspace/datasets/h2o_small_train_augccpvdz.npy', allow_pickle=True).item()
    pos = np.asarray(h2o['positions'][0], dtype=np.float64)
    zs = np.asarray(h2o['atom_numbers']).reshape(-1)[-3:] if np.ndim(h2o['atom_numbers']) else None
    zs = np.asarray(h2o['atom_numbers'])
    zs = zs[0] if zs.ndim > 1 else zs
    shifts = np.array([[3.5 * i, 3.5 * j, 3.5 * k] for i in range(4) for j in range(4) for k in range(2)])
    big_pos = (pos[None] + shifts[:, None]).reshape(-1, 3)
    big_z = np.tile(zs, len(shifts))
    G = int(os.environ.get('BIG_GRID', 30000))
    rng = np.random.RandomState(0)
    lo, hi = big_pos.min(0) - 2, big_pos.max(0) + 2
    coords = torch.tensor(rng.uniform(lo, hi, size=(1, G, 3)))
    cp = {'coords': coords, 'coord_weights': torch.ones(1, G, dtype=torch.float64) * 1e-3}
    atoms = {'positions': big_pos[None], 'atom_numbers': big_z[None]}
    inputs = model_input_from_atoms(atoms, density_expansion=True, pyscf_grid=False, coord_params=cp,
                                    dtype=torch.float32, cutoff=args.cutoff)
    bench_atoms(model, inputs, '32 waters, %d random grid points' % G, repr_cpu=True)
