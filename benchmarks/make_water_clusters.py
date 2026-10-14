#!/usr/bin/env python3
"""
Generate (H2O)_n water-cluster datasets for benchmarking.

Rigid waters (geometry of the first h2o_small frame) are placed with random orientations at
roughly liquid density (30 A^3 / molecule), rejecting O-O < 2.6 A.
Writes, per size n, into /workspace/datasets/water_clusters/:
  water{n}_augccpvdz.npy                          (np_dataset format, same as h2o_small)
  water{n}_augccpvdz_df_augccpvqzjkfit.npy        (dens_dataset format)
Sizes <= --max_real get real PBE/aug-cc-pVDZ + D4 labels (gpu4pyscf, DF fit in aug-cc-pVQZ-jkfit);
larger sizes get correctly shaped placeholder labels (timing only).
Usage: python benchmarks/make_water_clusters.py --sizes 1 2 4 8 16 32 64 --max_real 8
"""
import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np
ctypes.CDLL('/opt/conda/envs/equiv_dens_ml/lib/python3.10/site-packages/nvidia/cu13/lib/libcusolver.so.12',
            mode=ctypes.RTLD_GLOBAL)
import ase.units
import cupy
from scipy.spatial.transform import Rotation
from pyscf import gto, dft, df
import scipy.linalg
from dftd4.interface import DispersionModel, DampingParam

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
import equiv_dens.utils.base as utils

OUT = '/workspace/datasets/water_clusters'
BASIS, AUXBASIS = 'augccpvdz', 'augccpvqzjkfit'
BOHR = ase.units.Bohr


def make_cluster(n, water, rng):
    """water: (3, 3) O,H,H positions centred on O. Returns (3n, 3) positions in A."""
    box = (30.0 * n) ** (1 / 3)
    mols = []
    while len(mols) < n:
        w = Rotation.random(random_state=rng).apply(water) + rng.uniform(0, box, 3)
        if all(np.linalg.norm(w[0] - m[0]) > 2.6 and
               np.linalg.norm(w[:, None] - m[None], axis=-1).min() > 1.3 for m in mols):
            mols.append(w)
    return np.concatenate(mols)


def df_coeffs(mol, dm):
    """Density-fitting coefficients, blocked over aux shells to keep memory low."""
    auxmol = df.addons.make_auxmol(mol, AUXBASIS)
    j2c = auxmol.intor('int2c2e')
    j3 = np.zeros(auxmol.nao)
    bounds = auxmol.ao_loc_nr()
    step, s = 20, 0
    while s < auxmol.nbas:
        e = s
        while e < auxmol.nbas and bounds[e] - bounds[s] < 200:
            e += 1
        blk = df.incore.aux_e2(mol, auxmol, intor='int3c2e', shls_slice=(0, mol.nbas, 0, mol.nbas, s, e))
        j3[bounds[s]:bounds[e]] = np.einsum('ijP,ij->P', blk.reshape(mol.nao, mol.nao, -1), dm)
        s = e
    return scipy.linalg.solve(j2c, j3)


def compute_frame(positions, numbers):
    """PBE/aug-cc-pVDZ + D4 on GPU. Returns (mol.pack(), calc_dict) like generate_polythiophene_dataset."""
    mol = gto.M(atom=[(int(z), p) for z, p in zip(numbers, positions)], basis=BASIS)
    mf = dft.RKS(mol)
    mf.chkfile = None
    mf.xc = 'pbe'
    mf.max_cycle = 1000
    mf = mf.to_gpu()
    mf.kernel()
    grad = mf.nuc_grad_method().kernel()
    d4 = DispersionModel(numbers.astype(np.int32), positions / ase.units.Bohr)
    r = d4.get_dispersion(DampingParam(method='pbe'), grad=True)
    e = float(mf.e_tot) + float(r['energy'])
    grad = np.asarray(grad) + r['gradient']
    mo_coeff, mo_occ = cupy.asnumpy(mf.mo_coeff), cupy.asnumpy(mf.mo_occ)
    dm = (mo_coeff * mo_occ) @ mo_coeff.T
    calc = {'mo_coeff': mo_coeff, 'mo_occ': mo_occ, 'energy': e, 'forces': -grad / BOHR,
            'df_coeff': df_coeffs(mol, dm), 'auxbasis': AUXBASIS}
    return mol.pack(), calc


def placeholder_frame(positions, numbers, rng):
    mol = gto.M(atom=[(int(z), p) for z, p in zip(numbers, positions)], basis=BASIS)
    auxmol = df.addons.make_auxmol(mol, AUXBASIS)
    nocc = mol.nelectron // 2
    calc = {'mo_coeff': np.eye(mol.nao), 'mo_occ': np.r_[np.full(nocc, 2.0), np.zeros(mol.nao - nocc)],
            'energy': -76.3 * len(numbers) / 3, 'forces': rng.normal(0, 0.05, (len(numbers), 3)),
            'df_coeff': rng.normal(0, 0.1, auxmol.nao), 'auxbasis': AUXBASIS}
    return mol.pack(), calc


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument('--n_conf', type=int, default=4)
    p.add_argument('--max_real', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    cupy.get_default_memory_pool().set_limit(size=int(2.5 * 1024 ** 3))
    os.makedirs(OUT, exist_ok=True)

    ref = np.load('/workspace/datasets/h2o_small_train_augccpvdz.npy', allow_pickle=True).item()
    pos0 = ref['positions'][0]  # H, H, O order
    water = np.stack([pos0[2], pos0[0], pos0[1]])
    water = water - water[0]
    numbers_w = np.array([8, 1, 1])

    status_path = os.path.join(OUT, 'label_status.json')
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}
    for n in args.sizes:
        rng = np.random.default_rng(args.seed + n)
        numbers = np.tile(numbers_w, n)
        real = n <= args.max_real
        results = []
        t0 = time.time()
        for c in range(args.n_conf):
            pos = make_cluster(n, water, rng)
            results.append(compute_frame(pos, numbers) if real else placeholder_frame(pos, numbers, rng))
            print(f'n={n} conf {c} done ({time.time() - t0:.0f}s)', flush=True)
            cupy.get_default_memory_pool().free_all_blocks()
        npy = utils.calc_dict_to_npy(results, convert_forces=False, compress_atoms=True)
        np.save(f'{OUT}/water{n}_augccpvdz.npy', npy, allow_pickle=True)
        np.save(f'{OUT}/water{n}_augccpvdz_df_augccpvqzjkfit.npy', results, allow_pickle=True)
        status[str(n)] = 'real' if real else 'placeholder'
        json.dump(status, open(status_path, 'w'), indent=1)
        print(f'n={n} written ({status[str(n)]}), {time.time() - t0:.0f}s', flush=True)
