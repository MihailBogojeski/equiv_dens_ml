"""Build left/full Ramachandran splits of the alanine dipeptide dataset.
stage build: split by phi, map MO chunks to global indices, write np + dens files.
stage check: PySCF recomputation of one geometry + re-read of outputs."""
import argparse, glob, os
import numpy as np
from scipy.spatial import cKDTree
from pyscf import gto

p = argparse.ArgumentParser()
p.add_argument('stage', choices=['build', 'check'])
p.add_argument('--src', default='/home/ml-dft/equiv_dens/datasets/alanine_dipeptide')
p.add_argument('--out', default='/home/ml-dft/equiv_dens/datasets/alanine_dipeptide/rama')
p.add_argument('--seed', type=int, default=42)
args = p.parse_args()
SETS = ['train_left', 'train_full', 'valid_left', 'valid_full', 'test']
HA2KCAL = 627.5096080305927
BOHR = 0.529177210903


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    x = (v * w).sum(-1)
    y = (np.cross(b1, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def dists(pos):
    iu = np.triu_indices(pos.shape[1], 1)
    d = np.linalg.norm(pos[:, :, None] - pos[:, None], axis=-1)
    return d[:, iu[0], iu[1]]


src = np.load(f'{args.src}/alanine_dipeptide_dft.npy', allow_pickle=True).item()
pos = src['positions']

if args.stage == 'build':
    P = pos
    phi = dihedral(P[:, 4], P[:, 6], P[:, 8], P[:, 14])
    psi = dihedral(P[:, 6], P[:, 8], P[:, 14], P[:, 16])
    L, R = np.where(phi < 0)[0], np.where(phi > 0)[0]
    rng = np.random.default_rng(args.seed)
    pL, pR = rng.permutation(L), rng.permutation(R)
    test = np.concatenate([pL[:500], pR[:500]])
    valid_left = pL[500:1300]
    train_left = pL[1300:3300]
    train_full = np.concatenate([train_left, pR[500:1000]])
    used = np.concatenate([test, valid_left, train_full])
    rest = np.setdiff1d(np.arange(len(P)), used)
    valid_full = rng.choice(rest, 800, replace=False)
    idx = dict(train_left=train_left, train_full=train_full, valid_left=valid_left,
               valid_full=valid_full, test=test)
    os.makedirs(args.out, exist_ok=True)
    np.savez(f'{args.out}/split_indices.npz', phi=phi, psi=psi, **idx)
    print('n L/R', len(L), len(R), {k: len(v) for k, v in idx.items()})
    print('train_left phi max', phi[train_left].max(), 'train_full R count', (phi[train_full] > 0).sum())
    assert len(np.unique(np.concatenate([test, valid_left, train_full, valid_full]))) == 1000 + 800 + 2500 + 800

    # map MO chunks to global indices via interatomic distances
    tree = cKDTree(dists(pos))
    need = np.unique(np.concatenate(list(idx.values())))
    need_set = set(need.tolist())
    found = {}  # global idx -> (mol_dict, calc_dict)
    for c in range(11):
        f = f'{args.src}/alanine_dipeptide_dft_mo_calc_{c}.npy'
        calc = np.load(f, allow_pickle=True)
        cpos = np.array([[a[1] for a in m['atom']] for m, _ in calc])
        d, gi = tree.query(dists(cpos))
        shift = cpos - pos[gi]
        shift = shift - shift.mean(1, keepdims=True)  # residual after removing translation
        print(f'chunk {c}: n={len(calc)} global {gi.min()}..{gi.max()} contiguous={bool((np.diff(gi)==1).all())} '
              f'unique={len(np.unique(gi))} max dist-feature err={d.max():.2e} max residual after translation={np.abs(shift).max():.2e}', flush=True)
        np.save(f'{args.out}/chunk_{c}_global_idx.npy', gi)
        for j, g in enumerate(gi):
            if g in need_set and g not in found:
                found[g] = calc[j]
        del calc
    print('found', len(found), 'of', len(need))

    for name, ix in idx.items():
        n = len(ix)
        np_d = dict(positions=pos[ix], atom_numbers=np.tile(src['atom_numbers'], (n, 1)),
                    atom_types=[src['atom_numbers'].astype(np.uint8) for _ in range(n)],
                    energy=src['energy'][ix], forces=src['forces'][ix])
        np.save(f'{args.out}/alanine_rama_{name}.npy', np_d, allow_pickle=True)
        dens = np.empty((n, 2), dtype=object)
        for k, g in enumerate(ix):
            m, c = found[g]
            m = dict(m)
            m['atom'] = [(np.uint8(gto.mole.charge(str(s))), pos[g, a].copy()) for a, (s, _) in enumerate(m['atom'])]
            dens[k, 0] = m
            dens[k, 1] = dict(mo_coeff=c['mo_coeff'], mo_occ=c['mo_occ'])
        np.save(f'{args.out}/alanine_rama_{name}_mo_calc.npy', dens, allow_pickle=True)
        print('wrote', name, n, flush=True)
        del dens

else:
    from pyscf import dft
    rng = np.random.default_rng(0)
    for name in SETS:
        np_d = np.load(f'{args.out}/alanine_rama_{name}.npy', allow_pickle=True).item()
        dens = np.load(f'{args.out}/alanine_rama_{name}_mo_calc.npy', allow_pickle=True)
        gidx = np.load(f'{args.out}/split_indices.npz')[name]
        assert len(dens) == len(np_d['positions']) == len(gidx)
        for k in rng.choice(len(gidx), 5, replace=False):
            g = gidx[k]
            m = dens[k, 0]
            cp = np.array([a[1] for a in m['atom']])
            e1 = np.abs(cp - np_d['positions'][k]).max()
            e2 = np.abs(np_d['positions'][k] - pos[g]).max()
            e3 = abs(np_d['energy'][k] - src['energy'][g]).max() + np.abs(np_d['forces'][k] - src['forces'][g]).max()
            print(f'{name} k={k} g={g} mol-vs-np pos {e1:.1e} np-vs-src pos {e2:.1e} E/F diff {e3:.1e}')
        print(name, 'n', len(gidx), 'keys', list(np_d), 'mo_coeff', dens[0, 1]['mo_coeff'].shape)
        if name == 'test':
            k = 0
            g = gidx[k]
            mol = gto.Mole(**dens[k, 0]); mol.build()
            mf = dft.RKS(mol); mf.xc = 'PBE'; mf.kernel()
            gr = mf.nuc_grad_method().kernel()
            E = mf.e_tot * HA2KCAL
            F = -gr * HA2KCAL / BOHR
            print('converged', mf.converged, 'E recomputed kcal/mol', E, 'stored', src['energy'][g, 0], 'diff', E - src['energy'][g, 0])
            print('forces max abs diff', np.abs(F - src['forces'][g]).max(), 'max abs force', np.abs(src['forces'][g]).max())
            mo, occ = dens[k, 1]['mo_coeff'], dens[k, 1]['mo_occ']
            dm0 = (mo * occ) @ mo.T
            print('density matrix max abs diff', np.abs(dm0 - mf.make_rdm1()).max(), 'max abs dm', np.abs(dm0).max(), 'nelec stored', np.trace(dm0 @ mol.intor('int1e_ovlp')))
