import ase
import numpy as np
import time
from pyscf.scf import hf
from pyscf import gto, dft
import equiv_dens.utils.base as utils
import sys
import os

import ase.io
import dftd4.pyscf as d4disp

hf.MUTE_CHKFILE = True

# %load_ext autoreload
# %autoreload 2

# %%
xyz_file = sys.argv[1]
save_file = sys.argv[2]
xyz_data = list(ase.io.iread(xyz_file))

# %%
idx = np.arange(0, len(xyz_data))
# idx = np.arange(0, 2)
if os.path.exists(save_file):
    results = np.load(save_file, allow_pickle=True).tolist()
else:
    results = []
# for mol in xyz_data:
for i in range(len(results), len(idx)):
    print('i', i)
    mol = xyz_data[i]
    npy_data = utils.ase_to_npy2([mol])
    pos = npy_data['positions'][0]
    atom_nums = npy_data['atom_numbers'][0]
    print('npy data atoms', npy_data['atom_numbers'][0])

    start = time.time()
    atom = []
    for j in range(len(atom_nums)):
        if atom_nums[j] < 1:
            continue
        atom.append((atom_nums[j], pos[j, :]))
    mol = gto.M(atom=atom, basis='augccpvdz')

    mf = dft.RKS(mol)
    mf.chkfile = False
    mf.xc = 'pbe'
    mf.max_cycle = 1000
    d4mf = d4disp.energy(mf).run()
    grad = d4mf.nuc_grad_method()
    gradients = grad.kernel()
    # print('combined gradient', grad.kernel())
    res = []
    res.append(mol.pack())
    calc_dict = {}
    print('mo occ', d4mf.mo_occ)
    calc_dict['mo_coeff'] = d4mf.mo_coeff
    calc_dict['mo_occ'] = d4mf.mo_occ
    calc_dict['energy'] = d4mf.e_tot
    calc_dict['forces'] = -gradients/ase.units.Bohr
    res.append(calc_dict)
    results.append(res)
    print('elapsed time', time.time() - start)
    np.save(save_file, results, allow_pickle=True)

np.save(save_file, results, allow_pickle=True)

res_npy = utils.calc_dict_to_npy(results, compress_atoms=False, convert_forces=False)
np.save(save_file[:-4] + '_npy.npy', res_npy, allow_pickle=True)

if len(sys.argv) > 3:
    idx = np.concatenate([np.arange(0, 10), np.arange(1000, 1010), np.arange(2000, 2001)])
    # idx = np.concatenate([np.arange(0, 5), np.arange(1000, 1010), np.arange(2000, 2001)])
    # idx = np.concatenate([np.arange(0, 5)])
    npy_file = sys.argv[3]
    thio_poly = np.load(npy_file, allow_pickle=True).item()
    for res_i, i in enumerate(idx):
        nonzero = thio_poly['atom_numbers'][i] > 0
        # print('energy old', utils.hartree_to_kcal(thio_poly['energy'][i]))
        # print('forces old', utils.hartree_to_kcal(thio_poly['forces'][i, nonzero]))
        # print('energy new', utils.hartree_to_kcal(results[res_i][1]['energy']))
        # print('forces new', utils.hartree_to_kcal(results[res_i][1]['forces']))
        print('energy diff', utils.hartree_to_kcal(results[res_i][1]['energy'] - thio_poly['energy'][i]))
        print('forces diff', utils.hartree_to_kcal(results[res_i][1]['forces'] - thio_poly['forces'][i, nonzero]))
        # print('mo coeffs diff', results[0][1]['mo_coeff'][res_i] - thio_poly[i][1]['mo_coeff'][0])
