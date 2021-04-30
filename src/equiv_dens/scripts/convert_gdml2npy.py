import numpy as np
import argparse
import equiv_dens.utils.base as utils
from pyscf import gto
import os
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gdml_data_prefix', type=str, help='Path to data in db format')
    parser.add_argument('npy_data_name', type=str, help='Path and name of converted file.')
    parser.add_argument('--data_dir', type=str, default='/home/ml-dft/equiv_dens/datasets', help='Path to data directory')
    args = parser.parse_args()

    np_data = {}
    pyscf_dicts = []
    files = os.listdir(args.data_dir)
    for f in files:
        if re.search(args.gdml_data_prefix, f) is None:
            continue
        gdml_data = dict(np.load(os.path.join(args.data_dir, f), allow_pickle=True))
        if 'positions' not in np_data.keys():
            np_data['positions'] = gdml_data['R']
            np_data['energy'] = gdml_data['E'].reshape(-1, 1)
            np_data['forces'] = gdml_data['F']
            np_data['atom_numbers'] = gdml_data['z']
            symbols = utils.numbers_to_symbols(np_data['atom_numbers'])
            np_data['atom_types'] = np.array([symb for symb in symbols])
        else:
            np_data['positions'] = np.concatenate([np_data['positions'], gdml_data['R']], axis=0)
            np_data['energy'] = np.concatenate([np_data['energy'], gdml_data['E'].reshape(-1, 1)], axis=0)
            np_data['forces'] = np.concatenate([np_data['forces'], gdml_data['F']], axis=0)

        if 'mo_coeff' in gdml_data.keys():
            print('basis', gdml_data['basis'].item())
            mol = gto.M(atom=[(np_data['atom_types'][i], np_data['positions'][0, i, :])
                              for i in range(len(gdml_data['z']))], basis=gdml_data['basis'].item())
            for i in range(gdml_data['mo_coeff'].shape[0]):
                # print('i', i)
                # print(lslakjdflasdf)
                calc_pyscf = []
                mol_dict = mol.pack()
                pyscf_atoms = [(np_data['atom_types'][j], np_data['positions'][i, j, :])
                               for j in range(len(gdml_data['z']))]
                mol_dict['atom'] = pyscf_atoms
                calc_dict = {'mo_coeff': gdml_data['mo_coeff'][i], 'mo_occ': gdml_data['mo_occ'][i]}
                # print('mo_coeff shape', calc['mo_coeff'].shape)
                # print('mo_occ shape', calc['mo_occ'].shape)
                pyscf_dicts.append((mol_dict, calc_dict))

    np.save(os.path.join(args.data_dir, args.npy_data_name + '.npy'), np_data, allow_pickle=True)
    np.save(os.path.join(args.data_dir, args.npy_data_name + '_mo_calc.npy'), pyscf_dicts, allow_pickle=True)
