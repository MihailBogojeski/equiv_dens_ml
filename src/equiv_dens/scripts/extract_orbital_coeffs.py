import numpy as np
from pyscf import gto
import sys
from ase.data import chemical_symbols

basis = sys.argv[1]
libcint = sys.argv[2]

if 'jkfit' in basis:
    df_str = '_df'
else:
    df_str = ''

if libcint:
    libcint_str = '_libcint'
else:
    libcint_str = ''

mol = gto.M(atom='O  0  0  1; H  0,  0, 2; N 0,  0, 3; C 0, 0, 4; S 0, 0, 5; F 0 0 6; Cl 0 0 7', basis=basis)
ao_coeffs = {}
ao_basis = {}
repeat = {}
for i in range(mol._bas.shape[0]):
    a_ind = mol.bas_atom(i)
    a_num = mol._atm[a_ind, 0]
    symbol = chemical_symbols[a_num]
    if symbol not in ao_basis:
        ao_basis[symbol] = []
        ao_coeffs[symbol] = []
        repeat[symbol] = a_ind
    if repeat[symbol] != a_ind:
        continue
    L = mol.bas_angular(i)
    nprim = mol.bas_nprim(i)
    nctr = mol.bas_nctr(i)
    exp = mol.bas_exp(i)
    if libcint:
        ctr = mol._libcint_ctr_coeff(i)
    else:
        ctr = mol.bas_ctr_coeff(i)
    for j in range(nctr):
        ao_basis[symbol].append((a_num, nprim, L))
        ao_coeffs[symbol].append((np.array(exp), np.array(ctr[:, j])))

ao_coeffs = np.array(ao_coeffs).item()
print('ao basis', ao_basis)
print('ao coeffs', ao_coeffs)

print('ao_coeffs O', ao_coeffs['O'])
print('ao basis O', ao_basis['O'])
print('mol basis O', mol._basis['O'])

np.save('datasets/' + basis + '_orbital_basis' + libcint_str + df_str + '.npy', ao_basis, allow_pickle=True)
np.save('datasets/' + basis + '_radial_coeffs' + libcint_str + df_str + '.npy', ao_coeffs, allow_pickle=True)

ao_basis_old = np.load('datasets/' + basis + '_orbital_basis' + libcint_str + df_str + '.npy', allow_pickle=True).item()
print('ao_basis', ao_basis_old)
ao_coeffs_old = np.load('datasets/' + basis + '_radial_coeffs' + libcint_str + df_str + '.npy', allow_pickle=True).item()
print('ao_coeffs', ao_coeffs_old)

print(ao_basis == ao_basis_old)

for key in ao_coeffs:
    print('atom :', key)
    for i in range(len(ao_coeffs[key])):
        print('orb', i)
        #print('new', ao_coeffs[key][i])
        #print('old', ao_coeffs_old[key][i])
        print(np.isclose(ao_coeffs[key][i][0], ao_coeffs_old[key][i][0]))
        print(np.isclose(ao_coeffs[key][i][1], ao_coeffs_old[key][i][1]))
