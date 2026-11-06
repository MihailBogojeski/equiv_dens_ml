# %%
import numpy as np
from pyscf import gto, dft
from pyscf.scf import hf
hf.MUTE_CHKFILE = True

# %%
# load data

atom_pos = [0, 0, 0]

# test different basis sets
basis_sets = ['631gs', '631gss', 'def2svp', 'ccpvdz', 'augccpvdz']
atoms = [1, 6, 7, 8, 16]
for basis in basis_sets:
    print('basis', basis)
    savefile = 'datasets/atomization_energy_' + basis + '.npy'
    atom_en = {0: 0}
    for a in atoms:
        print('atom', a)
        atom_dict = list(zip([a], [atom_pos]))
        if (a % 2 == 1):
            atom = gto.M(atom=atom_dict, spin=None, basis=basis)
        else:
            atom = gto.M(atom=atom_dict, basis=basis, spin=None)
        mf = dft.RKS(atom)
        mf.max_cycle = 1000
        mf.chkfile = False
        mf.xc = 'pbe'
        mf.kernel()
        atom_en[a] = mf.e_tot

    print('atomization energies', basis, ':', atom_en)
    np.save(savefile, atom_en)
