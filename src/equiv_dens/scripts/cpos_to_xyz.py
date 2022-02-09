import numpy as np
from equiv_dens.utils import base as utils
import ase.io
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]
mol_list = []
atom_types = []
atom_grouping = []
last_atom_value = 0
grouping_pos = -1
value_type_dict = {}
with open(input_file, 'r') as f:
    line = f.readline().strip()
    total_atoms = np.int(line.split()[10])
    print('total atoms', total_atoms)
    atom_count = 0
    skip_count = 0
    atom_list = []
    line_count = 0
    while line:
        # print('line count', line_count)
        coords = line.split()
        # print('coords', coords)
        if line_count < 5:
            pass
        elif len(coords) == 1:
            atom_grouping.append(coords[0][0])
        elif len(coords) != 3:
            atom_value = coords[0]
            if atom_value != last_atom_value:
                grouping_pos += 1
            if atom_value not in value_type_dict:
                value_type_dict[atom_value] = atom_grouping[grouping_pos]
            # print('atom_value', atom_value)
            # print('last_atom_value', last_atom_value)
            # print('atom_type', value_type_dict[atom_value])
            atom_types.append(value_type_dict[atom_value])
            last_atom_value = atom_value
        elif atom_count < total_atoms and skip_count == 0:
            num_coords = np.array(coords).astype(np.double)
            # print('atom count', atom_count)
            # print('num coords', num_coords)
            atom_list.append(num_coords)
            atom_count += 1
            # print('adding to mol')
        elif atom_count >= total_atoms:
            atom_count = 0
            mol_list.append(atom_list)
            atom_list = []
            # print('skipping')
            # print('skip count', skip_count)
            skip_count = 1
        elif skip_count < 2:
            # print('skipping')
            # print('skip count', skip_count)
            skip_count += 1
        else:
            # print('skipping')
            # print('skip count', skip_count)
            skip_count = 0
        line_count += 1
        line = f.readline().strip()


pos = np.array(mol_list)
pos = utils.bohr_to_angstrom(pos)
print('pos.shape', pos.shape)
print('atom types', atom_types)
mol = utils.npy_to_ase(pos, atom_types)
ase.io.write(output_file, mol)
# mol = utils.npy_to_ase(pos)

