import ase.io.gromacs
import ase.io
import io
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('read_file', type=str, help='.gro file to read molecules from')
parser.add_argument('write_file', type=str, help='.xyz file to write molecules to')
parser.add_argument('--write_file_simple', type=str, default=None, help='.xyz file to write simple version of the molecules to')

args = parser.parse_args()

mol_lines = []
mols = []
mols_simple = []
with open(args.read_file, 'r') as f:
    for line in f.readlines():
        if 'diala' in line and len(mol_lines) > 0:
            mol_f = io.StringIO()
            mol_f.writelines(mol_lines)
            mol_f.seek(0)
            # print('pointer', mol_f.tell())
            # print('mol f', mol_f.getvalue())
            # print('pointer', mol_f.tell())
            # print('mol f realines', mol_f.readlines())
            # print('pointer', mol_f.tell())
            mol = ase.io.gromacs.read_gromacs(mol_f)
            mols.append(mol)
            mols_simple.append(ase.Atoms(symbols=mol.get_chemical_symbols(), positions=mol.get_positions()))
            mol_lines = []
        mol_lines.append(line)

print('mol distances', mols[0].get_distances(0, range(len(mols[0].get_chemical_symbols()))))
print('total mol num', len(mols))
ase.io.write(args.write_file, mols)
if args.write_file_simple is not None:
    print('writing simple xyz files')
    ase.io.write(args.write_file_simple, mols_simple)
