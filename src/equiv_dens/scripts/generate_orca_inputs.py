#!/usr/bin/env python
import argparse
import os
import logging
import numpy as np

from shutil import rmtree
from tqdm import tqdm
from ase.data import chemical_symbols
from ase import Atoms

import schnetpack as spk

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def generate_inputs(database, workdir, orca_header, basename='input', random_rot=None):
    header = open(orca_header, 'r').read()
    xyx_format = '{:2s}' + 3 * ' {:15.8f}' + '\n'

    for idx in tqdm(range(len(database))):
        entry = database[idx]

        atom_types = entry['_atomic_numbers'].numpy()
        positions = entry['_positions'].numpy()

        if random_rot is not None:
            # Genrate random vector on unit sphere
            vector = np.random.normal(size=3)
            vector /= np.linalg.norm(vector)

            # Get angle
            degrees = np.random.uniform(-1, 1) * random_rot

            # Rotate molecule
            atom = Atoms(atom_types, positions)
            atom.rotate(degrees, vector)

            # Update geometry
            positions = atom.positions

        filename = os.path.join(workdir, '{:s}_{:06d}.oinp'.format(basename, idx + 1))

        with open(filename, 'w') as infile:
            infile.write(header)
            for atom in range(len(atom_types)):
                infile.write(xyx_format.format(chemical_symbols[atom_types[atom]], *positions[atom]))
            infile.write('*')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dbpath', type=str, help='ASE database with relevant molecules.')
    parser.add_argument('orca_header', type=str, help='Orca input specifications.')
    parser.add_argument('workdir', type=str, help='Working directory.')
    parser.add_argument('--nselect', type=int, default=None, help='Number of molecules drawn from database. By default, all molecules are used.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--split', type=str, default=None, help='Path to split file.')
    parser.add_argument('--basename', type=str, default='input', help='Basename of input file.')
    parser.add_argument('--random_rot', type=float, default=None, help='Rotate geometries by N dedrees around random vector.')
    args = parser.parse_args()

    if not os.path.isdir(args.workdir):
        os.makedirs(args.workdir)
    else:
        logging.info('Directory {:s} already exists.'.format(args.workdir))
        rmtree(args.workdir)
        os.makedirs(args.workdir)

    spk.utils.set_random_seed(args.seed)

    database = spk.data.AtomsData(args.dbpath)

    if args.split is None:
        split_file = os.path.join(args.workdir, 'split.npz')
    else:
        split_file = args.split

    # If no selection is given, use full dataset
    if args.nselect is None:
        args.nselect = len(database)
        selected = database
    else:
        diff = len(database) - args.nselect
        if diff < 0:
            raise ValueError('Invalid number of samples for database of length {:d}'.format(len(database)))
        # This automatically shuffles the points in the dataset
        selected, remainder, _ = database.create_splits(args.nselect, diff, split_file=split_file)

    logging.info('Selected {:d} molecules...'.format(len(selected)))

    generate_inputs(selected, args.workdir, args.orca_header, basename=args.basename, random_rot=args.random_rot)
