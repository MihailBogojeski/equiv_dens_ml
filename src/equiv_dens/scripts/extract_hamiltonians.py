#!/usr/bin/env python
import argparse
import os

import numpy as np
import hamiltonian_parsing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('database', type=str, help='Path to ASE database.')
    parser.add_argument('outputs', type=str, help='Path to fhi-AIMS calculation directories or ORCA output files.')
    parser.add_argument('--basisdef', type=str, default=None, help='Path to ASE database.')
    parser.add_argument('--orbital_energies', default=None, help='Path to single atom orbital energies')
    parser.add_argument('--noout', action='store_true',
                        help='No output files available.')
    parser.add_argument('--check_convergence', action='store_true', help='Filter non converged calculations.')
    parser.add_argument('--mindist', default=None, type=float, help='Screen structures for short distances')
    parser.add_argument('--format', default='orca', type=str, choices=['aims', 'orca'], help='Input file format.')
    parser.add_argument('--forces', action='store_true', help='Extract forces. Currently only implemented for ORCA.')
    parser.add_argument('--energy_offset', type=float, default=None, help='Remove offset from molecule energies.')
    args = parser.parse_args()

    outputs = [os.path.join(args.outputs, d) for d in os.listdir(args.outputs)]

    if args.basisdef is None:
        if args.format == 'aims':
            basisdef = hamiltonian_parsing.extract_basis_definition_aims(outputs)
        elif args.format == 'orca':
            basisdef = hamiltonian_parsing.extract_basis_definition_orca(outputs)
        else:
            raise NotImplementedError(
                'Unrecognized reference data format {:s}'.format(args.format)
            )
    else:
        basisdef = np.load(args.basisdef)

    if args.orbital_energies is None:
        orbital_energies = None
    else:
        orbital_energies = np.load(args.orbital_energies)

    if args.format == 'aims':
        data = hamiltonian_parsing.AimsHamiltonianParser(args.database, basisdef,
                                              check_convergence=args.check_convergence,
                                              min_dist=args.mindist,
                                              orbital_energies=orbital_energies,
                                              noout=args.noout,
                                              forces=args.forces,
                                              energy_offset=args.energy_offset)

    elif args.format == 'orca':
        data = hamiltonian_parsing.OrcaHamiltonianParser(args.database, basisdef,
                                              check_convergence=args.check_convergence,
                                              min_dist=args.mindist,
                                              orbital_energies=orbital_energies,
                                              forces=args.forces,
                                              energy_offset=args.energy_offset)
    else:
        raise NotImplementedError('Unrecognized reference data format {:s}'.format(args.format))

    data.parse_directories(outputs)
