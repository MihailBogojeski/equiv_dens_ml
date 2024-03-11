import numpy as np
from horton import *  # pylint: disable=wildcard-import,unused-wildcard-import

def horton_partition(filepath, partitioning_type='Becke'):

    # Load the molden/fchk file from quantum chemistry
    fn_fchk = context.get_fn(filepath)
    mol = IOData.from_file(fn_fchk)

    # Calculate density on a grid
    grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers, mode='only')
    moldens = mol.obasis.compute_grid_density_dm(mol.get_dm_full(), grid.points)

    # Do the partitioning
    if partitioning_type = "Becke":
        wpart = BeckeWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens, local = True)
    elif partitioning_type = "IterativeStock":
        wpart = IterativeStockWPart(mol.coordinates, mol.numbers, mol.pseudo_numbers, grid, moldens, local = True)

    wpart.do_moments()

    cartesian_multipoles = wpart['cartesian_multipoles']
    pure_multipoles = wpart['pure_multipoles']
    radial_moments = wpart['radial_moments']

    partial_charges = cartesian_multipoles[:,0]
    partial_dipoles = cartesian_multipoles[:,1:4]

    return partial_charges, partial_dipoles
