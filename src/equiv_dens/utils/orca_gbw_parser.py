'''
Parser for GBW “Geometry-Basis-Wavefunction” (.gbw) ORCA file
# Pointer @+0:  Internal ORCA data structures
# Pointer @+8:  Geometry
# Pointer @+16: BasisSet
# Pointer @+24: Orbitals
# Pointer @+32: ECP data

# The pointer to the orbitals is at byte 24 in the .gbw (long int)
# The first 5 long int values represent pointers into the file: 
'''


import struct
import sys
from pathlib import Path
import numpy as np

def parse_orca_gbw(gbw_path):

    '''
    ORCA uses the Mulliken convention, where the first index of the MO coefficients corresponds 
    to the atomic orbital (AO) index, and the second index corresponds to the MO index. 
    PySCF, on the other hand, uses the standard convention, 
    where the first index corresponds to the MO index and the second index corresponds to the AO index
    '''


    with open(gbw_path, "rb") as handle:
            
        handle.seek(24)
        offset = struct.unpack("<q", handle.read(8))[0]
        handle.seek(offset)
        
        
        # pointers
        operators = struct.unpack("<i", handle.read(4))[0]
        dimension = struct.unpack("<i", handle.read(4))[0]

        # precision '<dddd'
        coeffs_fmt = "<" + dimension**2 * "d"

        for i in range(operators):
            print(f'Operator: {i} \n')
            coeffs = struct.unpack(coeffs_fmt, handle.read(8 * dimension**2))
            occupations = [o[0] for o in struct.iter_unpack("<d", handle.read(8 * dimension)) ]
            energies = struct.iter_unpack("<d", handle.read(8 * dimension))
            irreps = [i[0] for i in struct.iter_unpack("<i", handle.read(4 * dimension))]
            cores = [c[0] for c in struct.iter_unpack("<i", handle.read(4 * dimension))]

            # MOs are returned in columns to have the same format as in pyscf
            coeffs = np.array(coeffs,np.double).reshape(-1, dimension)
            energies = np.array([en[0] for en in energies])
        
        return coeffs, energies

            

def parse_orca_dens(dens_path):


    with open(dens_path, "rb") as handle:
        # The pointer to the orbitals is at byte 24 in the .gbw (long int)
        handle.seek(0, 2)
        file_size = handle.tell()
        handle.seek(0, 0)
        
        offset, _ = struct.unpack(
            "ii", handle.read(8)
        )  # Don't know about the second integer
        # print("offset", offset)
        dens_size = offset - 8
        assert dens_size % 8 == 0
        dens_floats = dens_size // 8
        # print(f"Expecting {dens_floats} density doubles")
        densities = struct.unpack("d" * dens_floats, handle.read(dens_size))
        ndens = struct.unpack("i", handle.read(4))[0]
        
        # Block of 512 bytes meta data. I don't really know what is contained in there.
        meta = handle.read(512)
        until = meta.find(b"\x00")
        base_name = meta[:until].decode()
        #base_name, _ = get_name(meta)
        # Now multiple 512 byte blocks for each density follow
        dens_names = list()
        for i in range(ndens):
            dens_meta = handle.read(512)
            dens_name = dens_meta[:dens_meta.find(b"\x00")].decode()
            dens_names.append(dens_name)
            # don't know about the first item, 2nd items seems to 0
            _, _, nao1, nao2 = struct.unpack("iiii", handle.read(16))
            assert nao1 == nao2
            _ = struct.unpack("b", handle.read(1))[0]  # 0 byte
            assert _ == 0
        
        # Construct density matrices
        assert dens_floats % ndens == 0
        nao = int(np.sqrt(dens_floats // ndens))
        dens_shape = (nao, nao)
        densities = np.array(densities).reshape(ndens, *dens_shape)
        dens_exts = [Path(dens_name).suffix[1:] for dens_name in dens_names]
        dens_dict = {dens_ext: dens for dens_ext, dens in zip(dens_exts, densities)}

        return dens_dict