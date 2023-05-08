'''
Parser for GBW “Geometry-Basis-Wavefunction” (.gbw) ORCA file
https://orcaforum.kofo.mpg.de/viewtopic.php?f=8&t=3299 from PERL rewritten
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
from copy import deepcopy

def parse_orca_gbw(gbw_path):

    '''
    ORCA uses the Mulliken convention, where the first index of the MO coefficients corresponds 
    to the atomic orbital (AO) index, and the second index corresponds to the MO index. 
    PySCF, on the other hand, uses the standard convention, #
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

            coeffs = struct.unpack(coeffs_fmt, handle.read(8 * dimension**2))
            occupations = [o[0] for o in struct.iter_unpack("<d", handle.read(8 * dimension)) ]
            energies = struct.iter_unpack("<d", handle.read(8 * dimension))
            irreps = [i[0] for i in struct.iter_unpack("<i", handle.read(4 * dimension))]
            cores = [c[0] for c in struct.iter_unpack("<i", handle.read(4 * dimension))]

            # MOs are returned in columns to have the same format as in pyscf
            coeffs = np.array(coeffs,np.double).reshape(-1, dimension)
            energies = np.array([en[0] for en in energies])
        
        return coeffs, energies,occupations

            

def parse_orca_dens(dens_path):

    # taken from pysisyphus.calculators.ORCA
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
    



def parse_orca_ovlp_matrix(out_path):

    with open(out_path,"r") as f:

        # tags for ovlp matrix

        tag1 ='OVERLAP MATRIX\n'
        tag2 = 'S**(-1/2) MATRIX\n'
        tag3 = 'Number of basis functions'
        tag4 = 'Number of shells'


        lines = "".join(f.readlines())
        num_basis = int(lines.split(tag3)[1].split(tag4)[0].split("\n")[0].split("...")[-1])
        input_string = ",".join(lines.split(tag1)[2].split(tag2)[0].split("\n")[2:-2])
        values = input_string.split(",")

        m = len(values)-num_basis
        filter_idx = []

        for n in range(len(values)):
            
            A  = len([n for n in values[0].split(" ") if n != ""])
            A_2 = len([n for n in values[n].split(" ") if n != ""])
            
            B  = len([n for n in values[-1].split(" ") if n != ""])
            
            if (A_2 != A) and (A_2 != B):
                filter_idx.append(n)

        A_3 = (A-1)*-1

        if A != A_2 or len(filter_idx) != 0:
            values2 = [values[n] for n in range((filter_idx[-1])) if n not in filter_idx[:-1]]
            blocks = list(range(0,len(filter_idx)))


            final = []
            for b in blocks:
                
                block = values2[(b*num_basis):(b+1)*num_basis]
                
                tmp_blocks = np.concatenate([
                np.array(block[n].split("  ")[A_3:],dtype=np.float32) for n in range(len(block))
                ]).reshape(num_basis,(A_3*-1))
                
                final.append(tmp_blocks)

            A_4 = (A_2-1)*-1
            block = values[filter_idx[-1]+1:]
            tmp_blocks = np.concatenate([
            np.array(block[n].split("  ")[A_4:],dtype=np.float32) for n in range(len(block))
            ]).reshape(num_basis,(A_4*-1))

            final.append(tmp_blocks)
        
        else:

            final = []

                
            tmp_blocks = np.concatenate([
            np.array(values[n].split("  ")[A_3:],dtype=np.float32) for n in range(len(values))
            ]).reshape(num_basis,(A_3*-1))
                
            final.append(tmp_blocks)


        ovlp_matrix = np.hstack(tuple(final))
    
    return ovlp_matrix
    

def parse_orca_basisset(out_path,elem):
    
    
    with open(out_path,"r") as f:
        
        
        tag1 = 'BASIS SET IN INPUT FORMAT\n'
        tag2 = 'AUXILIARY/J BASIS SET INFORMATION\n'
       
        lines = "".join(f.readlines())
        tag4 = "NewGTO "+elem
        A1 = (lines.split(tag1)[1].split(tag2)[0].split(tag4)[1].split("# ")[0].split("end")[0])
        
        basisset = []

        crude_basisset = A1.split("\n")[1:-1]

        for w in range(len(crude_basisset)):

            if bool(set([*crude_basisset[w]]) & set(["S","P","D"])):

                    shell = [*crude_basisset[w]][1]
                    s = elem+"\t"+shell+"\n"
                    basisset.append(s)
            else:
                    tmp = "\t\t".join([a for a in crude_basisset[w].split(" ") if a != ""][1:])+"\n"
                    basisset.append(tmp)

        basisset = "".join(basisset)
        
        return basisset
    

def orbital_ordering():
    
    # ordering from ORCA to pyscf means first entry is ORCA and second one Pyscf
    # Pyscf has px,py,pz ordering [0,1,2] and orca pz,px,py ordering [2,0,1] (Condon-shortly notation for p orbs)
    # based on orbital cartesian coordinate number to be added to index
    # meaning index 12 is px orbital in ORCA
    # PySCF px orbital is 2 entries later than px ORCA so adding 2 to index ergo idx = 14
    
    
    p_orbs = {
        "x":2,  # 0 : 2 pz
        "y":-1,  # 1 : 0 px
        "z":-1   # 2 : 1 py
             }
    d_orbs = {
        "xy":2, # 0 : 2
        "yz":2, # 1 : 3
        "z^2":-1, # 2: 1
        "xz":1, # 3: 4
        "x2-y2":-4 # 4:0
    }
    
    # 5 to 6 is sign change ORCA pos number wird zu pyscf neg number
    # 6 to 0 is sign change ORCA pos number wird zu pyscf neg number
    f_orbs = {
        "-3":3, # 0 : 3
        "-2":3, # 1 : 4
        "-1":0, # 2 : 2
        "+0":2, # 3 : 5
        "+1":-3, # 4 : 1
        "+2":1, # 5 : 6
        "+3":-6  # 6 : 0
    } 
    
    conv_orbs = {"p":p_orbs, "d":d_orbs, "f":f_orbs}
    
    return conv_orbs


def def2_tzvp_orca_to_pyscf_swap(arr,ao_labels):
    
    '''
    ordering from ORCA (Version 5.0.1) to pyscf (2.2.1) means first entry is ORCA and second one Pyscf
    Pyscf has px,py,pz ordering [0,1,2] and orca pz,px,py ordering [2,0,1] (Condon-shortly notation for p orbs)
    based on orbital cartesian coordinate number to be added to index
    meaning index 12 is px orbital in ORCA
    PySCF px orbital is 2 entries later than px ORCA so adding 2 to index ergo idx = 14


    Pyscf       ORCA
    px   ->     pz
    py   ->     px
    pz   ->     py
    
    dxy  ->     dz^2
    dyz  ->     dxz
    dz^2 ->     dyz
    dxz  ->     dx^2-y^2
    dx^2-y^2 -> dxy

    f-3  ->     f+3
    f-2  ->     f+0
    f-1  ->     f+1
    f+0  ->     f-1
    f+1  ->     f+2
    f+2  ->     f-2
    f+3  ->     f-3
    
    '''

    p_orbs = {
        "x":2,  # 0 : 2 pz
        "y":-1,  # 1 : 0 px
        "z":-1   # 2 : 1 py
             }
    d_orbs = {
        "xy":2, # 0 : 2
        "yz":2, # 1 : 3
        "z^2":-1, # 2: 1
        "xz":1, # 3: 4
        "x2-y2":-4 # 4:0
    }
    
    # 5 to 6 is sign change ORCA pos number wird zu pyscf neg number
    # 6 to 0 is sign change ORCA pos number wird zu pyscf neg number
    f_orbs = {
        "-3":3, # 0 : 3
        "-2":3, # 1 : 4
        "-1":0, # 2 : 2
        "+0":2, # 3 : 5
        "+1":-3, # 4 : 1
        "+2":1, # 5 : 6
        "+3":-6  # 6 : 0
    } 
    
    conv_orbs = {"p":p_orbs, "d":d_orbs, "f":f_orbs}

     
    C = deepcopy(arr)

    idx_map_orca = []
    idx_map_pyscf = []
    sign_change = []

    for idx,ao in enumerate(ao_labels):

        for orb in ["p","d","f"]:

            if orb in ao_labels[idx]:
                conv = conv_orbs[orb][ao_labels[idx].split(" ")[2].split(orb)[-1]]
                idx_map_pyscf.append(idx)
                idx_map_orca.append(idx+conv)
                
                # Collecting indices for f orbital sign flip
                if ao_labels[idx].split(" ")[2].split(orb)[-1] in ["-3","+3"]:
                    sign_change.append(idx)
    
    C[idx_map_orca] = C[idx_map_pyscf]  

    for n in range(C.shape[0]):

        C[[n],idx_map_orca] = C[[n],idx_map_pyscf]
        for s in sign_change:
            C[[n],s] *= -1
        
    
    # sign change of f orbitals[-3, +3]
    # sign change for f+0 not applicable
    for s in sign_change:
        C[s] *= -1  
    

    return C