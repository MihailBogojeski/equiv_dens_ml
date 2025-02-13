# %%
import numpy as np


# orca-def2svp orbital basis
orca_def2svp_orbital_basis = {
    'H': [(1, 1, 0), (1, 1, 0), (1, 1, 1)],
    'C': [(6, 1, 0), (6, 1, 0), (6, 1, 0), (6, 1, 1), (6, 1, 1), (6, 1, 2)],
    'N': [(7, 1, 0), (7, 1, 0), (7, 1, 0), (7, 1, 1), (7, 1, 1), (7, 1, 2)],
    'O': [(8, 1, 0), (8, 1, 0), (8, 1, 0), (8, 1, 1), (8, 1, 1), (8, 1, 2)]
}
# print("---------------------------------")
# orca_dev2svp_orbitals_file = "../../../datasets/orca_def2svp_orbital_basis.npy"
# orca_dev2svp_orbital_basis = np.load(orca_dev2svp_orbitals_file, allow_pickle=True).item()

print("orca_def2svp")
print(orca_def2svp_orbital_basis)
for z in orca_def2svp_orbital_basis:
    print(z, orca_def2svp_orbital_basis[z], type(orca_def2svp_orbital_basis[z][0][0]))

# Convert to np.int32
for key in orca_def2svp_orbital_basis:
    orca_def2svp_orbital_basis[key] = np.array(orca_def2svp_orbital_basis[key], dtype=np.int32)

np.save('../../../datasets/orca_def2svp_orbital_basis.npy', orca_def2svp_orbital_basis, allow_pickle=True)
