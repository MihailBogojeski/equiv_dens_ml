# %%
import numpy as np

# %%
h2o_small = np.load('../../../datasets/h2o_small_train_augccpvdz.npy', allow_pickle=True).item()
h2o = np.load('../../../datasets/h2o_train_orca_def2svp.npy', allow_pickle=True).item()

print("h2o small keys:", h2o_small.keys())
print("h2o keys:", h2o.keys())

print("---------------------------------")
print("compare input units")

print("energy")
print(f"h2o small: {np.mean(h2o_small['energy'])} / {np.var(h2o_small['energy'])}")
print(f"h2o: {np.mean(h2o['energy'])} / {np.var(h2o['energy'])}")

print("position")
h2o_small_positions = np.concatenate(h2o_small['positions'])
h2o_small_pos_norm = np.linalg.norm(h2o_small_positions, axis=1)
h2o_small_pos_norm_mean = np.mean(h2o_small_pos_norm)
h2o_small_pos_norm_var = np.var(h2o_small_pos_norm)

h2o_positions = np.concatenate(h2o['positions'])
h2o_pos_norm = np.linalg.norm(h2o_positions, axis=1)
h2o_pos_norm_mean = np.mean(h2o_pos_norm)
h2o_pos_norm_var = np.var(h2o_pos_norm)

print(f"h2o small: {h2o_small_pos_norm_mean} / {h2o_small_pos_norm_var}")
print(f"h2o: {h2o_pos_norm_mean} / {h2o_pos_norm_var}")
print(f"h2o small / h2o positions norm mean scale: {h2o_small_pos_norm_mean / h2o_pos_norm_mean}")
print(f"h2o / h2o small positions norm mean scale: {h2o_pos_norm_mean / h2o_small_pos_norm_mean}")

print("forces")
h2o_small_forces = np.concatenate(h2o_small['forces'])
h2o_small_forces_norm = np.linalg.norm(h2o_small_forces, axis=1)
h2o_small_forces_norm_mean = np.mean(h2o_small_forces_norm)
h2o_small_forces_norm_var = np.var(h2o_small_forces_norm)

h2o_forces = np.concatenate(h2o['forces'])
h2o_forces_norm = np.linalg.norm(h2o_forces, axis=1)
h2o_forces_norm_mean = np.mean(h2o_forces_norm)
h2o_forces_norm_var = np.var(h2o_forces_norm)

print(f"h2o small: {h2o_small_forces_norm_mean} / {h2o_small_forces_norm_var}")
print(f"h2o: {h2o_forces_norm_mean} / {h2o_forces_norm_var}")
print(f"h2o small / h2o forces norm mean scale: {h2o_small_forces_norm_mean / h2o_forces_norm_mean}")
print(f"h2o / h2o small forces norm mean scale: {h2o_forces_norm_mean / h2o_small_forces_norm_mean}")


# %%

h2o_small_calc_results = np.load('../../../datasets/h2o_small_train_dft_augccpvdz_df_hm_dm_oe_calc.npy', allow_pickle=True)
h2o_calc_results = np.load('../../../datasets/h2o_train_orca_def2svp_hm_dm_oe_calc.npy', allow_pickle=True)

print(f"h2o small calc results: {h2o_small_calc_results[0][1].keys()}")
print(f"h2o calc results: {h2o_calc_results[0][1].keys()}")

print("---------------------------------")
print("compare output units")

print("eigenvalues of hamiltonian")
h2o_small_mo_energies = []
for i in range(len(h2o_small_calc_results)):
    mo_energies = h2o_small_calc_results[i][1]['mo_energies']
    h2o_small_mo_energies.append(mo_energies[0])
print(f"avg first mo energy h2o small: {np.mean(h2o_small_mo_energies)}")

h2o_mo_energies = []
for i in range(len(h2o_calc_results)):
    mo_energies = h2o_calc_results[i][1]['mo_energies']
    h2o_mo_energies.append(mo_energies[0])
print(f"avg first mo energy h2o: {np.mean(h2o_mo_energies)}")
# %%
