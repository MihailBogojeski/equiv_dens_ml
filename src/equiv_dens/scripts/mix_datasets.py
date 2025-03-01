# %%

import numpy as np
from equiv_dens.utils.base import calc_dict_to_npy

data_dict = {
    "water": {
        "np_train": "../../../datasets/water_train_orca_def2svp.npy",
        "np_valid": "../../../datasets/water_valid_orca_def2svp.npy",
        "np_test": "../../../datasets/water_test_orca_def2svp.npy",
        "calc_train": "../../../datasets/water_train_orca_def2svp_hm_dm_oe_calc.npy",
        "calc_valid": "../../../datasets/water_valid_orca_def2svp_hm_dm_oe_calc.npy",
        "calc_test": "../../../datasets/water_test_orca_def2svp_hm_dm_oe_calc.npy",
        "split_sizes": [500, 500, 3999]
    },
    "ethanol": {
        "np_train": "../../../datasets/ethanol_train_orca_def2svp.npy",
        "np_valid": "../../../datasets/ethanol_valid_orca_def2svp.npy",
        "np_test": "../../../datasets/ethanol_test_orca_def2svp.npy",
        "calc_train": "../../../datasets/ethanol_train_orca_def2svp_hm_dm_oe_calc.npy",
        "calc_valid": "../../../datasets/ethanol_valid_orca_def2svp_hm_dm_oe_calc.npy",
        "calc_test": "../../../datasets/ethanol_test_orca_def2svp_hm_dm_oe_calc.npy",
        "split_sizes": [25000, 500, 4500]
    },
    "mda-enol": {
        "np_train": "../../../datasets/mda-enol_train_orca_def2svp.npy",
        "np_valid": "../../../datasets/mda-enol_valid_orca_def2svp.npy",
        "np_test": "../../../datasets/mda-enol_test_orca_def2svp.npy",
        "calc_train": "../../../datasets/mda-enol_train_orca_def2svp_hm_dm_oe_calc.npy",
        "calc_valid": "../../../datasets/mda-enol_valid_orca_def2svp_hm_dm_oe_calc.npy",
        "calc_test": "../../../datasets/mda-enol_test_orca_def2svp_hm_dm_oe_calc.npy",
        "split_sizes": [25000, 500, 1478]
    },
    "uracil": {
        "np_train": "../../../datasets/uracil_train_orca_def2svp.npy",
        "np_valid": "../../../datasets/uracil_valid_orca_def2svp.npy",
        "np_test": "../../../datasets/uracil_test_orca_def2svp.npy",
        "calc_train": "../../../datasets/uracil_train_orca_def2svp_hm_dm_oe_calc.npy",
        "calc_valid": "../../../datasets/uracil_valid_orca_def2svp_hm_dm_oe_calc.npy",
        "calc_test": "../../../datasets/uracil_test_orca_def2svp_hm_dm_oe_calc.npy",
        "split_sizes": [25000, 500, 4500]
    }
}
# %%
def subset_of_calc(calc_results, n_structures=10, idx=None, seed=None):

    if n_structures is None:
        return calc_results
    
    np.random.seed(seed)

    if idx is None:
        idx = np.random.choice(len(calc_results), n_structures, replace=False)

    subset_calc_results = []
    for i in idx:
        subset_calc_results.append(calc_results[i])

    return subset_calc_results


def join_calc_results(calc_results):
    new_calc_results = []
    for calc_result in calc_results:
        new_calc_results.extend(calc_result)
    return new_calc_results


# %%
SEED = 0

# MIX_MOLECULES = ["water", "ethanol"]
mix_splits = ["train", "valid", "test"]
# mix_split_sizes = {
#     "water": [500, 500, 3999],
#     "ethanol": [500, 500, 4500],
#     "mda-enol": [25000, 500, 1478],
#     "uracil": [25000, 500, 4500]
# }
mix_split_sizes = {
    "water": [10, 10, 10],
    "ethanol": [10, 10, 10],
    "mda-enol": [0, 0, 0],
    "uracil": [0, 0, 0]
}


for i_split, split in enumerate(mix_splits):
    calc_results = []
    for mol in mix_split_sizes.keys():

        print(f"{mol} {split} {mix_split_sizes[mol][i_split]}")

        if mix_split_sizes[mol][i_split] < 1:  # if mix ratio for molecule and split is 0, skip
            continue

        calc_result = np.load(data_dict[mol][f"calc_{split}"], allow_pickle=True)

        calc_result = subset_of_calc(calc_result, n_structures=mix_split_sizes[mol][i_split], seed=SEED)
        calc_results.append(calc_result)

    calc_results = join_calc_results(calc_results)

    if len(calc_results) < 1:
        print(f"no calc_results for {split}. check mixing ratios")
        continue

    print(f"calc_results for split {split}: {len(calc_results)}")
    calc_save_path = f"../../../datasets/mix_{split}_orca_def2svp_hm_dm_oe_calc.npy"
    np.save(calc_save_path, calc_results, allow_pickle=True)

    np_dataset = calc_dict_to_npy(calc_results,
                                  convert_forces=False,
                                  compress_atoms=True)
    
    np_save_path = f"../../../datasets/mix_{split}_orca_def2svp.npy"
    np.save(np_save_path, np_dataset, allow_pickle=True)
    


