Install schnetpack (instuctions on github)
Checkout to older version
  git checkout 1459bd6603ae484d8da482d92c3591900f770081

save these files from master branch:
  datasets/augccpvqzjkfit_orbital_basis_df.npy
  datasets/augccpvqzjkfit_orbital_basis_libcint_df.npy
  datasets/augccpvqzjkfit_radial_coeffs_df.npy
  datasets/augccpvqzjkfit_radial_coeffs_libcint_df.npy

Switch to old branch:
  git checkout 2c9d0d4d74dc6f263000bd5d03c82eea05e6dce9

Copy the above files in the datasets folder.

Config MD settings in config/paper/thiophene_poly_all_001_coreless_dpm.txt

Start MD run:
  python run.py md @config/md/nn/polythiophene_2mer.txt

Convert md trajectory to numpy files:
  python scripts/md/schnetpack_md_to_npy.py md_logs/2023-05-27_9yXHffp1/<simulation_file_name>.hdf5 <take_every_n_steps> <n_trajectories>

Evaluate dipole moments on trajectories using newer model:
  python run.py infer --model config/paper/thiophene_poly_all_001_coreless_dpm.txt --trajectory md_logs/2023-05-27_9yXHffp1/simulation_local_all_2mer_md_001_0.npy --dpm-intor --batch-size 5
