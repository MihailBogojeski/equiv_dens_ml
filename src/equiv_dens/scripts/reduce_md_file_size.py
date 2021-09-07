from schnetpack.md.utils import HDF5Loader
import json
import time
import numpy as np

data = HDF5Loader('/home/ml-dft/equiv_dens/md_logs/2021-06-21_UAD77s3O/simulation_gpu_dipole_test_new_lvl4.hdf5', load_properties=False)

# {'energy': [0, 1],                                                    
#  'forces': [1, 28],                                                   
#  'dipole_moment': [28, 31],                                           
#  'density': [31, 181747]}                                             

slc = slice(313000, 320000)

shapes = json.loads(data.database['properties'].attrs['shapes'])
positions = json.loads(data.database['properties'].attrs['positions'])
props = {key: data.properties[key] for key in data.properties.keys()}
print('atomic numbers', props['_atomic_numbers'])
props['_atomic_numbers'] = props['_atomic_numbers'][[0]]
for prop in props.keys():
    if isinstance(props[prop], np.ndarray) and prop != '_atomic_numbers':
        print('prop', prop)
        print('shape before', props[prop].shape)
        props[prop] = props[prop][slc]
        print('shape after', props[prop].shape)
start = time.time()
all = data.database['properties'][slc, :, :, :positions['dipole_moment'][1]]
print('elapsed all', time.time() - start)
print('all.shape', all.shape)

print('atomic numbers', props['_atomic_numbers'])
print('atomic numbers shape', props['_atomic_numbers'].shape)
start = time.time()
prop_list = ['energy', 'forces', 'dipole_moment']
for prop in prop_list:
    props[prop] = all[..., slice(*positions[prop])].reshape(*all.shape[:-1], *shapes[prop])
    print(prop, 'shape', props[prop].shape)

np.savez('/home/ml-dft/equiv_dens/md_logs/2021-06-21_UAD77s3O/simulation_gpu_dipole_test_new_lvl4_313000.npz', **props)
