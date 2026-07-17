import sys
import numpy as np
import equiv_dens.utils.base as utils
from sklearn.cluster import KMeans
import torch


file = sys.argv[1]
num_samples = int(sys.argv[2])

data = np.load(file, allow_pickle=True).item()
pos = data['positions']

datasets = ['train', 'valid', 'test']
if pos.shape[0] > num_samples:
    inds = np.arange(pos.shape[0])
    sel_idx = np.random.choice(inds, size=(3 * num_samples,), replace=False)
    split_inds = [sel_idx[i * num_samples: (i+1)*num_samples] for i in range(3)]
    split_std = [np.std(data['forces'][split_inds[i]]) for i in range(3)]
    print('split std', split_std)
    sorted_idx = np.flip(np.argsort(split_std))
    print('split sort', sorted_idx)
   
    new_data = {}
    for j in range(len(sorted_idx)):
        for key in data.keys():
            if data[key].ndim > 1:
                new_data[key] = data[key][split_inds[sorted_idx[j]]]
            else:
                new_data[key] = data[key]
            print('key', key)
            print('shape', new_data[key].shape)
        save_file = file.split('.')[0] + '_random_' + str(num_samples) + '_' + datasets[j] + '.npy'
        np.save(save_file, new_data, allow_pickle=True)
