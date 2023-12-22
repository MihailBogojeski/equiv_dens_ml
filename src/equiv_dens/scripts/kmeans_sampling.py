import sys 
import numpy as np
import equiv_dens.utils.base as utils
from sklearn.cluster import KMeans
import torch


file = sys.argv[1]
num_samples = int(sys.argv[2])

subsample = 100000

data = np.load(file, allow_pickle=True).item()
pos = data['positions']

if pos.shape[0] > subsample:
    inds = np.arange(pos.shape[0])
    sel_idx = np.random.choice(inds, size=(subsample,), replace=False)
    for key in data.keys():
        print('key', key)
        print('shape', data[key].shape)
        if data[key].ndim > 1:
            data[key] = data[key][sel_idx]
        else:
            data[key] = data[key]
    pos = data['positions']


# print('pos shape', pos.shape)                                                                                                                                                                          
# print('num-samp', num_samp)                                                                                                                                                                            
inds = np.random.permutation(pos.shape[0])                                                                                                                                                               
# inds = np.arange(pos.shape[0])                                                                                                                                                                         
# print('inds', inds)                                                                                                                                                                                    
pos_shuff = pos[inds, :]                                                                                                                                                                                 
print(pos_shuff.shape)

dists, _ = utils.calculate_distances_and_directions(torch.tensor(pos_shuff))
dists = dists.numpy()
                                                                                                                                                                                                         
dists_flat = np.reshape(dists, (dists.shape[0], -1))                                                                                                                                               
# print('flat pos shape', pos_flat.shape)                                                                                                                                                                
clust = KMeans(n_clusters=num_samples).fit(dists_flat)
                                                                                                                                                                                                         
# np.save('clusters.npy', clusters)                                                                                                                                                                      
                                                                                                                                                                                                         
sample_inds = []                                                                                                                                                                                         
pos_dist = clust.transform(dists_flat)                                                                                                                                                                 
print(pos_dist.shape)
sample_inds.append(inds[np.argmin(pos_dist, axis=0)])                                                                                                                                                
                                                                                                                                                                                                         
save_file = file.split('.')[0] + '_kmeans_' + str(num_samples) + '.npy'
new_data = {}
for key in data.keys():
    print('key', key)
    print('shape', data[key].shape)
    if data[key].ndim > 1:
        new_data[key] = data[key][sample_inds[0]]
    else:
        new_data[key] = data[key]

np.save(save_file, new_data, allow_pickle=True)
