import numpy as np
import torch

grads = list(np.load('gradient_batches_600.npy', allow_pickle=True))

# for i in range(len(grads)):
#     print(type(grads[i][0]))
#     if isinstance(grads[i][0], torch.Tensor):
#         print('changing type')
#         grads[i][0] = grads[i][0].detach().cpu().numpy()
#         grads[i][1] = grads[i][1].detach().cpu().numpy()
#         grads[i][2] = grads[i][2].detach().cpu().numpy()
#
# np.save('gradient_batches_600.npy', grads, allow_pickle=True)

features = {'desc': None, 'desc_grad': None, 'desc_num_grad': None}

for i in range(len(grads)):
    print('desc size', grads[i][0].shape)
    print('grad size', grads[i][1].shape)
    print('n_grad size', grads[i][2].shape)
    if features['desc'] is None:
        features['desc'] = grads[i][0]
        features['desc_grad'] = grads[i][1]
        features['desc_num_grad'] = grads[i][2]
    else:
        features['desc'] = np.concatenate((features['desc'], grads[i][0]), axis=0)
        features['desc_grad'] = np.concatenate((features['desc_grad'], grads[i][1]), axis=0)
        features['desc_num_grad'] = np.concatenate((features['desc_num_grad'], grads[i][2]), axis=0)


print('desc shape', features['desc'].shape)
print('grad shape', features['desc_grad'].shape)
print('n_grad shape', features['desc_num_grad'].shape)

np.savez('h2o_density_grad_desc.npz', **features)
