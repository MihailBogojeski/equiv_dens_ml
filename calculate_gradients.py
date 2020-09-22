import os
import torch
from nn.neural_network_dens2 import NeuralNetwork
from training.density_dataset import AtomsDensityData
import numpy as np
from gradient_learning import utils as grad_utils

directory = '2020-08-20_LuCuJBCl'  # load directory name
model_name = 'LuCuJBCl'

checkpoint_dir = os.path.join(
    directory, 'checkpoints')  # checkpoint directory
# load latest checkpoint
checkpoint = torch.load(os.path.join(
    checkpoint_dir, 'latest_checkpoint.pth'), map_location='cpu')
latest_checkpoint = checkpoint['epoch']
ID = checkpoint['ID']  # load ID
args = checkpoint['args']  # overwrite args
args.use_gpu = True
args.load_from = os.path.join(directory, 'best_' + model_name + '.pth')

use_gpu = args.use_gpu and torch.cuda.is_available()

# load dataset(s)
print("loading density from" + args.dens_dataset + "...")
print("loading atoms from" + args.np_dataset + "...")

# density_file = '/home/mihail/data/water_rot/full_densities.hdf5'
# np_file = 'h2o_overlap_static.npy'

args.num_workers = 0

dataset = AtomsDensityData(np_path=args.np_dataset, density_path=args.dens_dataset,
                           orbitals_path=args.orbitals_file,
                           density_n_samp=args.density_subsamples,
                           required_properties=['density'],
                           center_positions=False,
                           radial_coeffs_file=args.radial_coeffs_file,
                           dtype=args.dtype)


equiv_model = NeuralNetwork(load_from=args.load_from)
equiv_model.to(args.dtype)
if args.use_gpu:
    equiv_model.cuda()
len(dataset)

splits = np.round(np.linspace(0, 4999 - 1, 1000)).astype(int)
print('splits', splits)
# continue computation

data_batches = np.load('gradient_batches.npy', allow_pickle=True)

for i in range(len(data_batches) + 1, len(splits)):
    torch.cuda.empty_cache()
    samples = dataset[list(range(splits[i - 1], splits[i]))]
    if use_gpu:
        for key in samples.keys():
            if isinstance(samples[key], torch.Tensor):
                samples[key] = samples[key].cuda()

    desc, a_d_desc, n_d_desc = grad_utils.from_r(equiv_model, samples['positions'])
    data_batches.append((desc, a_d_desc, n_d_desc))
    np.save('gradient_batches.npy', data_batches, allow_pickle=True)
