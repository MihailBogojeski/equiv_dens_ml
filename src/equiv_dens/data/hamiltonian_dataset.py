import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from itertools import accumulate
from .sqlite_database import HamiltonianDatabase


class HamiltonianDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, dtype=torch.float32):
        super(HamiltonianDataset, self).__init__()
        self.dtype  = dtype
        self.database = HamiltonianDatabase(filepath)

        #collect the orbitals, which give the shape of the hamiltonian
        orbitals = []
        for Z in self.database.Z:
            orbitals.append(tuple((int(Z),int(l)) for l in self.database.get_orbitals(Z)))
        self.orbitals = tuple(orbitals)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        return idx #just return the idx, the custom collate_fn does the querying

    #collate function to collate several data points into a batch, should be passed
    #to data loader (default_collate is not efficient, because there should be a batch-wise query)
    def collate_fn(self, batch):
        all_data = self.database[batch] #fetch the batch data
        R, E, F, H, S, C = [], [], [], [], [], []
        for batch_num, data in enumerate(all_data):
            R_, E_, F_, H_, S_, C_ = data
            R.append(torch.tensor(R_, dtype=self.dtype))
            E.append(torch.tensor(E_, dtype=self.dtype))
            F.append(torch.tensor(F_, dtype=self.dtype))
            H.append(torch.tensor(H_, dtype=self.dtype))
            S.append(torch.tensor(S_, dtype=self.dtype))
            C.append(torch.tensor(C_, dtype=self.dtype))

        return {'positions': torch.stack(R), 
                'energy': torch.stack(E),
                'forces': torch.stack(F),
                'full_hamiltonian': torch.stack(H),
                'overlap_matrix': torch.stack(S),
                'core_hamiltonian': torch.stack(C)}     

def seeded_random_split(dataset, lengths, seed=None):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    This is very similar to the pytorch equivalent, but this version allows a seed to
    be specified in order to make the split reproducible

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    n = len(dataset)
    n_train, n_valid, n_test = lengths[0], lengths[1], lengths[2]
    if n_train + n_valid + n_test != n or n_test < 0:
        n_train = min(n_train, n)
        n_valid = min(n_valid, n - n_train)
        n_test = max(0, n - n_train - n_valid)
        lengths = [n_train, n_valid, n_test]

    if sum(lengths) != n:
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = np.random.RandomState(seed=seed).permutation(n)
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in zip(accumulate(lengths), lengths)]
