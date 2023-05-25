import torch
from torch.utils.data import Sampler, Dataset
import numpy as np
from typing import Iterator, List, Iterable, Union


class SimilarSizeSampler(Sampler[int]):
    """Sample the data while attempting to sample molecules of the similar size (wrt. number of electrons) in the same batch."""

    def __init__(self, data_source: Dataset, replacement: bool = False,
                 num_samples: int = None, generator: torch.Generator = None,
                 shuffle: bool = False, max_bucket_size: int = None,
                 electron_batch_size: int = None) -> None:
        """Initialize a new instance of the SimilarSizeSampler class.

        Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): whether the samples are with or without replacement
        num_samples (int): number of samples to draw. If None, draw as many as possible.
        generator (torch.Generator): Generator used in sampling.
        shuffle (bool): set to True to have the sampler shuffle dataset.
        max_bucket_size (int): maximum number of samples in a bucket. If None,
        estimate bucket size so that it fits number of electrons for batching.
        electron_batch_size (int): Maximum number of total electrons in a batch.
        Used to estimate bucket size if necessary.
        """
        super().__init__(data_source)
        self.data_source = data_source
        print('data source len', len(data_source))
        self.dataset = None
        self.replacement = replacement
        self._num_samples = num_samples
        self.max_bucket_size = max_bucket_size
        self.generator = generator
        self.shuffle = shuffle
        self.electron_batch_size = electron_batch_size

        if isinstance(self.data_source, torch.utils.data.Subset):
            self.dataset = self.data_source.dataset
        else:
            self.dataset = self.data_source

        if max_bucket_size is None and electron_batch_size is None:
            self.max_bucket_size = (self.num_samples / 10) + 1

        if not isinstance(self.replacement, bool):
            raise TypeError("replacement should be a boolean value, but got "
                            "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))
        self.group_data_by_elec_num()

    def group_data_by_elec_num(self):
        self.num_electrons = []
        for i in range(self.num_samples):
            idx = self.data_source[i]
            self.num_electrons.append(torch.sum(self.dataset.get_basic_properties([idx])['atom_numbers']).item())
        sort_idx = np.argsort(self.num_electrons)
        if self.max_bucket_size is None:
            min_elec_num = self.num_electrons[sort_idx[0]]
            self.max_bucket_size = self.electron_batch_size // min_elec_num
        num_buckets = np.ceil(len(sort_idx)/self.max_bucket_size).astype(int)
        idxs = np.linspace(0, len(sort_idx) - 1, num=num_buckets).astype(int)
        self.elec_num_groups = {}
        for i in range(len(idxs) - 1):
            self.elec_num_groups[i] = sort_idx[idxs[i]:idxs[i + 1]]

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        if self.replacement:
            for _ in range(self.num_samples // 32):
                groups = torch.tensor(list(self.elec_num_groups.keys()))
                if self.shuffle:
                    perm_idx = torch.randperm(len(groups), generator=generator).to(dtype=torch.long)
                    groups = torch.index_select(groups, 0, perm_idx)
                groups = groups.tolist()
                all_idxs = []
                for group in groups:
                    idxs = torch.tensor(self.elec_num_groups[group])
                    if self.shuffle:
                        perm_idx = torch.randperm(len(idxs), generator=generator).to(dtype=torch.long)
                        idxs = torch.index_select(idxs, 0, perm_idx)
                    all_idxs.extend(idxs[:32].tolist())
                yield from all_idxs[:32]
            groups = torch.tensor(list(self.elec_num_groups.keys()))
            if self.shuffle:
                perm_idx = torch.randperm(len(groups), generator=generator).to(dtype=torch.long)
                groups = torch.index_select(groups, 0, perm_idx).tolist()
            all_idxs = []
            groups = groups.tolist()
            for group in groups:
                idxs = torch.tensor(self.elec_num_groups[group])
                if self.shuffle:
                    perm_idx = torch.randperm(len(idxs), generator=generator).to(dtype=torch.long)
                    idxs = torch.index_select(idxs, 0, perm_idx)
                all_idxs.extend(idxs.tolist())
            yield from all_idxs[:self.num_samples % 32]
        else:
            for _ in range(self.num_samples // n):
                groups = torch.tensor(list(self.elec_num_groups.keys()))
                if self.shuffle:
                    perm_idx = torch.randperm(len(groups), generator=generator).to(dtype=torch.long)
                    groups = torch.index_select(groups, 0, perm_idx)
                groups = groups.tolist()
                all_idxs = []
                for group in groups:
                    idxs = torch.tensor(self.elec_num_groups[group])
                    if self.shuffle:
                        perm_idx = torch.randperm(len(idxs), generator=generator).to(dtype=torch.long)
                        idxs = torch.index_select(idxs, 0, perm_idx)
                    all_idxs.extend(idxs.tolist())
                yield from all_idxs

    def __len__(self) -> int:
        return self.num_samples


class AdaptiveBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], max_num_elec: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(max_num_elec, int) or isinstance(max_num_elec, bool) or \
                max_num_elec <= 0:
            raise ValueError("max_num_elec should be a positive integer value, "
                             "but got max_num_elec={}".format(max_num_elec))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.max_num_elec = max_num_elec
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = []
                    sum_elec = 0
                    while True:
                        idx = next(sampler_iter)
                        sum_elec += self.sampler.num_electrons[idx]
                        if sum_elec > self.max_num_elec:
                            break
                        batch.append(idx)
                    yield batch
                except StopIteration:
                    break
        else:
            batch = []
            sum_elec = 0
            for idx in self.sampler:
                num_elec = self.sampler.num_electrons[idx]
                sum_elec += num_elec
                if sum_elec > self.max_num_elec:
                    yield batch
                    sum_elec = num_elec
                    batch = []
                batch.append(idx)
            if sum_elec > 0:
                yield batch

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.max_elec_num  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.max_elec_num - 1) // self.max_elec_num  # type: ignore[arg-type]

def set_up_data_loader(dataset: Dataset, batch_size: int = 1,
                       electron_num_batching: bool = False,
                       use_gpu: bool = False, shuffle: bool = True):
    """Set up data loader for the dataset.

    Args:
    dataset (Dataset): The dataset to be used in the loader.
    batch_size (int): The batch size.
    electron_num_batching (bool): True to use adaptive batching based on number of electrons.
    use_gpu (bool): True to use GPU.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        def collate_fn(batch):
            return dataset.dataset.get_properties(batch)
    else:
        def collate_fn(batch):
            return dataset.get_properties(batch)
    if electron_num_batching:
        sampler = SimilarSizeSampler(dataset, shuffle=shuffle, electron_batch_size=batch_size)
        batch_sampler = AdaptiveBatchSampler(sampler, max_num_elec=batch_size,
                                             drop_last=False)
        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler,
                                                  num_workers=0, pin_memory=use_gpu,
                                                  collate_fn=collate_fn)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  num_workers=0, pin_memory=use_gpu,
                                                  shuffle=shuffle,
                                                  collate_fn=collate_fn)
    return data_loader
