import torch
from torch.utils.data import Sampler
import numpy as np
from typing import Iterator, List, Iterable, Union


class SimilarSizeSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    def __init__(self, data_source, replacement: bool = False,
                 num_samples=None, generator=None, shuffle=False, max_bucket_size=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.max_bucket_size = max_bucket_size
        self.generator = generator
        self.shuffle = shuffle

        if max_bucket_size is None:
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
            self.num_electrons.append(torch.sum(self.data_source.get_basic_properties([idx])['atom_numbers']).item())
        sort_idx = np.argsort(self.num_electrons)
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
                groups = torch.tensor(list(self.elec_num_groups.keys()))
                if self.shuffle:
                    perm_idx = torch.randperm(len(groups), generator=generator).to(dtype=torch.long)
                    groups = torch.index_select(groups, 0, perm_idx).tolist()
                all_idxs = []
                for group in groups:
                    idxs = torch.tensor(self.elec_num_groups[group])
                    if self.shuffle:
                        perm_idx = torch.randperm(len(idxs), generator=generator).to(dtype=torch.long)
                        idxs = torch.index_select(idxs, 0, perm_idx)
                    all_idxs.extend(idxs.tolist())
                yield from all_idxs[:self.num_samples % n]
            
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
