from torch.utils.data import _utils
from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter, _DatasetKind


# This function used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate = _utils.collate.default_collate


class BatchLoader(DataLoader):

    def __iter__(self):
        return _BatchDataLoaderIter(self)

class _BatchDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        assert self.timeout == 0
        assert self.num_workers == 0


    def __next__(self):
        index = self._next_index()  # may raise StopIteration
        print('index', index)
        data = self.dataset[index]
        if self.pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data

