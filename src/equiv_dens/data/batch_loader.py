from torch.utils.data import _utils
from torch.utils.data.dataloader import DataLoader, _SingleProcessDataLoaderIter


# This function used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate = _utils.collate.default_collate


class BatchLoader(DataLoader):

    def __iter__(self):
        return _SingleProcessBatchLoaderIter(self)


class _SingleProcessBatchLoaderIter(_SingleProcessDataLoaderIter):

    def __next__(self):
        if hasattr(self, 'dataset'):
            dataset = self.dataset
        else:
            dataset = self._dataset
        if hasattr(self, 'pin_memory'):
            pin_memory = self.pin_memory
        else:
            pin_memory = self._pin_memory
        indices = self._next_index()  # may raise StopIteration
        batch = dataset[indices]
        if pin_memory:
            batch = _utils.pin_memory.pin_memory(batch)
        return batch
