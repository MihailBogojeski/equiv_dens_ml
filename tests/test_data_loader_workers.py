"""Training DataLoader must honor --num_workers.

The campaign jobs were killed on a100_chemistry after ~2 h with 0% GPU
util. The model was on CUDA (`Training on one GPU:`), but
``set_up_data_loader`` hardcoded ``num_workers=0`` and ``__getitem__``
returns only an index, so ``get_properties`` (neighbor lists, optional
PySCF grids) ran on the training process between steps. nvidia-smi
sampled idle. Workers run that collate off the GPU process so the
forward/backward can stay busy.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from equiv_dens.data.custom_samplers import set_up_data_loader  # noqa: E402


class _IndexDataset(Dataset):
    """Mimic AtomsDensityData: __getitem__ returns an index, work is in collate."""

    def __init__(self, n: int = 8):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx

    def get_properties(self, idx):
        return {"idx": torch.tensor(list(idx), dtype=torch.long)}


def test_num_workers_default_is_zero_for_back_compat():
    loader = set_up_data_loader(_IndexDataset(), batch_size=2)
    assert loader.num_workers == 0


def test_num_workers_is_honored():
    loader = set_up_data_loader(_IndexDataset(), batch_size=2, num_workers=2)
    assert loader.num_workers == 2
    assert loader.prefetch_factor == 2
    assert loader.persistent_workers is True
    ctx = loader.multiprocessing_context
    assert ctx is not None
    assert ctx.get_start_method() == "spawn"


def test_zero_workers_constructs():
    loader = set_up_data_loader(_IndexDataset(), batch_size=2, num_workers=0)
    assert loader.num_workers == 0


def test_collate_still_builds_a_batch():
    loader = set_up_data_loader(_IndexDataset(n=4), batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["idx"].shape[0] == 2


def test_spawn_collate_is_picklable():
    """spawn workers pickle collate_fn; a nested closure raises AttributeError.

    That is what killed the density resumes after num_workers was wired through
    (``Can't get local object 'set_up_data_loader.<locals>.collate_fn'``).
    """
    loader = set_up_data_loader(_IndexDataset(n=8), batch_size=2, num_workers=2)
    pickle.dumps(loader.collate_fn)
    subset_loader = set_up_data_loader(
        Subset(_IndexDataset(n=8), [0, 1, 2, 3]),
        batch_size=2,
        num_workers=2,
        shuffle=False,
    )
    pickle.dumps(subset_loader.collate_fn)
    batch = next(iter(loader))
    assert batch["idx"].shape[0] == 2


_TRAIN_SCRIPTS = (
    "scripts/training/train_only_en.py",
    "scripts/training/train_all.py",
    "scripts/training/train.py",
    "scripts/training/train_dens.py",
)


def test_training_scripts_pass_num_workers_to_every_loader(implementation_text):
    """Energy mode ignored --num_workers because the call never forwarded it.

    Inspect the canonical workflow behind the old script entry point. Count
    ``set_up_data_loader(`` against ``num_workers=args.num_workers``.
    """
    for rel in _TRAIN_SCRIPTS:
        text = implementation_text(_REPO_ROOT / rel)
        n_calls = text.count("set_up_data_loader(")
        n_workers = text.count("num_workers=args.num_workers")
        assert n_calls > 0, f"{rel} has no set_up_data_loader calls"
        assert n_workers >= n_calls, (
            f"{rel}: {n_calls} set_up_data_loader calls but only "
            f"{n_workers} pass num_workers=args.num_workers"
        )


def test_train_only_en_passes_batch_efficiency_not_use_gpu(implementation_text):
    """The 4th positional is batch_efficiency; stuffing use_gpu there is a bug."""
    text = implementation_text(_REPO_ROOT / "scripts/training/train_only_en.py")
    assert "args.batch_efficiency" in text
    assert "args.electron_num_batching, use_gpu, True)" not in text
