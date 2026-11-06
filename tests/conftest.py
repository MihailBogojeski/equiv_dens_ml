"""Test-only helpers for assertions about relocated implementation source."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def implementation_text():
    root = Path(__file__).resolve().parents[1]
    mapping = {entry['old_path']: entry['canonical_target']
               for entry in json.loads((root/'docs/refactor/migration_map.json').read_text())}

    def read(path, **kwargs):
        path = Path(path)
        relative = str(path.relative_to(root))
        target = mapping.get(relative)
        if target is not None:
            if target.startswith('equiv_dens.'):
                path = root/'src'/(target.replace('.', '/')+'.py')
            else:
                path = root/target
        return path.read_text(**kwargs)
    return read


# --- markers for tests that need local data or a GPU (skipped in upstream CI) ---------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
# present once scripts/data/fetch_reproduction_kit.sh (or download_hf_datasets.sh) has populated datasets/
_LOCAL_DATA_SENTINEL = _REPO_ROOT / "datasets" / "revision" / "sad_pbe_augccpvdz.npy"


def pytest_configure(config):
    config.addinivalue_line("markers", "local_data: needs the datasets/ tree from the Hugging Face kit; "
                                       "skipped when it is not populated")
    config.addinivalue_line("markers", "gpu: needs a CUDA device; skipped without one")


def pytest_collection_modifyitems(config, items):
    has_data = _LOCAL_DATA_SENTINEL.is_file()
    cuda = None
    for item in items:
        if "local_data" in item.keywords and not has_data:
            item.add_marker(pytest.mark.skip(reason="datasets/ tree not populated; run scripts/data/fetch_reproduction_kit.sh"))
        if "gpu" in item.keywords:
            if cuda is None:
                import torch
                cuda = torch.cuda.is_available()
            if not cuda:
                item.add_marker(pytest.mark.skip(reason="no CUDA device"))
