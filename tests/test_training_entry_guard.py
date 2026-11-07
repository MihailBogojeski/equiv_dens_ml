"""Spawn imports must not initialize W&B, open data, or start a trainer."""
import multiprocessing as mp
import runpy
from pathlib import Path
import pytest


def _worker(path):
    # A spawned child loads its parent's script as __mp_main__.
    runpy.run_path(path, run_name="__mp_main__")


@pytest.mark.parametrize("name", ["train_all.py", "train_only_en.py"])
def test_training_script_is_safe_to_spawn(name):
    path = str(Path(__file__).resolve().parents[1] / "scripts" / "training" / name)
    child = mp.get_context("spawn").Process(target=_worker, args=(path,))
    child.start()
    child.join(timeout=60)  # import of torch + schnetpack alone takes ~15 s on a loaded node
    if child.is_alive():
        child.terminate()
        child.join()
        pytest.fail("training script performed work in a spawned worker")
    assert child.exitcode == 0
