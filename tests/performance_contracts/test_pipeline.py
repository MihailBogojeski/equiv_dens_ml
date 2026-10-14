"""Dataset-free actual DenSNet composition, with isolated immutable reference.

No timing assertions. Override DENSNET_REFERENCE_ROOT only with an intentionally
qualified immutable source checkout. Each arm is imported in a new subprocess.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from benchmarks.fixtures import load_fixture, tensor_hash

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(os.environ.get("DENSNET_REFERENCE_ROOT", str(ROOT.parent / "equiv_dens_ml-paper-reference")))


def run_arm(source, *arguments):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-m", "benchmarks.pipeline", "--source-root", str(source), *map(str, arguments)]
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def test_saved_fixture_detects_modified_weights(tmp_path):
    path = tmp_path / "fixture.pt"
    run_arm(REFERENCE, "generate", "--output", path)
    fixture = torch.load(path, weights_only=False, map_location="cpu")
    key = next(key for key, tensor in fixture["state_dict"].items() if tensor.is_floating_point() and tensor.numel())
    fixture["state_dict"][key].view(-1)[0] += 1.0
    torch.save(fixture, path)
    with pytest.raises(ValueError, match="parameter hash mismatch"):
        load_fixture(path)
