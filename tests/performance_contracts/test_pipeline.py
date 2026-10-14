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


@pytest.mark.parametrize("case,dtype,fixed", [
    ("tiny", "float64", False),
    ("batch", "float64", False),
    ("tiny", "float64", True),
    ("tiny", "float32", False),
    ("wide", "float32", False),
])
def test_full_pipeline_reference_contracts(tmp_path, case, dtype, fixed):
    assert (REFERENCE / "src/equiv_dens/nn/dft_network.py").is_file(), "Immutable reference is required for this numerical contract"
    fixture_path, reference_path, comparison_path = [tmp_path / name for name in ("fixture.pt", "reference.pt", "comparison.json")]
    extra = ["--fixed-radial"] if fixed else []
    run_arm(REFERENCE, "generate", "--case", case, "--dtype", dtype, *extra, "--output", fixture_path)
    # FP32 finite differences are intentionally excluded: P1 specifies tiny
    # FP64 derivative calibration; FP32 gates compare the exact two arms.
    fd_options = [] if dtype == "float64" else ["--skip-finite-differences"]
    run_arm(REFERENCE, "snapshot", "--fixture", fixture_path, "--output", reference_path, *fd_options)
    run_arm(ROOT, "compare", "--fixture", fixture_path, "--reference", reference_path, "--output", comparison_path, *fd_options)
    comparison = json.loads(comparison_path.read_text())
    assert comparison["passed"], comparison["failures"]
    fixture = load_fixture(fixture_path)
    assert fixture["state_hash"] == tensor_hash(fixture["state_dict"])
    reference = torch.load(reference_path, weights_only=False, map_location="cpu")
    assert reference["state_hash"] == fixture["state_hash"]
    for key, contract in reference["contracts"].items():
        assert contract["grid_calls"] == int(key.startswith("density/"))
    for mode in ("coefficients", "density", "ef"):
        if fixed and mode == "ef":
            continue
        train = reference["contracts"][f"{mode}/train"]
        evaluation = reference["contracts"][f"{mode}/eval"]
        for field in ("df_coeffs", "density", "density_grad", "energy", "forces"):
            if field in train["outputs"]:
                torch.testing.assert_close(train["outputs"][field], evaluation["outputs"][field], rtol=0, atol=0)
        if mode != "ef":
            assert train["changed_parameters"]
            assert all(name.startswith("density_repr_model.") for name in train["changed_parameters"])
    if fixed:
        assert set(reference["exclusions"]) == {"ef", "force_training"}
        assert all(case["exception"] == "IndexError" for case in reference["exclusions"].values())
    else:
        training = reference["contracts"]["force_training/train"]
        assert training["forces_graph"]
        density_gradients = [gradient for name, gradient in training["gradients"].items() if name.startswith("density_repr_model.")]
        assert density_gradients and all(gradient is None for gradient in density_gradients)
        energy_gradients = [gradient for name, gradient in training["gradients"].items() if name.startswith("property_models.energy.") and gradient is not None]
        assert max(float(gradient.abs().max()) for gradient in energy_gradients) > 1e-12
        assert training["changed_parameters"]
        assert all(name.startswith("property_models.energy.") for name in training["changed_parameters"])
    if dtype == "float64":
        assert reference["finite_differences"]["spatial_fd_pass"], reference["finite_differences"]
        assert comparison["candidate_finite_differences"]["spatial_fd_pass"]
        if not fixed:
            assert reference["finite_differences"]["force_fd_pass"], reference["finite_differences"]
            assert reference["finite_differences"]["force_loss_mixed_fd_pass"], reference["finite_differences"]
            assert comparison["candidate_finite_differences"]["force_fd_pass"]
            assert comparison["candidate_finite_differences"]["force_loss_mixed_fd_pass"]
    for source_name, expected_root in (("reference_provenance", REFERENCE), ("candidate_provenance", ROOT)):
        imported = comparison[source_name]["imports"]
        assert imported
        assert all(Path(entry["path"]).is_relative_to(expected_root) for entry in imported.values())


def test_saved_fixture_detects_modified_weights(tmp_path):
    path = tmp_path / "fixture.pt"
    run_arm(REFERENCE, "generate", "--output", path)
    fixture = torch.load(path, weights_only=False, map_location="cpu")
    key = next(key for key, tensor in fixture["state_dict"].items() if tensor.is_floating_point() and tensor.numel())
    fixture["state_dict"][key].view(-1)[0] += 1.0
    torch.save(fixture, path)
    with pytest.raises(ValueError, match="parameter hash mismatch"):
        load_fixture(path)
