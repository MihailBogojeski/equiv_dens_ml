"""The separation numbers the OOD claim rests on.

Reviewer 1's objection is that the manuscript asserts test structures are out of
distribution without showing it, so this report *is* the answer to that
objection. Two things therefore have to hold: a number that says "separated"
must not be reachable by choosing a favourable projection, and the control split
-- which is in distribution by construction -- must come out overlapping. A bug
that made everything look separated would be invisible in the output, since
every tier is expected to be separated and the only thing contradicting it would
be the control.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts/revision"))

import ood_overlap_report as rep  # noqa: E402


def write_cvs(path: Path, label: str, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"label": label, "frames": rows}))
    return path


def frames(n: int, *, q: float, density: float, hbond: float = 3.0, seed: int = 0) -> list[dict]:
    """`n` frames scattered tightly around one point in CV space."""
    rng = np.random.default_rng(seed)
    return [
        {
            "q_tetrahedral": float(q + rng.normal(0, 0.01)),
            "n_hbond": float(hbond + rng.normal(0, 0.02)),
            "ring_6_frac": float(rng.uniform(0.0, 0.05)),
            "local_density": float(density + rng.normal(0, 0.01)),
            "surface_frac": float(rng.uniform(0.3, 0.4)),
            "mean_nn_oo": float(2.8 + rng.normal(0, 0.02)),
        }
        for _ in range(n)
    ]


def run(tmp_path: Path, train_rows, test_rows, *, label="tier", extra=()) -> dict:
    train = write_cvs(tmp_path / "train.json", "train", train_rows)
    test = write_cvs(tmp_path / "test.json", label, test_rows)
    out = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts/revision/ood_overlap_report.py"),
         "--train", str(train), "--test", str(test), "--out", str(out), *extra],
        check=True,
        capture_output=True,
        cwd=_REPO_ROOT,
    )
    return json.loads(out.read_text())


def test_both_default_planes_are_reported(tmp_path):
    report = run(tmp_path, frames(60, q=0.5, density=0.9), frames(40, q=0.9, density=0.9, seed=1))
    assert report["joint_cvs"] == [["q_tetrahedral", "local_density"], ["n_hbond", "local_density"]]
    per_plane = report["splits"]["tier"]["bhattacharyya_joint_per_plane"]
    assert set(per_plane) == {"q_tetrahedral,local_density", "n_hbond,local_density"}


def test_verdict_takes_the_worst_plane_not_the_kindest(tmp_path):
    """A split separated in order but not in density is not a separated split.

    This is the whole point of measuring more than one plane: the ice tier is
    designed to move tetrahedral order and leaves density nearly untouched, so a
    report quoting only the order plane would call it separated on the strength
    of the one axis it was built to move.
    """
    report = run(
        tmp_path,
        frames(60, q=0.30, density=0.90, hbond=3.0),
        frames(40, q=0.95, density=0.90, hbond=3.0, seed=1),
    )
    split = report["splits"]["tier"]
    per_plane = split["bhattacharyya_joint_per_plane"]
    assert per_plane["q_tetrahedral,local_density"] < per_plane["n_hbond,local_density"]
    assert split["bhattacharyya_joint"] == pytest.approx(max(per_plane.values()))
    assert split["verdict_joint_below_threshold"] is False


def test_a_split_separated_in_every_plane_passes(tmp_path):
    report = run(
        tmp_path,
        frames(60, q=0.30, density=0.90, hbond=3.0),
        frames(40, q=0.95, density=0.20, hbond=1.0, seed=1),
    )
    split = report["splits"]["tier"]
    assert split["bhattacharyya_joint"] < 0.05
    assert split["verdict_joint_below_threshold"] is True


def test_the_in_distribution_control_reads_as_overlapping(tmp_path):
    """If this ever passes as separated, every other number in the file is noise."""
    report = run(
        tmp_path,
        frames(80, q=0.5, density=0.9, seed=0),
        frames(60, q=0.5, density=0.9, seed=99),
        label="id_test",
    )
    split = report["splits"]["id_test"]
    assert split["bhattacharyya_joint"] > 0.8
    assert split["verdict_joint_below_threshold"] is False
    # Measured against the *median* spacing inside the training set, so a split
    # drawn from the training distribution should land either side of it about
    # equally. Near 0 would mean the control is being read as far away.
    assert 0.3 < split["nn_distance"]["frac_within_train_scale"] < 0.7


def test_an_explicit_plane_overrides_the_defaults(tmp_path):
    """Malonaldehyde uses its own coordinates and must not get the water planes."""
    report = run(
        tmp_path,
        frames(60, q=0.5, density=0.9),
        frames(40, q=0.9, density=0.5, seed=1),
        extra=("--joint", "q_tetrahedral,n_hbond"),
    )
    assert report["joint_cvs"] == [["q_tetrahedral", "n_hbond"]]
    assert set(report["splits"]["tier"]["bhattacharyya_joint_per_plane"]) == {"q_tetrahedral,n_hbond"}


def test_a_plane_naming_an_unknown_cv_is_dropped_rather_than_crashing(tmp_path):
    report = run(
        tmp_path,
        frames(60, q=0.5, density=0.9),
        frames(40, q=0.9, density=0.5, seed=1),
        extra=("--joint", "q_tetrahedral,not_a_cv", "--joint", "q_tetrahedral,local_density"),
    )
    assert report["joint_cvs"] == [["q_tetrahedral", "local_density"]]


def test_identical_samples_overlap_completely():
    a = np.random.default_rng(0).normal(size=(200, 2))
    assert rep.bhattacharyya_2d(a, a.copy()) == pytest.approx(1.0, abs=0.02)


def test_disjoint_samples_do_not_overlap():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 2))
    b = rng.normal(size=(200, 2)) + 50.0
    assert rep.bhattacharyya_2d(a, b) < 1e-6
