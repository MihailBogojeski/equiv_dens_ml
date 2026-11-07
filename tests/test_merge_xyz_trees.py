"""Merge geometry trees and write by_n manifests without clobbering comments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts/revision/lib"))
sys.path.insert(0, str(_REPO_ROOT / "scripts/revision/water"))

from merge_xyz_trees import merge_trees, write_xyz_comment  # noqa: E402


def _one_water_xyz(path: Path, comment: str, x: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "3\n"
        f"{comment}\n"
        f"O {x:.3f} 0.000 0.000\n"
        f"H {x + 0.76:.3f} 0.580 0.000\n"
        f"H {x - 0.76:.3f} 0.580 0.000\n"
    )


def test_merge_preserves_comments_and_writes_by_n(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    dest = tmp_path / "merged"
    _one_water_xyz(a / "train.xyz", "parent md40x n=1", 0.0)
    _one_water_xyz(b / "train.xyz", "parent newmotifs n=1", 5.0)
    manifest = merge_trees([a, b], dest, splits=("train",))
    text = (dest / "train.xyz").read_text()
    assert "parent md40x n=1" in text
    assert "parent newmotifs n=1" in text
    rec = manifest["splits"]["train"]
    assert rec["n_frames"] == 2
    assert rec["by_n"] == {"1": 2}
    on_disk = json.loads((dest / "manifest.json").read_text())
    assert on_disk["splits"]["train"]["n_frames"] == 2


def test_merge_dedup_drops_identical_geometry(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    dest = tmp_path / "merged"
    _one_water_xyz(a / "train.xyz", "first", 0.0)
    _one_water_xyz(b / "train.xyz", "copy", 0.0)
    manifest = merge_trees([a, b], dest, splits=("train",), rmsd_cutoff=0.05)
    assert manifest["splits"]["train"]["n_frames"] == 1
    assert "copy" not in (dest / "train.xyz").read_text()


def test_write_xyz_comment_roundtrip() -> None:
    assert write_xyz_comment("water n=6 boat") == "water n=6 boat"
