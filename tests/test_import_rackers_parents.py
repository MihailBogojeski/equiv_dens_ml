"""Rackers/Rakshit water-cluster XYZ import: splits, dedup, hold-out."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WATER = _REPO_ROOT / "scripts" / "revision" / "water"
_LIB = _REPO_ROOT / "scripts" / "revision" / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_WATER))

from generate_water_cluster_geoms import MOTIFS  # noqa: E402
from import_rackers_parents import (  # noqa: E402
    assign_split,
    import_rackers_xyz,
    n_waters,
    write_imported_trees,
)
from md_sampling import Frame  # noqa: E402


def _xyz_frame(n_water: int, shift: float = 0.0) -> tuple[list[str], np.ndarray]:
    symbols = ["O", "H", "H"] * n_water
    coords = np.zeros((3 * n_water, 3), dtype=float)
    for i in range(n_water):
        coords[3 * i] = (i * 2.8 + shift, 0.0, 0.0)
        coords[3 * i + 1] = (i * 2.8 + shift, 0.76, 0.58)
        coords[3 * i + 2] = (i * 2.8 + shift, -0.76, 0.58)
    return symbols, coords


def _write_multi_xyz(path: Path, frames: list[tuple[int, float]]) -> None:
    lines = []
    for n_water, shift in frames:
        symbols, coords = _xyz_frame(n_water, shift)
        lines.append(str(len(symbols)))
        lines.append(f"rackers n={n_water}")
        for sym, xyz in zip(symbols, coords):
            lines.append(f"{sym} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    path.write_text("\n".join(lines) + "\n")


def test_n_waters_and_split_assignment() -> None:
    assert n_waters(["O", "H", "H"] * 6) == 6
    assert assign_split(6) == "train"
    assert assign_split(8) == "ood_size"
    assert assign_split(12) == "ood_size"
    assert assign_split(15) == "ood_rackers"
    assert assign_split(30) == "ood_rackers"


def test_import_routes_n6_and_n30_and_holds_out_rackers(tmp_path: Path) -> None:
    src = tmp_path / "rackers.xyz"
    _write_multi_xyz(src, [(6, 0.0), (30, 0.0)])
    out_water = tmp_path / "water"
    out_ood = tmp_path / "ood"
    manifest = write_imported_trees(src, out_water, out_ood, existing=[], rmsd_cutoff=0.20)
    assert (out_water / "train.xyz").is_file()
    assert (out_ood / "ood_rackers.xyz").is_file()
    ood_text = (out_ood / "ood_rackers.xyz").read_text()
    assert "n=30" in ood_text
    assert "n=6" not in ood_text
    assert manifest["splits"]["train"]["n_frames"] == 1
    assert manifest["splits"]["ood_rackers"]["n_frames"] == 1
    assert "6" in manifest["splits"]["train"]["by_n"]
    assert "30" in manifest["splits"]["ood_rackers"]["by_n"]
    # Held-out split must not appear as a train file.
    assert not (out_water / "ood_rackers.xyz").exists()


def test_import_drops_rmsd_duplicate_of_existing_parent(tmp_path: Path) -> None:
    symbols, coords = _atoms_from_motif()
    existing = [Frame(symbols, coords, "campaign dimer", {})]
    src = tmp_path / "dup.xyz"
    lines = [str(len(symbols)), "dup of campaign"]
    for sym, xyz in zip(symbols, coords):
        lines.append(f"{sym} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}")
    src.write_text("\n".join(lines) + "\n")
    kept, rejected = import_rackers_xyz(src, existing=existing, rmsd_cutoff=0.20)
    assert kept == []
    assert len(rejected) == 1


def _atoms_from_motif():
    atoms = MOTIFS[2]["dimer"]()
    symbols = [s for s, _ in atoms]
    coords = np.asarray([p for _, p in atoms], dtype=float)
    return symbols, coords
