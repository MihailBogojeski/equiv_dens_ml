"""OMol25-style water distance-scaling scan."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "lib"))
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "water"))

import generate_ood_distance_scale as scan  # noqa: E402
import rigid_scale as rs  # noqa: E402


def _write_dimer(path: Path, r_oo: float = 2.8) -> Path:
    roh = 0.9572
    half = math.radians(104.52) / 2.0
    waters = []
    for sign in (1.0, -1.0):
        com = np.array([sign * r_oo / 2.0, 0.0, 0.0])
        local = np.array(
            [
                [0.0, 0.0, 0.0],
                [roh * math.sin(half), 0.0, roh * math.cos(half)],
                [-roh * math.sin(half), 0.0, roh * math.cos(half)],
            ]
        )
        waters.append(local + com)
    symbols, coords = rs.molecules_to_water_frame(waters)
    lines = [f"{len(symbols)}", "dimer parent"]
    for s, xyz in zip(symbols, coords):
        lines.append(f"{s} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_scan_writes_twenty_two_frames_and_tags_clashes(tmp_path: Path):
    parent = _write_dimer(tmp_path / "dimer.xyz")
    out_dir = tmp_path / "ood_scale"
    result = scan.build_distance_scale_split(
        [parent],
        out_dir,
        parent_labels=["n2_dimer"],
        scale_factors=[0.15, 1.05, 2.0, 3.0],
    )
    assert result["n_frames"] == 4
    manifest = json.loads(Path(result["manifest"]).read_text())
    rec = manifest["records"]
    assert len(rec) == 4
    assert all("lambda" in r and "r_oo_min" in r and "regime" in r for r in rec)
    crushed = next(r for r in rec if abs(r["lambda"] - 0.15) < 1e-12)
    assert crushed["clash"]
    assert not next(r for r in rec if abs(r["lambda"] - 2.0) < 1e-12)["clash"]
    doubled = next(r for r in rec if abs(r["lambda"] - 2.0) < 1e-12)
    assert abs(doubled["r_oo_min"] - 5.6) < 1e-6
    assert doubled["regime"] == "SR"
    far = next(r for r in rec if abs(r["lambda"] - 3.0) < 1e-12)
    assert far["regime"] == "LR"
    assert Path(result["xyz"]).is_file()


def test_default_omol25_grid_is_twenty_two_lambdas_per_parent(tmp_path: Path):
    parent = _write_dimer(tmp_path / "dimer.xyz")
    result = scan.build_distance_scale_split([parent], tmp_path / "grid", parent_labels=["n2"])
    assert result["n_frames"] == 22
    rec = json.loads(Path(result["manifest"]).read_text())["records"]
    for row in rec:
        assert abs(row["r_oo_min"] - 2.8 * row["lambda"]) < 1e-6
