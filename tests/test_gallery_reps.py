"""Gallery display frames must be chemically intact."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts" / "revision" / "analysis"))

from render_dataset_structures import Frame, heavy_min_distance, read_xyz_frames


def test_builder_writes_clash_free_nma_acac_pgg(tmp_path, monkeypatch):
    import build_gallery_reps as reps

    monkeypatch.setattr(reps, "OUT", tmp_path)
    nma = reps.nma_dimer()
    acac = reps.acac_enol()
    pgg = reps.pgg_tripeptide()
    for path in (nma, acac, pgg):
        frame = read_xyz_frames(path)[0]
        assert heavy_min_distance(frame) >= 1.15
    acac_frame = read_xyz_frames(acac)[0]
    o_idx = [i for i, s in enumerate(acac_frame.symbols) if s == "O"]
    oo = np.linalg.norm(
        np.array(acac_frame.coords[o_idx[0]]) - np.array(acac_frame.coords[o_idx[1]])
    )
    assert oo <= 2.80
    nma_frame = read_xyz_frames(nma)[0]
    xyz = np.array(nma_frame.coords)
    s = nma_frame.symbols
    h1 = [i for i, a in enumerate(s[:12]) if a != "H"]
    h2 = [i + 12 for i, a in enumerate(s[12:]) if a != "H"]
    inter = min(
        float(np.linalg.norm(xyz[i] - xyz[j]))
        for i in h1
        for j in h2
    )
    assert inter >= 1.70
