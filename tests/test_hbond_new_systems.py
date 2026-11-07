"""Gate and motif tests for the H-bond new-system campaign."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HBONDS = _REPO_ROOT / "scripts" / "revision" / "hbonds"
sys.path.insert(0, str(_HBONDS))

import collective_variables as cv  # noqa: E402
import overlap_gates as gates  # noqa: E402
from common import nearest_train_rmsds  # noqa: E402
from xyz_io import write_xyz  # noqa: E402


def _nma_bound_frame() -> tuple[list[str], np.ndarray]:
    """Two NMA units with N-H···O=C contacts (~2 Å)."""
    symbols = (
        ["C", "C", "O", "N", "H", "H", "H", "H", "H", "H"]
        + ["C", "C", "O", "N", "H", "H", "H", "H", "H", "H"]
    )
    pos = np.zeros((20, 3))
    # monomer 1 (donor N-H toward monomer 2 O)
    pos[2] = [0.0, 0.0, 0.0]
    pos[3] = [-1.1, 0.0, 0.0]
    pos[4] = [-1.7, 0.85, 0.0]
    # monomer 2
    pos[12] = [2.0, 0.1, 0.0]
    pos[13] = [3.1, 0.1, 0.0]
    pos[14] = [3.7, 0.95, 0.0]
    pos[2] = [1.85, -0.5, 0.0]  # O1 near H14 for reciprocal H-bond
    pos[4] = [-1.7, 0.85, 0.0]
    pos[14] = [3.7, 0.95, 0.0]
    # tune H-bond lengths
    pos[4] = pos[12] + np.array([-0.95, 0.15, 0.0])
    pos[14] = pos[2] + np.array([0.95, 0.15, 0.0])
    return symbols, pos


def _nma_stretched_frame() -> tuple[list[str], np.ndarray]:
    symbols, pos = _nma_bound_frame()
    pos = pos.copy()
    pos[4] = pos[12] + np.array([-2.8, 0.15, 0.0])
    pos[14] = pos[2] + np.array([2.8, 0.15, 0.0])
    return symbols, pos


def _meoh_water_isomer_a() -> tuple[list[str], np.ndarray]:
    symbols = ["C", "O", "H", "H", "H", "O", "H", "H"]
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [-0.5, -0.9, 0.0],
            [-0.5, 0.9, 0.0],
            [-0.9, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 0.9, 0.0],
            [2.0, -0.9, 0.0],
        ],
        dtype=float,
    )
    pos[5] = pos[1] + np.array([1.8, 0.0, 0.0])
    pos[2] = pos[5] + np.array([-0.95, 0.25, 0.0])
    return symbols, pos


def _meoh_water_isomer_b() -> tuple[list[str], np.ndarray]:
    symbols = ["C", "O", "H", "H", "H", "O", "H", "H"]
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [1.7, 0.95, 0.0],
            [-0.5, -0.9, 0.0],
            [-0.5, 0.9, 0.0],
            [3.2, 0.0, 0.0],
            [1.35, 0.25, 0.0],
            [3.9, -0.8, 0.0],
        ],
        dtype=float,
    )
    return symbols, pos


def _acac_enol_frame(delta: float = -0.8) -> tuple[list[str], np.ndarray]:
    symbols = ["C", "C", "O", "C", "C", "O", "C", "H", "H", "H", "H", "H", "H", "H", "H"]
    pos = np.full((15, 3), 10.0)
    pos[2] = [0.0, 0.0, 0.0]
    pos[5] = [2.0, 0.0, 0.0]
    t = 0.5 + 0.2 * delta
    pos[8] = pos[2] + t * (pos[5] - pos[2])
    return symbols, pos


def test_nma_two_short_r_ho_contacts():
    r_ho, theta, _ = cv.nma_hbond_contacts(*_nma_bound_frame())
    assert r_ho < 2.2, r_ho
    assert theta > 100.0, theta


def test_nma_stretch_increases_mean_r_ho():
    train = [_nma_bound_frame(), _nma_bound_frame()]
    ood = [_nma_stretched_frame()]
    result = gates.check_nma_stretch(train, ood)
    assert result.passed


def test_meoh_water_donor_gate_rejects_isomer_b():
    train = [_meoh_water_isomer_a()]
    assert gates.check_meoh_donor_gate(train).passed
    assert not gates.check_meoh_donor_gate([_meoh_water_isomer_b()]).passed


def test_acac_train_delta_and_scan_crosses_zero():
    train = [_acac_enol_frame(-0.8)]
    ood = [_acac_enol_frame(+0.8)]
    idx = cv.proton_transfer_indices(train[0][0], train[0][1])
    result = gates.check_acac_basin(train, ood, idx)
    assert result.passed


def test_paracetamol_min_rmsd_gap(monkeypatch):
    syms = ["C", "O", "H1", "H2", "H3"]
    train = [(syms, np.zeros((5, 3)))]
    val = [(syms, np.ones((5, 3)) * 0.5)]
    id_test = [(syms, np.ones((5, 3)) * 0.6)]
    ood = [(syms, np.ones((5, 3)) * 2.0)]

    def fake_rmsds(train_frames, test_frames):
        del train_frames
        out = []
        for _, pos in test_frames:
            if np.allclose(pos, 2.0):
                out.append(2.5)
            elif np.allclose(pos, 0.5):
                out.append(0.5)
            else:
                out.append(0.6)
        return np.asarray(out, dtype=float)

    monkeypatch.setattr(gates, "nearest_train_rmsds", fake_rmsds)
    result = gates.check_paracetamol_gaps(train, val, id_test, ood)
    assert result.passed


def test_pgg_family_ids_disjoint():
    manifest = {"train_family_ids": [1, 2, 3], "ood_family_ids": [4]}
    assert gates.check_pgg_families(manifest).passed


def test_pgg_train_is_forced_onto_large_orca_tier(implementation_text):
    """30-atom PGG misses LARGE_ATOM_CUTOFF=33 and times out in 4 h."""
    text = implementation_text(_HBONDS / "launch_hbond_campaign.sh")
    assert "ALWAYS_LARGE_SYSTEMS=${ALWAYS_LARGE_SYSTEMS:-pgg,crambin_frags}" in text
    assert "small_sbatch=scripts/revision/submit_water_orca.sbatch" in text
    assert 'small_sbatch=scripts/revision/submit_water_orca_large.sbatch' in text
    assert 'submit_tier "${SYSTEM}_${split}" "$small_sbatch" "$small_conc"' in text


def test_crambin_fragment_constraints():
    manifest = {
        "fragments": {"helix": {"natoms": 52}, "loop": {"natoms": 61}},
        "train_region_ids": ["helix", "loop"],
        "ood_region_ids": ["disulfide"],
        "full_protein_train_written": False,
    }
    assert gates.check_crambin_fragments(manifest, ood_frames=[("C", np.zeros((52, 3)))]).passed


def test_crambin_rejects_full_protein_train_flag():
    manifest = {
        "fragments": {"helix": {"natoms": 52}},
        "train_region_ids": ["helix"],
        "ood_region_ids": ["loop"],
        "full_protein_train_written": True,
    }
    assert not gates.check_crambin_fragments(manifest, ood_frames=[]).passed


def test_no_full_protein_xyz_writer_in_crambin_generator(implementation_text):
    source = implementation_text(_HBONDS / "generate_crambin_frags.py")
    assert "full_protein_train_written" in source
    assert "raise RuntimeError" in source or "refuse" in source.lower()


def test_nma_accept_bound_rejects_interpenetrating_monomers():
    from generate_nma_dimer import accept_bound

    symbols = (
        ["C", "C", "O", "N", "C", "H"]
        + ["C", "C", "O", "N", "C", "H"]
    )
    pos = np.zeros((12, 3))
    # monomer 1
    pos[0] = [-3.0, 0.0, 0.0]
    pos[1] = [-1.5, 0.0, 0.0]
    pos[2] = [-1.5, 1.2, 0.0]
    pos[3] = [-0.4, -0.9, 0.0]
    pos[4] = [-0.4, -2.4, 0.0]
    pos[5] = [0.5, -0.5, 0.0]
    # monomer 2, N-H···O=C contact ~1.9 Å
    pos[6] = [3.5, 0.2, 0.0]
    pos[7] = [2.0, 0.2, 0.0]
    pos[8] = [2.0, -1.0, 0.0]
    pos[9] = [0.9, 1.1, 0.0]
    pos[10] = [0.9, 2.6, 0.0]
    pos[11] = [0.0, 0.7, 0.0]
    assert accept_bound(pos, symbols)
    fused = pos.copy()
    fused[6:] -= np.array([3.2, 0.0, 0.0])
    assert not accept_bound(fused, symbols)


def test_fixture_xyz_roundtrip(tmp_path):
    frames = [_meoh_water_isomer_a(), _meoh_water_isomer_b()]
    path = tmp_path / "dimer.xyz"
    write_xyz(path, frames, ["a", "b"])
    from xyz_io import read_xyz

    loaded = read_xyz(path)
    assert len(loaded) == 2


def test_rmsd_gate_on_synthetic_shift():
    syms = ["C", "O", "H", "H", "H", "O", "H", "H"]
    train = [(syms, np.array([[0, 0, 0], [1.4, 0, 0], [0.5, 0.9, 0], [0.5, -0.9, 0], [2.0, 0.1, 0], [3.2, 0, 0], [3.9, 0.7, 0], [2.6, -0.8, 0]], dtype=float))]
    ood = [(syms, np.array([[5.0, 4.0, 3.0], [6.4, 4.2, 3.1], [5.6, 5.0, 3.0], [5.2, 3.1, 3.5], [6.8, 4.3, 3.2], [7.5, 4.0, 3.0], [8.2, 4.8, 3.1], [7.0, 3.0, 3.4]], dtype=float))]
    result = gates.check_rmsd_gate(train, ood)
    assert result.passed
