"""MD 10× thickener: snapshot schedule, steric/basin gates, RMSD/CV dedup.

The expansion must not be another Gaussian jitter of the analytic minima.
These tests pin the contract of that protocol with a fake calculator so they
run without tblite. A real GFN2 run is the same loop with a different engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WATER = _REPO_ROOT / "scripts" / "revision" / "water"
_LIB = _REPO_ROOT / "scripts" / "revision" / "lib"
sys.path.insert(0, str(_LIB))
sys.path.insert(0, str(_WATER))

from generate_water_cluster_geoms import MOTIFS, _valid  # noqa: E402
from md_sampling import (  # noqa: E402
    FakeExplodingCalculator,
    FakeHarmonicCalculator,
    Frame,
    cv_distance,
    cv_vector,
    dedup_frames,
    heavy_atom_rmsd,
    in_enol_basin,
    snapshot_indices,
    steric_ok,
    thicken_one_parent,
)


def _atoms_to_arrays(atoms):
    symbols = [s for s, _ in atoms]
    coords = np.asarray([p for _, p in atoms], dtype=float)
    return symbols, coords


def _water_dimer():
    return _atoms_to_arrays(MOTIFS[2]["dimer"]())


def _water_hexamer(name: str = "cyclic"):
    return _atoms_to_arrays(MOTIFS[6][name]())


def test_snapshot_schedule_n_keep_40_requests_forty_frames() -> None:
    steps = snapshot_indices(n_eq=20, n_keep=40, stride=80)
    assert len(steps) == 40
    assert steps[0] == 100
    assert steps[-1] == 20 + 40 * 80
    assert all(steps[i + 1] - steps[i] == 80 for i in range(39))


def test_thicken_cli_default_n_keep_is_ten_not_forty() -> None:
    import thicken_with_md as thicken_mod

    # The 10× default is what current docs and Phase A use. md40x must pass
    # --n-keep 40 on the CLI rather than changing this default.
    source = Path(thicken_mod.__file__).read_text()
    assert '"--n-keep", type=int, default=10' in source
    assert "water_clusters_md10x" in source
    assert "malonaldehyde_md10x" in source


def test_snapshot_schedule_keeps_ten_after_equilibration() -> None:
    steps = snapshot_indices(n_eq=20, n_keep=10, stride=5)
    assert len(steps) == 10
    assert steps[0] >= 20
    assert steps == [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
    assert all(steps[i + 1] - steps[i] == 5 for i in range(9))


def test_snapshot_schedule_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        snapshot_indices(n_eq=-1, n_keep=10, stride=5)
    with pytest.raises(ValueError):
        snapshot_indices(n_eq=10, n_keep=0, stride=5)
    with pytest.raises(ValueError):
        snapshot_indices(n_eq=10, n_keep=10, stride=0)


def test_steric_accepts_valid_dimer() -> None:
    symbols, coords = _water_dimer()
    assert _valid(list(zip(symbols, [tuple(p) for p in coords])))
    assert steric_ok(symbols, coords, kind="water")


def test_steric_rejects_overlapping_oxygens() -> None:
    symbols, coords = _water_dimer()
    coords = coords.copy()
    # Collapse the two oxygens (indices 0 and 3 in OHH OHH order).
    coords[3] = coords[0] + np.array([1.4, 0.0, 0.0])
    assert not steric_ok(symbols, coords, kind="water")


def test_steric_rejects_exploded_oh_bond() -> None:
    symbols, coords = _water_dimer()
    coords = coords.copy()
    coords[1] = coords[0] + np.array([2.4, 0.0, 0.0])
    assert not steric_ok(symbols, coords, kind="water")


def test_malon_basin_keeps_enol_and_rejects_near_ts() -> None:
    # Donor O at 0, H along +x, acceptor O further along +x.
    # delta = r(O_d-H) - r(H-O_a). Enol: 0.98 - 1.70 = -0.72. TS: 1.20 - 1.20 = 0.
    idx = {"donor": 0, "hydrogen": 1, "acceptor": 2}
    enol = np.array([[0.0, 0.0, 0.0], [0.98, 0.0, 0.0], [2.68, 0.0, 0.0]])
    ts = np.array([[0.0, 0.0, 0.0], [1.20, 0.0, 0.0], [2.40, 0.0, 0.0]])
    assert in_enol_basin(enol, idx, delta_max=-0.40)
    assert not in_enol_basin(ts, idx, delta_max=-0.40)


def test_heavy_atom_rmsd_is_zero_for_rigid_translation() -> None:
    symbols, coords = _water_hexamer()
    shifted = coords + np.array([1.7, -0.4, 0.9])
    assert heavy_atom_rmsd(coords, shifted, symbols) < 1e-8


def test_rmsd_dedup_drops_near_duplicates_keeps_distinct() -> None:
    symbols, coords = _water_hexamer()
    twin = coords + np.random.default_rng(0).normal(0.0, 0.01, size=coords.shape)
    far = coords.copy()
    far[0] = far[0] + np.array([0.8, 0.0, 0.0])
    frames = [
        Frame(symbols, coords, "parent", {"parent": 0}),
        Frame(symbols, twin, "twin", {"parent": 0}),
        Frame(symbols, far, "far", {"parent": 1}),
    ]
    kept, rejected = dedup_frames(frames, rmsd_cutoff=0.15, cv_floor=None)
    comments = {f.comment for f in kept}
    assert "parent" in comments
    assert "twin" not in comments
    assert "far" in comments
    assert any(r["reason"] == "rmsd" for r in rejected)


def test_cv_dedup_drops_near_neighbours_in_cv_space() -> None:
    symbols, coords = _water_hexamer()
    twin = coords + 0.002
    frames = [
        Frame(symbols, coords, "a", {}),
        Frame(symbols, twin, "b", {}),
    ]
    # RMSD cutoff off so only the CV floor can drop the twin.
    kept, rejected = dedup_frames(frames, rmsd_cutoff=0.0, cv_floor=0.02)
    assert len(kept) == 1
    assert rejected and rejected[0]["reason"] == "cv"


def test_cv_vector_is_finite_for_hexamer() -> None:
    symbols, coords = _water_hexamer()
    vec = cv_vector(symbols, coords)
    assert vec.ndim == 1
    assert np.isfinite(vec).all()
    assert cv_distance(symbols, coords, symbols, coords) == pytest.approx(0.0)


def test_langevin_subsample_keeps_ten_frames() -> None:
    symbols, coords = _water_dimer()
    frames, meta = thicken_one_parent(
        symbols,
        coords,
        n_keep=10,
        n_eq=8,
        stride=4,
        timestep_fs=1.0,
        temperature_k=200.0,
        calculator=FakeHarmonicCalculator(k=8.0),
        kind="water",
        seed=3,
    )
    assert len(frames) == 10
    assert meta["n_accepted"] == 10
    assert meta["n_rejected_steric"] == 0
    for frame in frames:
        assert steric_ok(frame.symbols, frame.coords, kind="water")
        # Harmonic well around the parent: intramolecular OH must not explode.
        for i, sym in enumerate(frame.symbols):
            if sym != "O":
                continue
            oh = [
                float(np.linalg.norm(frame.coords[j] - frame.coords[i]))
                for j, t in enumerate(frame.symbols)
                if t == "H"
            ]
            close = [d for d in oh if d < 1.25]
            assert len(close) >= 2


def test_langevin_rejects_exploded_bonds_and_retries() -> None:
    symbols, coords = _water_dimer()
    frames, meta = thicken_one_parent(
        symbols,
        coords,
        n_keep=10,
        n_eq=4,
        stride=2,
        timestep_fs=1.0,
        temperature_k=1500.0,
        calculator=FakeExplodingCalculator(),
        kind="water",
        seed=1,
        max_attempts=3,
    )
    assert frames == []
    assert meta["n_accepted"] == 0
    assert meta["n_rejected_steric"] > 0


def test_malon_train_basin_filter_during_thicken() -> None:
    # Tiny O–H···O fragment standing in for the transferring triad.
    symbols = ["O", "H", "O", "C", "C", "C", "H", "H", "H"]
    coords = np.array(
        [
            [0.00, 0.0, 0.0],
            [0.98, 0.0, 0.0],
            [2.70, 0.0, 0.0],
            [-1.40, 0.0, 0.0],
            [-2.10, 1.2, 0.0],
            [-1.40, 2.4, 0.0],
            [-2.00, -0.8, 0.0],
            [-3.18, 1.2, 0.0],
            [-2.00, 3.2, 0.0],
        ],
        dtype=float,
    )
    idx = {"donor": 0, "hydrogen": 1, "acceptor": 2}
    frames, meta = thicken_one_parent(
        symbols,
        coords,
        n_keep=10,
        n_eq=6,
        stride=3,
        timestep_fs=1.0,
        temperature_k=80.0,
        calculator=FakeHarmonicCalculator(k=20.0),
        kind="malon_train",
        seed=5,
        malon_idx=idx,
        delta_max=-0.40,
    )
    assert len(frames) == 10
    for frame in frames:
        assert in_enol_basin(frame.coords, idx, delta_max=-0.40)
    assert meta["n_rejected_basin"] == 0


def test_ice_thickener_uses_rigid_molecule_samples() -> None:
    symbols, coords = _water_hexamer("prism")
    frames, meta = thicken_one_parent(
        symbols,
        coords,
        n_keep=10,
        kind="ice",
        seed=11,
        temperature_k=150.0,
        calculator=None,
        ice_mode="rigid",
    )
    assert len(frames) == 10
    assert meta["protocol"] == "rigid_thermal"
    for frame in frames:
        assert steric_ok(frame.symbols, frame.coords, kind="ice")


def test_stratified_subset_keeps_every_size_bin() -> None:
    from select_stratified_subset import Frame, stratified_sample

    frames = []
    for n, count in ((6, 20), (12, 10), (18, 8)):
        symbols = (["O", "H", "H"] * (n // 3))
        for i in range(count):
            coords = np.zeros((n, 3), dtype=float)
            coords[:, 0] = np.arange(n) + 0.01 * i
            frames.append(Frame(symbols, coords, f"n={n} i={i}", {}))
    kept = stratified_sample(frames, frac=0.1, seed=1)
    sizes = {len(f.symbols) for f in kept}
    assert sizes == {6, 12, 18}
    assert 3 <= len(kept) <= 8


def test_extra_motifs_are_distinct_from_campaign_table() -> None:
    from generate_water_cluster_geoms import EXTRA_MOTIFS, MOTIFS

    assert 6 in EXTRA_MOTIFS and "boat" in EXTRA_MOTIFS[6]
    for n, extra in EXTRA_MOTIFS.items():
        overlap = set(extra) & set(MOTIFS.get(n, {}))
        assert not overlap, f"EXTRA_MOTIFS[{n}] reuses campaign names {overlap}"


def test_motif_pack_keeps_distinct_minima() -> None:
    from generate_new_motif_parents import keep_distinct_minima, random_pack_waters

    rng = np.random.default_rng(4)
    packed = []
    for seed in range(8):
        symbols, coords = random_pack_waters(6, rng=np.random.default_rng(seed))
        packed.append(Frame(symbols, coords, f"pack{seed}", {"energy": float(seed)}))
    # Duplicate the first packing with a tiny shift and the same energy.
    packed.append(
        Frame(
            packed[0].symbols,
            packed[0].coords + 0.01,
            "dup",
            {"energy": packed[0].meta["energy"]},
        )
    )
    kept = keep_distinct_minima(packed, rmsd_cutoff=0.35, energy_tol=1e-3)
    comments = {f.comment for f in kept}
    assert "dup" not in comments
    assert len(kept) >= 2
    for i, a in enumerate(kept):
        for b in kept[i + 1 :]:
            assert heavy_atom_rmsd(a.coords, b.coords, a.symbols) >= 0.35
