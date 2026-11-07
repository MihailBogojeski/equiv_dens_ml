"""Nested, size-stratified water train subsets for data-efficiency curves.

Raw --num_train on the full npy is a seeded random draw and can be all dimers
at N=50. These tests pin the contract: equal n_water counts, and N=50 subset
of N=125 subset of the rest, so the curve is nested, not five independent draws.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WATER = _REPO_ROOT / "scripts" / "revision" / "water"
sys.path.insert(0, str(_WATER))

from make_learning_curve_splits import (  # noqa: E402
    DEFAULT_PER_STRATUM,
    DEFAULT_SIZES,
    n_water_per_frame,
    nested_stratified_indices,
    slice_dataset,
    write_learning_curve_splits,
)


def _toy_atom_numbers(n_per_size: int = 4) -> np.ndarray:
    """Five cluster sizes (n=2..6), padded to 18 atoms. Last slots are 0."""
    frames = []
    for n_water in (2, 3, 4, 5, 6):
        for _ in range(n_per_size):
            z = np.zeros(18, dtype=int)
            z[: 3 * n_water] = np.tile([8, 1, 1], n_water)
            frames.append(z)
    return np.stack(frames)


def test_n_water_counts_oxygens_not_padded_slots() -> None:
    z = _toy_atom_numbers(n_per_size=1)
    counts = n_water_per_frame(z)
    assert counts.tolist() == [2, 3, 4, 5, 6]


def test_nested_indices_are_balanced_and_nested() -> None:
    n_water = n_water_per_frame(_toy_atom_numbers(n_per_size=4))
    sizes = (5, 10, 20)
    per_stratum = {5: 1, 10: 2, 20: 4}
    splits = nested_stratified_indices(n_water, sizes=sizes, seed=7, per_stratum=per_stratum)

    assert set(splits) == set(sizes)
    smaller = None
    for n in sizes:
        idx = splits[n]
        assert len(idx) == n
        counts = n_water[np.asarray(idx)]
        assert sorted(counts.tolist()) == sorted([2, 3, 4, 5, 6] * per_stratum[n])
        if smaller is not None:
            assert set(smaller) <= set(idx)
        smaller = idx


def test_nested_indices_are_deterministic() -> None:
    n_water = n_water_per_frame(_toy_atom_numbers())
    a = nested_stratified_indices(n_water, sizes=(5, 10), seed=21, per_stratum={5: 1, 10: 2})
    b = nested_stratified_indices(n_water, sizes=(5, 10), seed=21, per_stratum={5: 1, 10: 2})
    assert a == b


def test_default_sizes_match_plan_per_stratum() -> None:
    assert DEFAULT_SIZES == (50, 125, 250, 500, 1250)
    assert DEFAULT_PER_STRATUM == {50: 10, 125: 25, 250: 50, 500: 100, 1250: 250}


def test_slice_dataset_keeps_frame_aligned_arrays() -> None:
    n = 6
    base = {
        "positions": np.arange(n * 3 * 3, dtype=float).reshape(n, 3, 3),
        "atom_numbers": np.tile(np.array([8, 1, 1], dtype=int), (n, 1)),
        "source_index": np.arange(n, dtype=int),
        "atom_types": np.array([1, 8], dtype=int),
    }
    dens = np.array([{"i": i} for i in range(n)], dtype=object)
    sliced_base, sliced_dens = slice_dataset(base, dens, [1, 4])
    assert sliced_base["source_index"].tolist() == [1, 4]
    assert [row["i"] for row in sliced_dens] == [1, 4]
    assert sliced_base["atom_types"].tolist() == [1, 8]
    np.testing.assert_array_equal(sliced_base["positions"], base["positions"][[1, 4]])


def test_write_learning_curve_splits_writes_nested_npy(tmp_path: Path) -> None:
    z = _toy_atom_numbers(n_per_size=4)
    n = z.shape[0]
    base = {
        "positions": np.zeros((n, 18, 3), dtype=float),
        "atom_numbers": z,
        "source_index": np.arange(n, dtype=int),
    }
    dens = np.array([{"i": i} for i in range(n)], dtype=object)
    base_path = tmp_path / "water_train_pbe_d4_avdz_base.npy"
    dens_path = tmp_path / "water_train_pbe_d4_avdz.npy"
    np.save(base_path, base, allow_pickle=True)
    np.save(dens_path, dens, allow_pickle=True)

    out = tmp_path / "learning_curves"
    manifest = write_learning_curve_splits(
        base_path,
        dens_path,
        out,
        sizes=(5, 10, 20),
        per_stratum={5: 1, 10: 2, 20: 4},
        seed=7,
        theory="pbe_d4_avdz",
    )
    assert manifest["sizes"] == [5, 10, 20]
    payload = json.loads((out / "splits.json").read_text())
    assert set(payload["indices"]["5"]) <= set(payload["indices"]["10"])

    small = np.load(out / "water_train_n5_pbe_d4_avdz_base.npy", allow_pickle=True).item()
    assert small["atom_numbers"].shape[0] == 5
    assert n_water_per_frame(small["atom_numbers"]).tolist() == pytest.approx(
        sorted(n_water_per_frame(small["atom_numbers"]).tolist())
    )
    assert sorted(n_water_per_frame(small["atom_numbers"]).tolist()) == [2, 3, 4, 5, 6]
