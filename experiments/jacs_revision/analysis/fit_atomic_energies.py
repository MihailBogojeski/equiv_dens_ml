#!/usr/bin/env python3
"""Fit per-element energies so mixed-N labels stay extensive.

A single global mean (``--center_energy`` without ``--atomic_energies``) makes
``E - mean(E)`` a molecule-level constant. An extensive head cannot represent
that across cluster sizes. Least-squares ``e_Z`` on the training compositions
is the same ``--atomic_energies`` path the paper used for polythiophene, with
a baseline matched to the actual DFT totals.

Padding columns with ``Z=0`` are ignored. ``e_0`` is written as 0.0 so older
callers that still index the pad stay safe.

Units follow the ``*_base.npy`` file (Hartree for the revision water/QM7x sets).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def composition_counts(atom_numbers: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return per-frame element counts and the Z>0 element list."""
    nuclear_charges = np.asarray(atom_numbers)
    if nuclear_charges.ndim != 2:
        raise ValueError(
            f"atom_numbers must be (n_frames, n_pad), got {nuclear_charges.shape}"
        )
    elements = sorted({int(z) for z in np.unique(nuclear_charges) if int(z) > 0})
    if not elements:
        raise ValueError("atom_numbers contains no Z>0 atoms")
    counts = np.stack(
        [(nuclear_charges == element).sum(axis=1) for element in elements],
        axis=1,
    ).astype(np.float64)
    return counts, elements


def fit_atomic_energies(
    atom_numbers: np.ndarray,
    energies: np.ndarray,
    ridge: float = 1e-6,
) -> dict[int, float]:
    """Ridge least-squares ``e_Z`` for ``E ≈ sum_I e_{Z_I}``.

    Water is rank-deficient (always 2 H per O). Ridge picks one solution in
    that null space; the residual ``E - n_water * e_water`` is unique.
    """
    counts, elements = composition_counts(atom_numbers)
    targets = np.asarray(energies, dtype=np.float64).reshape(-1)
    if counts.shape[0] != targets.shape[0]:
        raise ValueError(
            f"frame count mismatch: atom_numbers {counts.shape[0]} vs "
            f"energies {targets.shape[0]}"
        )
    gram = counts.T @ counts + ridge * np.eye(len(elements))
    coefficients = np.linalg.solve(gram, counts.T @ targets)
    table = {0: 0.0}
    table.update(
        {element: float(coef) for element, coef in zip(elements, coefficients)}
    )
    return table


def residual_mae(atom_numbers: np.ndarray, energies: np.ndarray, table: dict[int, float]) -> float:
    counts, elements = composition_counts(atom_numbers)
    baseline = np.zeros(counts.shape[0], dtype=np.float64)
    for index, element in enumerate(elements):
        baseline += table[element] * counts[:, index]
    return float(np.mean(np.abs(np.asarray(energies, dtype=np.float64) - baseline)))


def load_base(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    data = payload.item() if getattr(payload, "shape", None) == () else payload
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a dict-like atoms file")
    if "atom_numbers" not in data or "energy" not in data:
        raise KeyError(f"{path} needs atom_numbers and energy, has {list(data)}")
    return np.asarray(data["atom_numbers"]), np.asarray(data["energy"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("np_dataset", type=Path, help="Training *_base.npy")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output dict npy with integer Z keys (Hartree)",
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    args = parser.parse_args()

    atom_numbers, energies = load_base(args.np_dataset)
    table = fit_atomic_energies(atom_numbers, energies, ridge=args.ridge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, table)

    mae = residual_mae(atom_numbers, energies, table)
    print(f"wrote {args.output}")
    print(f"elements { {z: table[z] for z in sorted(table)} }")
    print(f"residual MAE {mae:.6f} Eh ({mae * 627.5094740631:.3f} kcal/mol)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
