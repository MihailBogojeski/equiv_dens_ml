"""Collective variables for the H-bond new-system campaign."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.jacs_revision.hbonds.common import angle_deg, distance


def _indices_by_symbol(symbols: list[str], element: str) -> list[int]:
    return [i for i, s in enumerate(symbols) if s == element]


def nma_hbond_contacts(symbols: list[str], positions: np.ndarray) -> tuple[float, float, float]:
    """Mean r(H···O), mean theta(N–H···O), and delta_hb for the NMA dimer.

    Each monomer has one N–H and one C=O. The cyclic dimer has two H-bonds.
    """
    nitrogens = _indices_by_symbol(symbols, "N")
    oxygens = _indices_by_symbol(symbols, "O")
    hydrogens = _indices_by_symbol(symbols, "H")
    if len(nitrogens) != 2 or len(oxygens) != 2:
        raise ValueError("expected two N and two O in NMA dimer")

    contacts: list[tuple[float, float]] = []
    for n in nitrogens:
        nh_candidates = [h for h in hydrogens if distance(positions, n, h) < 1.2]
        if not nh_candidates:
            continue
        best_pair = None
        for h in nh_candidates:
            o = min(oxygens, key=lambda oxy: distance(positions, h, oxy))
            r_ho = distance(positions, h, o)
            if best_pair is None or r_ho < best_pair[0]:
                best_pair = (r_ho, h, o)
        if best_pair is None:
            continue
        r_ho, h, o = best_pair
        theta = angle_deg(positions, n, h, o)
        contacts.append((r_ho, theta))
    if not contacts:
        return float("nan"), float("nan"), float("nan")
    r_vals = [c[0] for c in contacts]
    t_vals = [c[1] for c in contacts]
    return float(np.mean(r_vals)), float(np.mean(t_vals)), float(np.mean(r_vals) - 1.0)


def meoh_water_donor_is_methanol(positions: np.ndarray, symbols: list[str]) -> bool:
    """True when the hydroxyl H of methanol points at water (isomer A)."""
    carbons = _indices_by_symbol(symbols, "C")
    oxygens = _indices_by_symbol(symbols, "O")
    hydrogens = _indices_by_symbol(symbols, "H")
    if len(carbons) != 1 or len(oxygens) != 2:
        raise ValueError("expected one C and two O for MeOH–water")
    meoh_o = max(oxygens, key=lambda o: sum(distance(positions, o, c) < 1.5 for c in carbons))
    water_o = oxygens[0] if oxygens[1] == meoh_o else oxygens[1]
    meoh_h = None
    for h in hydrogens:
        if distance(positions, meoh_o, h) < 1.2:
            meoh_h = h
            break
    if meoh_h is None:
        return False
    r_meoh = distance(positions, meoh_h, water_o)
    r_water = min(distance(positions, h, meoh_o) for h in hydrogens if h != meoh_h)
    return r_meoh < r_water


def meoh_water_cv(positions: np.ndarray, symbols: list[str]) -> dict[str, float]:
    """Donor identity and H-bond geometry for the methanol–water dimer."""
    carbons = _indices_by_symbol(symbols, "C")
    oxygens = _indices_by_symbol(symbols, "O")
    hydrogens = _indices_by_symbol(symbols, "H")
    meoh_o = max(oxygens, key=lambda o: sum(distance(positions, o, c) < 1.5 for c in carbons))
    water_o = oxygens[0] if oxygens[1] == meoh_o else oxygens[1]
    donor_is_meoh = meoh_water_donor_is_methanol(positions, symbols)
    if donor_is_meoh:
        donor_o, acceptor_o = meoh_o, water_o
        donor_h = next(h for h in hydrogens if distance(positions, donor_o, h) < 1.2)
    else:
        donor_o, acceptor_o = water_o, meoh_o
        donor_h = min(
            (h for h in hydrogens if distance(positions, donor_o, h) < 1.2),
            key=lambda h: distance(positions, donor_o, h),
        )
    r_oo = distance(positions, meoh_o, water_o)
    r_ho = distance(positions, donor_h, acceptor_o)
    r_dh = distance(positions, donor_o, donor_h)
    theta = angle_deg(positions, donor_o, donor_h, acceptor_o)
    return {
        "donor_is_meoh": float(donor_is_meoh),
        "r_oo": r_oo,
        "r_ho": r_ho,
        "delta": r_dh - r_ho,
        "theta_dha": theta,
    }


def proton_transfer_indices(symbols: list[str], positions: np.ndarray) -> dict[str, int]:
    """Locate donor O, transferring H, and acceptor O from connectivity."""
    oxygens = _indices_by_symbol(symbols, "O")
    hydrogens = _indices_by_symbol(symbols, "H")
    if len(oxygens) != 2:
        raise ValueError(f"expected 2 oxygens, found {len(oxygens)}")
    best = None
    for h in hydrogens:
        for o in oxygens:
            dist = distance(positions, h, o)
            if dist < 1.2 and (best is None or dist < best[0]):
                best = (dist, h, o)
    if best is None:
        raise ValueError("no hydroxyl hydrogen found")
    _, h_idx, donor = best
    acceptor = oxygens[0] if oxygens[1] == donor else oxygens[1]
    return {"donor": donor, "hydrogen": h_idx, "acceptor": acceptor}


def delta_coordinate(positions: np.ndarray, idx: dict[str, int]) -> float:
    r_donor = distance(positions, idx["hydrogen"], idx["donor"])
    r_acceptor = distance(positions, idx["hydrogen"], idx["acceptor"])
    return r_donor - r_acceptor


def compute_frame_cvs(
    system: str,
    symbols: list[str],
    positions: np.ndarray,
    *,
    pt_indices: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return CV dictionary for one frame."""
    if system == "nma_dimer":
        r_ho, theta, delta_hb = nma_hbond_contacts(symbols, positions)
        return {"r_ho": r_ho, "theta_nho": theta, "delta_hb": delta_hb}
    if system == "meoh_water":
        return meoh_water_cv(positions, symbols)
    if system in ("acac", "malonaldehyde"):
        idx = pt_indices if pt_indices is not None else proton_transfer_indices(symbols, positions)
        return {"delta": delta_coordinate(positions, idx)}
    if system == "paracetamol":
        return {"rg": float(np.sqrt(np.mean(np.sum((positions - positions.mean(0)) ** 2, axis=1))))}
    if system == "pgg":
        return {"rg": float(np.sqrt(np.mean(np.sum((positions - positions.mean(0)) ** 2, axis=1))))}
    if system == "crambin_frags":
        return {"rg": float(np.sqrt(np.mean(np.sum((positions - positions.mean(0)) ** 2, axis=1))))}
    raise ValueError(f"unknown system {system!r}")


def cv_names_for_system(system: str) -> tuple[str, ...]:
    """Primary CV names used for overlap reporting."""
    mapping = {
        "nma_dimer": ("r_ho",),
        "meoh_water": ("donor_is_meoh", "r_ho"),
        "acac": ("delta",),
        "paracetamol": ("rg",),
        "pgg": ("rg",),
        "crambin_frags": ("rg",),
    }
    return mapping[system]
