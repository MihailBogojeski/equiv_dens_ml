"""Shared MD-thickening primitives for the geometry expansion.

The original campaign placed frames by Gaussian jitter around analytic water
minima (and a single GFN2 snapshot per ice/droplet/malon parent). Reviewer-
facing sampling is now a short Langevin trajectory per parent, subsampled after
a brief equilibration, with the same steric and malonaldehyde-basin gates the
generators already use.

GFN2-xTB (tblite) is the default engine because malonaldehyde and droplet
melting already used it; a fake harmonic calculator is injected by the unit
tests. Large ice/droplet parents (n ≥ 16) may fall back to ASE TIP4P.
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/water/md_sampling.py'


import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)

import numpy as np

from experiments.jacs_revision.water.generate_ood_water import (  # noqa: E402
    molecules_are_valid,
    split_molecules,
    thermal_displace,
)
from experiments.jacs_revision.water.generate_water_cluster_geoms import _valid  # noqa: E402

MALON_TRAIN_DELTA_MAX = -0.40
HEAVY_SYMBOLS = frozenset({"C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})

# CV components that exist for every water cluster size used here. q-tetrahedral
# is NaN for n=2–4 (no 3- or 4-coordinate oxygens), so it is not a dedup axis.
CV_DEDUP_KEYS = ("n_hbond", "r_gyration", "surface_frac", "mean_nn_oo", "n_water")


@dataclass
class Frame:
    """One geometry plus the bookkeeping the manifest needs."""

    symbols: list[str]
    coords: np.ndarray
    comment: str
    meta: dict[str, Any] = field(default_factory=dict)


def snapshot_indices(n_eq: int, n_keep: int, stride: int) -> list[int]:
    """Absolute Langevin step numbers of the kept snapshots.

    Step ``k`` is the configuration after ``k`` integrator steps. Equilibration
    occupies ``[1, n_eq]``; the first kept frame is ``n_eq + stride``.
    """
    if n_eq < 0 or n_keep < 1 or stride < 1:
        raise ValueError(f"need n_eq>=0, n_keep>=1, stride>=1; got {n_eq}, {n_keep}, {stride}")
    return [n_eq + (i + 1) * stride for i in range(n_keep)]


def _pairs_as_atoms(symbols: list[str], coords: np.ndarray) -> list[tuple[str, tuple[float, float, float]]]:
    return [(s, (float(p[0]), float(p[1]), float(p[2]))) for s, p in zip(symbols, coords)]


def steric_ok(symbols: list[str], coords: np.ndarray, kind: str = "water") -> bool:
    """Reject unphysical contacts. ``kind`` selects the existing generator gate."""
    coords = np.asarray(coords, dtype=float)
    if kind in {"water", "droplet"}:
        return bool(_valid(_pairs_as_atoms(symbols, coords)))
    if kind in {"ice"}:
        if set(symbols) <= {"O", "H"} and symbols[0] == "O" and len(symbols) % 3 == 0:
            return bool(molecules_are_valid(split_molecules(symbols, coords)))
        return bool(_valid(_pairs_as_atoms(symbols, coords)))
    if kind.startswith("malon"):
        return _malon_steric_ok(symbols, coords)
    return bool(_valid(_pairs_as_atoms(symbols, coords)))


def _malon_steric_ok(symbols: list[str], coords: np.ndarray) -> bool:
    """No crushed atoms; the transferring proton stays bound to some oxygen."""
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            if dist < 0.55:
                return False
    oxygens = [i for i, s in enumerate(symbols) if s == "O"]
    hydrogens = [i for i, s in enumerate(symbols) if s == "H"]
    if not oxygens or not hydrogens:
        return True
    # At least one H remains within a hydrogen-bond length of an oxygen.
    bound = False
    for h in hydrogens:
        if min(float(np.linalg.norm(coords[h] - coords[o])) for o in oxygens) < 1.8:
            bound = True
            break
    return bound


def malon_delta(positions: np.ndarray, idx: dict[str, int]) -> float:
    """δ = r(O_donor–H) − r(H–O_acceptor), in Angstrom."""
    r_donor = float(np.linalg.norm(positions[idx["hydrogen"]] - positions[idx["donor"]]))
    r_acceptor = float(np.linalg.norm(positions[idx["hydrogen"]] - positions[idx["acceptor"]]))
    return r_donor - r_acceptor


def in_enol_basin(positions: np.ndarray, idx: dict[str, int], delta_max: float = MALON_TRAIN_DELTA_MAX) -> bool:
    """True when the proton is still in the training enol well."""
    return malon_delta(positions, idx) <= delta_max


def heavy_atom_rmsd(coords_a: np.ndarray, coords_b: np.ndarray, symbols: list[str]) -> float:
    """Kabsch RMSD over C/N/O/… after removing the centroid. Hydrogens ignored."""
    a_all = np.asarray(coords_a, dtype=float)
    b_all = np.asarray(coords_b, dtype=float)
    if a_all.shape != b_all.shape:
        raise ValueError(f"RMSD needs matching shapes, got {a_all.shape} vs {b_all.shape}")
    mask = np.array([s in HEAVY_SYMBOLS for s in symbols], dtype=bool)
    if mask.shape[0] != a_all.shape[0] or int(mask.sum()) < 2:
        mask = np.ones(a_all.shape[0], dtype=bool)
    a = a_all[mask]
    b = b_all[mask]
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    covariance = a.T @ b
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    rotated = a @ rotation
    return float(np.sqrt(((rotated - b) ** 2).sum() / len(a)))


def _atomic_numbers(symbols: list[str]) -> np.ndarray:
    from ase.data import atomic_numbers

    return np.array([atomic_numbers[s] for s in symbols], dtype=int)


def cv_vector(symbols: list[str], coords: np.ndarray) -> np.ndarray:
    """Finite collective-variable vector used for split-internal dedup."""
    from experiments.jacs_revision.water.water_collective_variables import frame_cvs

    cvs = frame_cvs(_atomic_numbers(symbols), np.asarray(coords, dtype=float))
    values = []
    for key in CV_DEDUP_KEYS:
        value = cvs.get(key, float("nan"))
        if value is None or not np.isfinite(value):
            continue
        values.append(float(value))
    if not values:
        return np.zeros(1, dtype=float)
    return np.asarray(values, dtype=float)


def cv_distance(
    symbols_a: list[str],
    coords_a: np.ndarray,
    symbols_b: list[str],
    coords_b: np.ndarray,
) -> float:
    """Euclidean distance in the finite CV subspace shared by both frames."""
    va = cv_vector(symbols_a, coords_a)
    vb = cv_vector(symbols_b, coords_b)
    n = min(len(va), len(vb))
    if n == 0:
        return 0.0
    return float(np.linalg.norm(va[:n] - vb[:n]))


def dedup_frames(
    frames: Iterable[Frame],
    rmsd_cutoff: float,
    cv_floor: float | None,
    existing: Iterable[Frame] | None = None,
) -> tuple[list[Frame], list[dict[str, Any]]]:
    """Drop a frame if it is too close in heavy-atom RMSD or in CV space.

    ``rmsd_cutoff <= 0`` disables the RMSD gate. ``cv_floor is None`` disables
    the CV gate. ``existing`` is the already-accepted set (Phase B vs Phase A).
    """
    kept: list[Frame] = list(existing or [])
    n_prior = len(kept)
    rejected: list[dict[str, Any]] = []
    for frame in frames:
        drop_reason = None
        drop_value = None
        for other in kept:
            if len(other.symbols) != len(frame.symbols):
                continue
            if rmsd_cutoff > 0.0:
                rmsd = heavy_atom_rmsd(frame.coords, other.coords, frame.symbols)
                if rmsd < rmsd_cutoff:
                    drop_reason = "rmsd"
                    drop_value = rmsd
                    break
            if cv_floor is not None:
                dist = cv_distance(frame.symbols, frame.coords, other.symbols, other.coords)
                if dist < cv_floor:
                    drop_reason = "cv"
                    drop_value = dist
                    break
        if drop_reason is not None:
            rejected.append(
                {
                    "comment": frame.comment,
                    "reason": drop_reason,
                    "value": drop_value,
                    "meta": frame.meta,
                }
            )
            continue
        kept.append(frame)
    return kept[n_prior:], rejected


class FakeHarmonicCalculator:
    """ASE-shaped calculator: each atom is sprung to its first-seen position.

    Used by the unit tests so a 10× Langevin subsample can be asserted without
    tblite. Production runs inject GFN2 or TIP4P instead.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, k: float = 5.0) -> None:
        self.k = float(k)
        self.ref: np.ndarray | None = None
        self.results: dict[str, Any] = {}

    def calculate(self, atoms=None, properties=("energy",), system_changes=None) -> None:
        del properties, system_changes
        if atoms is None:
            raise ValueError("FakeHarmonicCalculator.calculate needs atoms")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if self.ref is None:
            self.ref = positions.copy()
        delta = positions - self.ref
        energy = 0.5 * self.k * float((delta**2).sum())
        forces = -self.k * delta
        self.results = {"energy": energy, "forces": forces}

    def get_potential_energy(self, atoms=None, **kwargs) -> float:
        del kwargs
        if atoms is not None:
            self.calculate(atoms)
        return float(self.results["energy"])

    def get_forces(self, atoms=None, **kwargs) -> np.ndarray:
        del kwargs
        if atoms is not None:
            self.calculate(atoms)
        return np.asarray(self.results["forces"], dtype=float)


class FakeExplodingCalculator:
    """Pushes every hydrogen away from the nearest oxygen. Tests the reject path."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, strength: float = 8.0) -> None:
        self.strength = float(strength)
        self.results: dict[str, Any] = {}

    def calculate(self, atoms=None, properties=("energy",), system_changes=None) -> None:
        del properties, system_changes
        if atoms is None:
            raise ValueError("FakeExplodingCalculator.calculate needs atoms")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        symbols = atoms.get_chemical_symbols()
        forces = np.zeros_like(positions)
        oxygens = [i for i, s in enumerate(symbols) if s == "O"]
        for i, symbol in enumerate(symbols):
            if symbol != "H" or not oxygens:
                continue
            nearest = min(oxygens, key=lambda o: float(np.linalg.norm(positions[i] - positions[o])))
            vec = positions[i] - positions[nearest]
            norm = float(np.linalg.norm(vec)) or 1.0
            forces[i] = self.strength * vec / norm
            forces[nearest] -= forces[i]
        energy = -self.strength * float(np.linalg.norm(forces))
        self.results = {"energy": energy, "forces": forces}

    def get_potential_energy(self, atoms=None, **kwargs) -> float:
        del kwargs
        if atoms is not None:
            self.calculate(atoms)
        return float(self.results["energy"])

    def get_forces(self, atoms=None, **kwargs) -> np.ndarray:
        del kwargs
        if atoms is not None:
            self.calculate(atoms)
        return np.asarray(self.results["forces"], dtype=float)


def attach_calculator(atoms, calculator) -> None:
    """Bind an ASE or ASE-shaped calculator, wrapping fakes that lack ``atoms``."""
    if calculator is None:
        return
    if hasattr(calculator, "implemented_properties") and not hasattr(calculator, "parameters"):
        from ase.calculators.calculator import Calculator, all_changes

        inner = calculator

        class _Wrapper(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                Calculator.calculate(self, atoms, properties, system_changes)
                inner.calculate(self.atoms)
                self.results = {
                    "energy": float(inner.results["energy"]),
                    "forces": np.asarray(inner.results["forces"], dtype=float),
                }

        atoms.calc = _Wrapper()
        return
    atoms.calc = calculator


def run_langevin_snapshots(
    symbols: list[str],
    coords: np.ndarray,
    *,
    calculator,
    n_eq: int,
    n_keep: int,
    stride: int,
    timestep_fs: float,
    temperature_k: float,
    seed: int,
    friction: float = 0.02,
) -> list[tuple[list[str], np.ndarray]]:
    """Equilibrate, then keep ``n_keep`` snapshots spaced by ``stride`` steps."""
    from ase import Atoms, units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    work = Atoms(symbols=list(symbols), positions=np.asarray(coords, dtype=float))
    attach_calculator(work, calculator)
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(work, temperature_K=temperature_k, rng=rng)
    dyn = Langevin(
        work,
        timestep_fs * units.fs,
        temperature_K=temperature_k,
        friction=friction,
        rng=rng,
    )
    kept: list[tuple[list[str], np.ndarray]] = []
    targets = set(snapshot_indices(n_eq, n_keep, stride))
    n_total = n_eq + n_keep * stride
    for step in range(1, n_total + 1):
        dyn.run(1)
        if step in targets:
            kept.append((list(work.get_chemical_symbols()), work.get_positions().copy()))
    return kept


def _thicken_ice_rigid(
    symbols: list[str],
    coords: np.ndarray,
    *,
    n_keep: int,
    temperature_k: float,
    seed: int,
) -> list[Frame]:
    mols = split_molecules(list(symbols), np.asarray(coords, dtype=float))
    rng = np.random.default_rng(seed)
    frames: list[Frame] = []
    attempts = 0
    while len(frames) < n_keep and attempts < n_keep * 40:
        attempts += 1
        trial = thermal_displace(mols, temperature_k, rng)
        if not molecules_are_valid(trial):
            continue
        from experiments.jacs_revision.water.generate_ood_water import molecules_to_frame

        sym, xyz = molecules_to_frame(trial)
        frames.append(
            Frame(
                list(sym),
                np.asarray(xyz, dtype=float),
                f"ice-rigid k={len(frames)}",
                {"protocol": "rigid_thermal", "md_step": attempts, "temperature_k": temperature_k},
            )
        )
    return frames


def thicken_one_parent(
    symbols: list[str],
    coords: np.ndarray,
    *,
    n_keep: int = 10,
    n_eq: int = 50,
    stride: int = 50,
    timestep_fs: float = 1.0,
    temperature_k: float = 300.0,
    calculator=None,
    kind: str = "water",
    seed: int = 0,
    delta_max: float = MALON_TRAIN_DELTA_MAX,
    malon_idx: dict[str, int] | None = None,
    max_attempts: int = 5,
    ice_mode: str = "md",
) -> tuple[list[Frame], dict[str, Any]]:
    """Take up to ``n_keep`` accepted snapshots from one parent geometry.

    Returns the frames and a meta record (accepted / rejected counts, protocol).
    Ice ``ice_mode='rigid'`` skips the calculator and uses the existing
    rigid-molecule thermal displace, which keeps the oxygen network crystalline.
    """
    meta: dict[str, Any] = {
        "n_accepted": 0,
        "n_rejected_steric": 0,
        "n_rejected_basin": 0,
        "protocol": "langevin",
        "temperature_k": temperature_k,
        "n_eq": n_eq,
        "stride": stride,
        "kind": kind,
        "seed": seed,
    }
    if kind == "ice" and (ice_mode == "rigid" or calculator is None):
        frames = _thicken_ice_rigid(
            symbols, coords, n_keep=n_keep, temperature_k=temperature_k, seed=seed
        )
        meta["protocol"] = "rigid_thermal"
        meta["n_accepted"] = len(frames)
        return frames, meta

    accepted: list[Frame] = []
    for attempt in range(max_attempts):
        if calculator is None:
            raise RuntimeError("thicken_one_parent needs a calculator unless ice_mode='rigid'")
        snapshots = run_langevin_snapshots(
            symbols,
            coords,
            calculator=calculator,
            n_eq=n_eq,
            n_keep=n_keep,
            stride=stride,
            timestep_fs=timestep_fs,
            temperature_k=temperature_k,
            seed=seed + 17 * attempt,
        )
        for snap_i, (sym, xyz) in enumerate(snapshots):
            if not steric_ok(sym, xyz, kind=kind if kind != "ice" else "ice"):
                meta["n_rejected_steric"] += 1
                continue
            if kind == "malon_train":
                if malon_idx is None:
                    raise ValueError("malon_train thickening needs malon_idx")
                if not in_enol_basin(xyz, malon_idx, delta_max=delta_max):
                    meta["n_rejected_basin"] += 1
                    continue
            step = snapshot_indices(n_eq, n_keep, stride)[snap_i]
            accepted.append(
                Frame(
                    list(sym),
                    np.asarray(xyz, dtype=float),
                    f"{kind} parent-md step={step}",
                    {
                        "md_step": step,
                        "attempt": attempt,
                        "temperature_k": temperature_k,
                    },
                )
            )
            if len(accepted) >= n_keep:
                break
        if len(accepted) >= n_keep:
            break
        # Fake exploding (and a rare GFN2 blow-up) must not reuse a calculator
        # that has already walked far from the parent.
        if hasattr(calculator, "ref"):
            calculator.ref = None

    accepted = accepted[:n_keep]
    meta["n_accepted"] = len(accepted)
    return accepted, meta


def read_xyz_symbol_frames(path: Path) -> list[tuple[list[str], np.ndarray, str]]:
    """Multi-frame XYZ → (symbols, coords, comment)."""
    lines = Path(path).read_text().splitlines()
    frames: list[tuple[list[str], np.ndarray, str]] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        nat = int(lines[i].strip())
        comment = lines[i + 1].strip() if i + 1 < len(lines) else ""
        symbols: list[str] = []
        coords: list[list[float]] = []
        for row in lines[i + 2 : i + 2 + nat]:
            parts = row.split()
            symbols.append(parts[0])
            coords.append([float(v) for v in parts[1:4]])
        frames.append((symbols, np.asarray(coords, dtype=float), comment))
        i += 2 + nat
    return frames


def write_xyz_frames(path: Path, frames: list[Frame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for frame in frames:
            handle.write(f"{len(frame.symbols)}\n")
            handle.write(f"{frame.comment}\n")
            for symbol, xyz in zip(frame.symbols, frame.coords):
                handle.write(f"{symbol:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}\n")


def infer_kind(split: str, n_atoms: int, symbols: list[str]) -> str:
    """Map an XYZ split name onto the steric / basin protocol."""
    if split in {"train", "val"} and "C" in symbols:
        return "malon_train"
    if split == "ood_proton_transfer":
        return "malon_scan"
    if split == "ood_order":
        return "ice"
    if split == "ood_density":
        return "droplet"
    return "water"


def protocol_for(kind: str, n_atoms: int) -> dict[str, Any]:
    """Default Langevin lengths: short for n=2–6, longer for ice/droplets."""
    n_water = n_atoms // 3
    if kind == "ice":
        return {"n_eq": 80, "stride": 80, "timestep_fs": 1.0, "temperature_k": 150.0, "ice_mode": "md"}
    if kind == "droplet":
        return {"n_eq": 100, "stride": 100, "timestep_fs": 1.0, "temperature_k": 320.0, "ice_mode": "md"}
    if kind.startswith("malon"):
        return {"n_eq": 80, "stride": 50, "timestep_fs": 0.5, "temperature_k": 300.0, "ice_mode": "md"}
    if n_water <= 6:
        return {"n_eq": 50, "stride": 50, "timestep_fs": 1.0, "temperature_k": 300.0, "ice_mode": "md"}
    return {"n_eq": 80, "stride": 80, "timestep_fs": 1.0, "temperature_k": 300.0, "ice_mode": "md"}


def build_gfn2_calculator():
    from tblite.ase import TBLite

    return TBLite(method="GFN2-xTB", verbosity=0)


def build_tip4p_calculator():
    from ase.calculators.tip4p import TIP4P

    return TIP4P()


def choose_calculator(name: str, kind: str, n_atoms: int):
    """``auto``: GFN2 for small waters and malon; TIP4P for n ≥ 16 ice/droplets."""
    if name == "fake":
        return FakeHarmonicCalculator(k=8.0)
    if name == "none":
        return None
    if name == "gfn2":
        return build_gfn2_calculator()
    if name == "tip4p":
        return build_tip4p_calculator()
    if name != "auto":
        raise ValueError(f"unknown calculator {name!r}")
    n_water = n_atoms // 3
    if kind.startswith("malon"):
        return build_gfn2_calculator()
    if kind in {"ice", "droplet"} and n_water >= 16:
        try:
            return build_tip4p_calculator()
        except Exception:
            return build_gfn2_calculator()
    return build_gfn2_calculator()
