#!/usr/bin/env python3
"""Push normal modes to extreme Cartesian amplitudes.

GFN2-xTB supplies the Hessian directions. Frames are written at a
requested RMS Cartesian displacement (the 10 A rung is the O(1) nm
geometry). Stretched or clashing frames are tagged, not dropped.

Usage:
  python scripts/revision/analysis/generate_extreme_modes.py --system ethanol
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/generate_extreme_modes.py'


import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR, _SCRIPT_DIR.parent / "water")
from ase import Atoms  # noqa: E402
from ase.data import atomic_numbers, chemical_symbols  # noqa: E402
from ase.optimize import BFGS  # noqa: E402
from ase.vibrations import Vibrations  # noqa: E402
from experiments.jacs_revision.water.generate_ood_distance_scale import read_xyz_symbol_frames  # noqa: E402
from experiments.jacs_revision.water.generate_ood_water import write_xyz  # noqa: E402
from experiments.jacs_revision.lib.normal_modes import (  # noqa: E402
    displace_along_mode,
    displacement_metrics,
    drop_rigid_modes,
    equilibrium_bond_pairs,
    project_modes,
    validity_tags,
)
from tblite.ase import TBLite  # noqa: E402

_REPO_ROOT = repo_root()

SMALL_RMSD = (0.5, 1.0, 2.0, 5.0, 10.0)
OLIGOMER_RMSD = (1.0, 5.0, 10.0)

SYSTEMS = {
    "ethanol": {
        "parent": "datasets/ethanol_train_10.xyz",
        "n_modes": None,
        "rmsd": SMALL_RMSD,
    },
    "ethanethiol": {
        "parent": "datasets/ethanethiol_combo_Aidx-1000.npy",
        "n_modes": None,
        "rmsd": SMALL_RMSD,
    },
    "resorcinol": {
        "parent": "datasets/resorcinol_combo_kmeansidx-1000_train.npy",
        "n_modes": None,
        "rmsd": SMALL_RMSD,
    },
    "thiophene_2mer": {
        "parent": "datasets/thiophene2mer_init.npy",
        "n_modes": None,
        "rmsd": SMALL_RMSD,
    },
    "thiophene_12mer": {
        "parent": "datasets/thiophene12mer_init.npy",
        "n_modes": 100,
        "rmsd": OLIGOMER_RMSD,
    },
    "water_n2": {
        "parent": "datasets/revision/water_clusters/minima/n2_dimer.xyz",
        "n_modes": None,
        "rmsd": SMALL_RMSD,
    },
    "water_n6": {
        "parent": "datasets/revision/water_clusters/minima/n6_prism.xyz",
        "n_modes": 50,
        "rmsd": SMALL_RMSD,
    },
    "malonaldehyde": {
        "parent": "datasets/revision/malonaldehyde/train.xyz",
        "n_modes": None,
        "rmsd": SMALL_RMSD,
    },
}


def load_first_geometry(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return symbols, positions (N,3), atomic numbers for the first usable frame."""
    path = Path(path)
    if path.suffix.lower() == ".xyz":
        symbols, coords, _ = read_xyz_symbol_frames(path)[0]
        numbers = np.array([atomic_numbers[s] for s in symbols], dtype=int)
        return symbols, np.asarray(coords, dtype=float), numbers

    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item() if data.ndim == 0 else data
    if not isinstance(data, dict):
        raise ValueError(f"unsupported npy in {path}")
    pos = np.asarray(data["positions"], dtype=float)
    z = np.asarray(data["atom_numbers"])
    if pos.ndim == 2:
        pos = pos[None, ...]
    if z.ndim == 1:
        z = np.broadcast_to(z, (pos.shape[0], len(z)))
    frame_z = np.asarray(z[0])
    mask = frame_z > 0
    numbers = frame_z[mask].astype(int)
    coords = pos[0, mask]
    symbols = [chemical_symbols[int(n)] for n in numbers]
    return symbols, np.asarray(coords, dtype=float), numbers


def optimize_and_hessian(symbols: list[str], positions: np.ndarray, workdir: Path, fmax: float = 0.05):
    """GFN2-xTB local minimum and Cartesian Hessian modes."""
    workdir.mkdir(parents=True, exist_ok=True)
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)
    BFGS(atoms, logfile=str(workdir / "opt.log")).run(fmax=fmax)
    vib = Vibrations(atoms, name=str(workdir / "vib"))
    vib.run()
    data = vib.get_vibrations()
    freqs = np.real(np.asarray(data.get_frequencies(), dtype=complex))
    modes = data.get_modes()
    evecs = np.stack([np.asarray(m, dtype=float).reshape(-1) for m in modes], axis=1)
    masses = atoms.get_masses()
    evals, evecs = drop_rigid_modes(np.abs(freqs), evecs, n_drop=6)
    freq_kept = np.abs(freqs)[np.argsort(np.abs(freqs))][6:]
    return atoms.get_positions().copy(), masses, freq_kept, evecs


def build_mode_scan(
    symbols: list[str],
    positions: np.ndarray,
    masses: np.ndarray,
    evecs: np.ndarray,
    frequencies: np.ndarray,
    rmsd_ladder: tuple[float, ...] | list[float],
    n_modes: int | None = None,
    signs: tuple[int, ...] = (-1, 1),
) -> tuple[list[tuple[list[str], np.ndarray]], list[dict]]:
    if n_modes is None:
        n_modes = evecs.shape[1]
    evecs = np.asarray(evecs, dtype=float)[:, :n_modes]
    frequencies = np.asarray(frequencies, dtype=float)[:n_modes]
    bonds = equilibrium_bond_pairs(positions)
    frames: list[tuple[list[str], np.ndarray]] = []
    records: list[dict] = []
    for k in range(evecs.shape[1]):
        mode = evecs[:, k].reshape(len(positions), 3)
        for sign in signs:
            for rmsd in rmsd_ladder:
                xyz = displace_along_mode(positions, mode, sign * float(rmsd))
                q = project_modes(xyz - positions, evecs, masses)
                tags = validity_tags(symbols, xyz, positions, bonds)
                metrics = displacement_metrics(xyz, positions)
                frames.append((list(symbols), xyz))
                records.append(
                    {
                        "mode_index": int(k),
                        "frequency_cm1": float(frequencies[k]),
                        "sign": int(sign),
                        "rmsd_target": float(abs(rmsd)),
                        "q_k": float(q[k]),
                        **metrics,
                        **tags,
                    }
                )
    return frames, records


def write_mode_bundle(
    out_dir: Path,
    molecule: str,
    symbols: list[str],
    equilibrium: np.ndarray,
    masses: np.ndarray,
    frequencies: np.ndarray,
    evecs: np.ndarray,
    frames: list[tuple[list[str], np.ndarray]],
    records: list[dict],
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    comments = [
        (
            f"extreme_mode mol={molecule} k={r['mode_index']} sign={r['sign']:+d} "
            f"rmsd={r['rmsd_target']:.3f}"
        )
        for r in records
    ]
    xyz_path = out_dir / "ood_modes.xyz"
    write_xyz(xyz_path, frames, comments)
    np.savez(
        out_dir / "modes.npz",
        equilibrium=np.asarray(equilibrium, dtype=float),
        modes=np.asarray(evecs, dtype=float),
        masses=np.asarray(masses, dtype=float),
        frequencies=np.asarray(frequencies, dtype=float),
        symbols=np.asarray(symbols),
    )
    manifest = {
        "molecule": molecule,
        "xyz": str(xyz_path),
        "modes": str(out_dir / "modes.npz"),
        "n_frames": len(frames),
        "n_modes": int(evecs.shape[1]),
        "n_atoms": int(len(equilibrium)),
        "records": records,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["manifest"] = str(manifest_path)
    return manifest


def run_system(name: str, out_root: Path, work_root: Path) -> dict:
    spec = SYSTEMS[name]
    parent = _REPO_ROOT / spec["parent"]
    symbols, positions, _numbers = load_first_geometry(parent)
    workdir = work_root / name
    eq, masses, freqs, evecs = optimize_and_hessian(symbols, positions, workdir)
    n_keep = spec["n_modes"] if spec["n_modes"] is not None else evecs.shape[1]
    evecs = evecs[:, :n_keep]
    freqs = freqs[:n_keep]
    frames, records = build_mode_scan(
        symbols,
        eq,
        masses,
        evecs,
        freqs,
        spec["rmsd"],
        n_modes=n_keep,
    )
    return write_mode_bundle(out_root / name, name, symbols, eq, masses, freqs, evecs, frames, records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", action="append", choices=sorted(SYSTEMS), dest="systems")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "datasets/revision/ood_modes",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_REPO_ROOT / "results/revision/extreme_modes_work",
    )
    args = parser.parse_args()
    names = list(SYSTEMS) if args.all else (args.systems or [])
    if not names:
        raise SystemExit("pass --system NAME or --all")
    summaries = []
    for name in names:
        print(f"extreme modes: {name}", flush=True)
        manifest = run_system(name, args.output_dir, args.work_dir)
        summaries.append({k: manifest[k] for k in ("molecule", "n_frames", "n_modes", "xyz")})
        print(json.dumps(summaries[-1], indent=2), flush=True)
    print(json.dumps({"systems": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
