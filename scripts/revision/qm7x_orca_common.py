#!/usr/bin/env python3
"""Shared QM7-X ORCA helpers: inputs, parsers, and DensNet calc_dict conversion.

ORCA labels (energy, forces, MOs, DF coefficients) are produced at one
consistent level of theory:

    ! PBE0 aug-cc-pVDZ TightSCF EnGrad

Official QM7-X PBE0+MBD values are kept only as metadata. Training uses the
ORCA numbers. Forces follow scripts/revision/generate_dft_labels.py:
Hartree/Bohr gradient -> Hartree/Angstrom via ``-grad / ase.units.Bohr``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import ase
import numpy as np
from pyscf import gto

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from omol_csh_density_fit import fit_density  # noqa: E402
from omol_csh_orca_labels import pyscf_ao_index  # noqa: E402

_L_OF_LETTER = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}
_COMPONENT_M = {
    0: {"s": 0},
    1: {"pz": 0, "px": 1, "py": -1},
    2: {"dz2": 0, "dxz": 1, "dyz": -1, "dx2y2": 2, "dxy": -2},
}
# ORCA inverts the phase of the |m| >= 3 real solid harmonics relative to PySCF.
FLIP_ABS_M = 3


def parse_json_labels(labels: list[str]) -> list[tuple[int, int, int, int]]:
    """'0C   3dxz' -> (atom, l, shell_index, m), in ORCA order."""
    out = []
    for lab in labels:
        match = re.match(r"^\s*(\d+)([A-Za-z]{1,2})\s+(\S+)\s*$", lab)
        if match is None:
            raise ValueError(f"cannot parse AO label {lab!r}")
        atom = int(match.group(1))
        orbital = match.group(3)
        shell = int(re.match(r"\d+", orbital).group(0))
        rest = orbital[len(str(shell)) :]
        ang = _L_OF_LETTER[rest[0]]
        mq = _COMPONENT_M[ang][rest] if ang <= 2 else int(rest[1:])
        out.append((atom, ang, shell, mq))
    return out


def orca_to_pyscf(mol, labels) -> tuple[np.ndarray, np.ndarray]:
    """perm[i] = ORCA index of the AO PySCF places at i, plus PySCF-order signs."""
    parsed = parse_json_labels(labels)
    if len(parsed) != mol.nao:
        raise ValueError(f"ORCA has {len(parsed)} AOs, PySCF has {mol.nao}")
    lookup = pyscf_ao_index(mol)
    perm = np.empty(mol.nao, dtype=int)
    signs = np.ones(mol.nao)
    for orca_index, key in enumerate(parsed):
        i = lookup[key]
        perm[i] = orca_index
        if abs(key[3]) >= FLIP_ABS_M:
            signs[i] = -1.0
    if sorted(perm.tolist()) != list(range(mol.nao)):
        raise ValueError("ORCA label map is not a permutation")
    return perm, signs

THEORY_KEYWORDS = "PBE0 aug-cc-pVDZ TightSCF EnGrad"
BASIS = "augccpvdz"
AUXBASIS = "augccpvqzjkfit"
XC = "pbe0"
CHARGE = 0
MULT = 1
NPROCS_DEFAULT = 8
MAXCORE_MB_DEFAULT = 4000

_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+([+-]?\d+\.\d+)")
_GRAD_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]{1,2})\s*:\s*"
    r"([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s+"
    r"([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s+"
    r"([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s*$"
)

ZENODO_RECORD = "4288677"
ZENODO_FILES = (
    "1000.xz",
    "2000.xz",
    "3000.xz",
    "4000.xz",
    "5000.xz",
    "6000.xz",
    "7000.xz",
    "8000.xz",
    "README.txt",
    "DupMols.dat",
    "createDB.py",
)


def zenodo_file_url(name: str) -> str:
    return f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}?download=1"


def unpadded_atoms(atom_numbers, positions) -> tuple[np.ndarray, np.ndarray]:
    """Drop dummy (Z=0) padding used in the DensNet base npy files."""
    z = np.asarray(atom_numbers, dtype=int).reshape(-1)
    xyz = np.asarray(positions, dtype=float).reshape(-1, 3)
    mask = z > 0
    return z[mask], xyz[mask]


def electron_count(atom_numbers, charge: int = CHARGE) -> int:
    z, _ = unpadded_atoms(atom_numbers, np.zeros((len(np.atleast_1d(atom_numbers)), 3)))
    return int(z.sum()) - int(charge)


def is_closed_shell(atom_numbers, charge: int = CHARGE) -> bool:
    return electron_count(atom_numbers, charge) % 2 == 0


def write_orca_inp(
    path: str | Path,
    atom_numbers,
    positions,
    *,
    charge: int = CHARGE,
    mult: int = MULT,
    nprocs: int = NPROCS_DEFAULT,
    maxcore_mb: int = MAXCORE_MB_DEFAULT,
) -> Path:
    z, xyz = unpadded_atoms(atom_numbers, positions)
    if not is_closed_shell(z, charge):
        raise ValueError("open-shell (odd electron count); closed-shell singlets only")
    lines = [
        f"! {THEORY_KEYWORDS}",
        f"%maxcore {int(maxcore_mb)}",
        "%pal",
        f"  nprocs {int(nprocs)}",
        "end",
        "%output",
        "  Print[P_MOs] 1",
        "end",
        f"* xyz {int(charge)} {int(mult)}",
    ]
    for anum, pos in zip(z, xyz):
        lines.append(f"  {int(anum)}  {pos[0]: .10f}  {pos[1]: .10f}  {pos[2]: .10f}")
    lines.append("*")
    lines.append("")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines))
    return dest


def parse_orca_energy(out_text: str) -> float:
    matches = _ENERGY_RE.findall(out_text)
    if not matches:
        raise ValueError("no FINAL SINGLE POINT ENERGY in ORCA output")
    return float(matches[-1])


def parse_orca_gradient(out_text: str) -> np.ndarray:
    """Cartesian gradient in Hartree/Bohr, shape (N, 3)."""
    lines = out_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "CARTESIAN GRADIENT" in line:
            start = i
    if start is None:
        raise ValueError("no CARTESIAN GRADIENT in ORCA output")
    rows = []
    for line in lines[start + 1 :]:
        if "norm of the cartesian" in line.lower():
            if rows:
                break
            continue
        if line.strip().startswith("----") and rows:
            break
        match = _GRAD_LINE_RE.match(line)
        if match:
            rows.append([float(match.group(3)), float(match.group(4)), float(match.group(5))])
        elif rows and not line.strip():
            break
    if not rows:
        raise ValueError("CARTESIAN GRADIENT block contained no atom lines")
    return np.asarray(rows, dtype=float)


def forces_from_gradient(gradient: np.ndarray) -> np.ndarray:
    """Negative gradient in Hartree/Angstrom (matches generate_dft_labels.py)."""
    return np.asarray(-np.asarray(gradient, dtype=float) / ase.units.Bohr, dtype=float)


def build_pyscf_mol(atom_numbers, positions, charge: int = CHARGE) -> gto.Mole:
    z, xyz = unpadded_atoms(atom_numbers, positions)
    mol = gto.Mole()
    mol.atom = [(int(anum), tuple(float(x) for x in pos)) for anum, pos in zip(z, xyz)]
    mol.basis = BASIS
    mol.charge = int(charge)
    mol.spin = 0
    mol.unit = "Angstrom"
    mol.cart = False
    mol.verbose = 0
    mol.build()
    return mol


def mos_from_orca_json(mol, data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mo_coeff, mo_occ, mo_energy) with AOs in PySCF order."""
    block = data["Molecule"]["MolecularOrbitals"]
    perm, signs = orca_to_pyscf(mol, block["OrbitalLabels"])
    nmo = len(block["MOs"])
    nao = mol.nao
    coeff_orca = np.empty((nao, nmo), dtype=float)
    occ = np.empty(nmo, dtype=float)
    energy = np.empty(nmo, dtype=float)
    for j, mo in enumerate(block["MOs"]):
        coeff_orca[:, j] = mo["MOCoefficients"]
        occ[j] = float(mo.get("Occupancy", 0.0))
        energy[j] = float(mo.get("OrbitalEnergy", 0.0))
    coeff = coeff_orca[perm, :] * signs[:, None]
    return coeff, occ, energy


def calc_dict_from_orca(
    mol,
    energy: float,
    forces: np.ndarray,
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
    *,
    fit_df: bool = True,
) -> dict[str, Any]:
    dm1 = np.einsum("pi,i,qi->pq", mo_coeff, mo_occ, mo_coeff)
    calc: dict[str, Any] = {
        "mo_coeff": np.asarray(mo_coeff, dtype=float),
        "mo_occ": np.asarray(mo_occ, dtype=float),
        "energy": float(energy),
        "forces": np.asarray(forces, dtype=float),
        "xc": XC,
        "auxbasis": AUXBASIS,
    }
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        dip_ints = mol.intor("int1e_r", comp=3)
    calc["dipole"] = np.einsum("xij,ji->x", dip_ints, dm1)
    if fit_df:
        coeffs, _, info = fit_density(mol, dm1, AUXBASIS)
        calc["df_coeff"] = np.asarray(coeffs, dtype=float)
        calc["df_info"] = info
    return calc


def load_base_npy(path: str | Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    data = raw.item() if getattr(raw, "shape", ()) == () else raw
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a dict-style base npy")
    return data


def iter_base_frames(data: dict) -> Iterator[dict[str, Any]]:
    positions = data["positions"]
    atom_numbers = data["atom_numbers"]
    n = len(positions)
    for i in range(n):
        z, xyz = unpadded_atoms(atom_numbers[i], positions[i])
        yield {
            "index": i,
            "atom_numbers": z.astype(int).tolist(),
            "positions": xyz.astype(float).tolist(),
        }


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return dest


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_shard(path: str | Path, shard_id: int, split: str, frames: list[dict]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {"shard_id": int(shard_id), "split": split, "frames": frames},
            indent=2,
        )
    )
    return dest


def read_shard(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def pad_frames(frames: list[dict]) -> dict[str, Any]:
    """Pack unpadded frames into the DensNet base-npy layout."""
    max_n = max(len(fr["atom_numbers"]) for fr in frames)
    n = len(frames)
    positions = np.zeros((n, max_n, 3), dtype=float)
    atom_numbers = np.zeros((n, max_n), dtype=int)
    energy = np.zeros((n,), dtype=float)
    forces = np.zeros((n, max_n, 3), dtype=float)
    dipole = np.zeros((n, 3), dtype=float)
    has_energy = False
    has_dipole = False
    for i, fr in enumerate(frames):
        z = np.asarray(fr["atom_numbers"], dtype=int)
        xyz = np.asarray(fr["positions"], dtype=float)
        nat = len(z)
        atom_numbers[i, :nat] = z
        positions[i, :nat] = xyz
        if "energy" in fr:
            energy[i] = fr["energy"]
            has_energy = True
        if "forces" in fr:
            forces[i, :nat] = np.asarray(fr["forces"], dtype=float)
        if "dipole" in fr:
            dipole[i] = np.asarray(fr["dipole"], dtype=float)
            has_dipole = True
    types = sorted({int(z) for z in atom_numbers.ravel() if z > 0})
    out = {
        "positions": positions,
        "atom_numbers": atom_numbers,
        "atom_types": types,
    }
    if has_energy:
        out["energy"] = energy
        out["forces"] = forces
    if has_dipole:
        out["dipole_moment"] = dipole
    return out


def iter_qm7x_records(h5_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield official QM7-X geometry records from one HDF5 file."""
    import h5py

    with h5py.File(h5_path, "r") as handle:
        for mol_id, mol in handle.items():
            if not hasattr(mol, "keys"):
                continue
            for conf_id, conf in mol.items():
                if not hasattr(conf, "keys") or "atNUM" not in conf or "atXYZ" not in conf:
                    continue
                z = np.asarray(conf["atNUM"][()], dtype=int).reshape(-1)
                xyz = np.asarray(conf["atXYZ"][()], dtype=float).reshape(-1, 3)
                rec: dict[str, Any] = {
                    "qm7x_id": f"{mol_id}/{conf_id}",
                    "source_file": Path(h5_path).name,
                    "atom_numbers": z.tolist(),
                    "positions": xyz.tolist(),
                }
                if "ePBE0+MBD" in conf:
                    rec["ePBE0_MBD_eV"] = float(np.asarray(conf["ePBE0+MBD"][()]).reshape(-1)[0])
                if "totFOR" in conf:
                    rec["totFOR_eV_A"] = np.asarray(conf["totFOR"][()], dtype=float).reshape(-1, 3).tolist()
                yield rec
