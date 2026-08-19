#!/usr/bin/env python3
"""Shared ORCA helpers: inputs, parsers, and DensNet calc_dict conversion.

The level of theory comes from scripts/revision/theory_levels.py rather than
being fixed here, so the same worker serves the QM7-X campaign at
PBE0/aug-cc-pVDZ and the water campaign at wB97M-V/def2-TZVPD. The module-level
``THEORY_KEYWORDS``/``BASIS``/``XC`` names remain the QM7-X defaults so runs
already in flight keep producing identical inputs.

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
from pyscf import df, gto

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import scipy.linalg

from omol_csh_density_fit import density_fit_rhs, solve_df_coeffs  # noqa: E402
from theory_levels import DEFAULT_LEVEL, TheoryLevel, get_level  # noqa: E402

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


def pyscf_ao_index(mol) -> dict[tuple[int, int, int, int], int]:
    """Map (atom, l, shell_index, m) -> PySCF AO index.

    Counts generally contracted functions (``nctr > 1``), which Dunning
    bases such as aug-cc-pVDZ use. Segmented bases (def2) keep nctr=1.
    Shell index is 1-based per angular momentum on each atom.
    """
    ao_loc = mol.ao_loc_nr()
    counters: dict[tuple[int, int], int] = {}
    index: dict[tuple[int, int, int, int], int] = {}
    for shell in range(mol.nbas):
        atom = mol.bas_atom(shell)
        ang = mol.bas_angular(shell)
        nctr = mol.bas_nctr(shell)
        n_comp = 2 * ang + 1
        ms = [1, -1, 0] if ang == 1 else list(range(-ang, ang + 1))
        for ictr in range(nctr):
            counters[(atom, ang)] = counters.get((atom, ang), 0) + 1
            n = counters[(atom, ang)]
            for j, m in enumerate(ms):
                index[(atom, ang, n, m)] = ao_loc[shell] + ictr * n_comp + j
    return index


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

DEFAULT_THEORY = get_level(DEFAULT_LEVEL)
THEORY_KEYWORDS = DEFAULT_THEORY.orca_keywords
BASIS = DEFAULT_THEORY.pyscf_basis
AUXBASIS = DEFAULT_THEORY.auxbasis
XC = DEFAULT_THEORY.pyscf_xc
CHARGE = 0
MULT = 1
NPROCS_DEFAULT = 8
MAXCORE_MB_DEFAULT = 4000

# Above this many AOs the in-core 3-centre tensor stops being affordable
# (nao^2 * naux * 8 bytes), so the auxiliary-shell-blocked accumulation is used.
INCORE_DF_MAX_NAO = 400


def _as_level(level: TheoryLevel | str | None) -> TheoryLevel:
    if level is None:
        return DEFAULT_THEORY
    return level if isinstance(level, TheoryLevel) else get_level(level)

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
    level: TheoryLevel | str | None = None,
    reference: bool = False,
) -> Path:
    theory = _as_level(level)
    keywords = theory.orca_reference_keywords if reference else theory.orca_keywords
    z, xyz = unpadded_atoms(atom_numbers, positions)
    if not is_closed_shell(z, charge):
        raise ValueError("open-shell (odd electron count); closed-shell singlets only")
    lines = [
        f"! {keywords}",
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


_ENGRAD_FLOAT_RE = re.compile(r"^\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)\s*$")


def parse_orca_engrad(text: str) -> tuple[float, np.ndarray]:
    """Parse an ORCA ``.engrad`` file -> (energy Hartree, gradient Hartree/Bohr)."""
    lines = [ln.strip() for ln in text.splitlines()]
    natoms = None
    energy = None
    grads: list[float] = []
    mode = None
    for line in lines:
        if line.startswith("#"):
            if "Number of atoms" in line:
                mode = "natoms"
            elif "current total energy" in line:
                mode = "energy"
            elif "current gradient" in line:
                mode = "grad"
            elif "atomic numbers" in line:
                mode = None
            continue
        if not line:
            continue
        if mode == "natoms" and natoms is None:
            natoms = int(line)
        elif mode == "energy" and energy is None:
            energy = float(line)
        elif mode == "grad":
            match = _ENGRAD_FLOAT_RE.match(line)
            if match:
                grads.append(float(match.group(1)))
            elif natoms is not None and len(grads) >= 3 * natoms:
                break
    if energy is None or natoms is None or len(grads) < 3 * natoms:
        raise ValueError("incomplete ORCA engrad file")
    return float(energy), np.asarray(grads[: 3 * natoms], dtype=float).reshape(natoms, 3)


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


def build_pyscf_mol(
    atom_numbers,
    positions,
    charge: int = CHARGE,
    level: TheoryLevel | str | None = None,
) -> gto.Mole:
    z, xyz = unpadded_atoms(atom_numbers, positions)
    mol = gto.Mole()
    mol.atom = [(int(anum), tuple(float(x) for x in pos)) for anum, pos in zip(z, xyz)]
    mol.basis = _as_level(level).pyscf_basis
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


def mo_orthonormality_error(mol, mo_coeff: np.ndarray) -> float:
    """max |C^T S C - I|, the self-contained check on the AO permutation.

    A wrong AO index or a wrong solid-harmonic phase breaks orthonormality in
    the PySCF overlap while leaving the ORCA energy untouched, so this catches
    a mis-mapped shell without needing a second code to compare against. It
    matters most for f functions: aug-cc-pVDZ never reaches |m| >= 3, so the
    ``FLIP_ABS_M`` phase rule is first exercised by def2-TZVPD.
    """
    ovlp = mol.intor("int1e_ovlp")
    coeff = np.asarray(mo_coeff, dtype=float)
    gram = coeff.T @ ovlp @ coeff
    return float(np.abs(gram - np.eye(gram.shape[0])).max())


def calc_dict_from_orca(
    mol,
    energy: float,
    forces: np.ndarray,
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
    *,
    fit_df: bool = True,
    level: TheoryLevel | str | None = None,
) -> dict[str, Any]:
    theory = _as_level(level)
    dm1 = np.einsum("pi,i,qi->pq", mo_coeff, mo_occ, mo_coeff)
    calc: dict[str, Any] = {
        "mo_coeff": np.asarray(mo_coeff, dtype=float),
        "mo_occ": np.asarray(mo_occ, dtype=float),
        "energy": float(energy),
        "forces": np.asarray(forces, dtype=float),
        "xc": theory.pyscf_xc,
        "auxbasis": theory.auxbasis,
    }
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        dip_ints = mol.intor("int1e_r", comp=3)
    calc["dipole"] = np.einsum("xij,ji->x", dip_ints, dm1)
    if fit_df:
        calc["df_coeff"] = density_fit_coeffs(mol, dm1, theory.auxbasis)
    return calc


def density_fit_coeffs(mol, dm1: np.ndarray, auxbasis: str = AUXBASIS) -> np.ndarray:
    """Project an AO density onto auxiliary coefficients.

    Small systems keep the original in-core route from generate_dft_labels.py.
    Past ``INCORE_DF_MAX_NAO`` the (nao, nao, naux) tensor no longer fits -- a
    24-water cluster in def2-TZVPD needs tens of GB -- so the right-hand side is
    accumulated over auxiliary shell blocks instead.
    """
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    nao = mol.nao
    naux = auxmol.nao
    if nao > INCORE_DF_MAX_NAO:
        rhs = density_fit_rhs(mol, auxmol, np.asarray(dm1, dtype=float))
        coeffs, _ = solve_df_coeffs(auxmol, rhs)
        return coeffs
    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
    ints_2c2e = auxmol.intor("int2c2e")
    df_coef = scipy.linalg.solve(ints_2c2e, ints_3c2e.reshape(nao * nao, naux).T)
    df_coef = df_coef.reshape(naux, nao, nao)
    return np.einsum("Pij,ij->P", df_coef, dm1)


def _run_cmd(cmd: list[str], cwd: Path, log: Path) -> None:
    import subprocess

    with log.open("ab") as fh:
        fh.write(("+ " + " ".join(cmd) + "\n").encode())
        proc = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def run_orca_single(
    atom_numbers,
    positions,
    job_dir: Path,
    *,
    level: TheoryLevel | str | None = None,
    orca_bin: str = "orca",
    orca2json_bin: str = "orca_2json",
    nprocs: int = NPROCS_DEFAULT,
    maxcore_mb: int = MAXCORE_MB_DEFAULT,
    charge: int = CHARGE,
    fit_df: bool = True,
    reference: bool = False,
) -> tuple[gto.Mole, dict[str, Any], dict[str, Any]]:
    """SCF one geometry in ORCA and return ``(mol, calc_dict, diagnostics)``.

    Shared by the array worker and the calibration harness so a measured cost
    is always the cost of the code that actually runs in production.
    """
    import time

    theory = _as_level(level)
    job_dir.mkdir(parents=True, exist_ok=True)
    inp = write_orca_inp(
        job_dir / "job.inp",
        atom_numbers,
        positions,
        charge=charge,
        nprocs=nprocs,
        maxcore_mb=maxcore_mb,
        level=theory,
        reference=reference,
    )

    out_path = job_dir / "job.out"
    t0 = time.perf_counter()
    _run_cmd([orca_bin, str(inp.name)], cwd=job_dir, log=out_path)
    t_scf = time.perf_counter() - t0

    engrad_path = job_dir / "job.engrad"
    if engrad_path.exists():
        energy, gradient = parse_orca_engrad(engrad_path.read_text())
    else:
        out_text = out_path.read_text(errors="replace")
        energy = parse_orca_energy(out_text)
        gradient = parse_orca_gradient(out_text)
    forces = forces_from_gradient(gradient)

    gbw = job_dir / "job.gbw"
    if not gbw.exists():
        raise FileNotFoundError(f"ORCA did not write {gbw}")
    _run_cmd([orca2json_bin, str(gbw.name)], cwd=job_dir, log=job_dir / "orca_2json.log")
    json_path = job_dir / "job.json"
    if not json_path.exists():
        candidates = list(job_dir.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"orca_2json produced no JSON in {job_dir}")
        json_path = candidates[0]
    data = json.loads(json_path.read_text())

    mol = build_pyscf_mol(atom_numbers, positions, charge=charge, level=theory)
    mo_coeff, mo_occ, _mo_e = mos_from_orca_json(mol, data)
    ortho_err = mo_orthonormality_error(mol, mo_coeff)

    t1 = time.perf_counter()
    calc = calc_dict_from_orca(mol, energy, forces, mo_coeff, mo_occ, fit_df=fit_df, level=theory)
    t_df = time.perf_counter() - t1

    diagnostics = {
        "theory": theory.key,
        "engine": "orca",
        "reference_keywords": reference,
        "nao": int(mol.nao),
        "naux": int(calc["df_coeff"].shape[0]) if "df_coeff" in calc else 0,
        "n_atoms": int(mol.natm),
        "n_elec": int(mol.nelectron),
        "t_scf_s": t_scf,
        "t_df_s": t_df,
        "mo_orthonormality_error": ortho_err,
    }
    return mol, calc, diagnostics


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
