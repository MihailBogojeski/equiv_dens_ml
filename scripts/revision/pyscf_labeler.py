#!/usr/bin/env python3
"""Label one geometry with PySCF or gpu4pyscf at a `theory_levels` entry.

Factored out of generate_dft_labels.py so the calibration harness and the
production GPU worker cannot drift apart: whatever the calibration measured is
literally the code that then runs. The returned calc_dict matches the ORCA path
in qm7x_orca_common.calc_dict_from_orca field for field, which is what allows
labels from the two engines to share a split.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import ase
import numpy as np
from pyscf import df, dft, gto
from pyscf.scf import hf

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from omol_csh_density_fit import density_fit_rhs, solve_df_coeffs  # noqa: E402
from theory_levels import TheoryLevel, get_level  # noqa: E402

hf.MUTE_CHKFILE = True

INCORE_DF_MAX_NAO = 400


def _as_level(level: TheoryLevel | str) -> TheoryLevel:
    return level if isinstance(level, TheoryLevel) else get_level(level)


def _to_numpy(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


def gpu_available() -> bool:
    try:
        import gpu4pyscf  # noqa: F401
    except Exception:
        return False
    return True


def build_mol(atom_numbers, positions, level: TheoryLevel | str, charge: int = 0) -> gto.Mole:
    theory = _as_level(level)
    atom = [
        (int(anum), tuple(float(v) for v in pos))
        for anum, pos in zip(np.asarray(atom_numbers).reshape(-1), np.asarray(positions).reshape(-1, 3))
        if int(anum) > 0
    ]
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = theory.pyscf_basis
    mol.charge = int(charge)
    mol.spin = 0
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()
    if mol.nelectron % 2 != 0:
        raise ValueError(f"odd electron count ({mol.nelectron}); closed-shell singlets only")
    return mol


def density_fit_coeffs(mol: gto.Mole, dm1: np.ndarray, auxbasis: str) -> np.ndarray:
    """Auxiliary-basis expansion of an AO density, blocked when it has to be."""
    import scipy.linalg

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


def label_frame(
    atom_numbers,
    positions,
    level: TheoryLevel | str,
    *,
    use_gpu: bool = False,
    fit_df: bool = True,
    with_forces: bool = True,
    charge: int = 0,
) -> tuple[gto.Mole, dict[str, Any], dict[str, Any]]:
    """Run one SCF and return ``(mol, calc_dict, diagnostics)``."""
    theory = _as_level(level)
    mol = build_mol(atom_numbers, positions, theory, charge=charge)

    mf = dft.RKS(mol)
    mf.chkfile = None
    mf.xc = theory.pyscf_xc
    mf.max_cycle = 1000
    if theory.nlc:
        # wB97M-V carries VV10; PySCF picks that up from the functional name, but
        # the NLC grid is pinned here so cost is reproducible across machines.
        mf.nlcgrids.level = theory.nlcgrids_level

    cycles: list[int] = []
    mf.callback = lambda envs: cycles.append(int(envs.get("cycle", 0)))

    on_gpu = False
    if use_gpu and gpu_available():
        mf = mf.to_gpu()
        on_gpu = True

    t0 = time.perf_counter()
    mf.kernel()
    t_scf = time.perf_counter() - t0
    converged = bool(mf.converged)

    t_grad = 0.0
    gradients = None
    if with_forces:
        t1 = time.perf_counter()
        try:
            gradients = _to_numpy(mf.nuc_grad_method().kernel())
        except Exception:
            mf = mf.to_cpu() if on_gpu else mf
            on_gpu = False
            gradients = _to_numpy(mf.nuc_grad_method().kernel())
        t_grad = time.perf_counter() - t1

    energy = float(mf.e_tot)
    mo_coeff = _to_numpy(mf.mo_coeff)
    mo_occ = _to_numpy(mf.mo_occ)

    if theory.d4:
        import dftd4.pyscf as d4disp

        e_disp, g_disp = d4disp.DFTD4Dispersion(mol, xc=theory.pyscf_xc).kernel()
        energy += float(e_disp)
        if gradients is not None:
            gradients = gradients + np.asarray(g_disp, dtype=float)

    dm1 = np.einsum("pi,i,qi->pq", mo_coeff, mo_occ, mo_coeff)

    calc: dict[str, Any] = {
        "mo_coeff": np.asarray(mo_coeff, dtype=float),
        "mo_occ": np.asarray(mo_occ, dtype=float),
        "energy": energy,
        "xc": theory.pyscf_xc,
        "auxbasis": theory.auxbasis,
    }
    if gradients is not None:
        calc["forces"] = np.asarray(-gradients / ase.units.Bohr, dtype=float)
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        dip_ints = mol.intor("int1e_r", comp=3)
    calc["dipole"] = np.einsum("xij,ji->x", dip_ints, dm1)

    t_df = 0.0
    if fit_df:
        t2 = time.perf_counter()
        calc["df_coeff"] = density_fit_coeffs(mol, dm1, theory.auxbasis)
        t_df = time.perf_counter() - t2

    diagnostics = {
        "theory": theory.key,
        "engine": "gpu4pyscf" if on_gpu else "pyscf",
        "nao": int(mol.nao),
        "naux": int(calc["df_coeff"].shape[0]) if "df_coeff" in calc else 0,
        "n_atoms": int(mol.natm),
        "n_elec": int(mol.nelectron),
        "converged": converged,
        "scf_cycles": max(cycles) + 1 if cycles else -1,
        "t_scf_s": t_scf,
        "t_grad_s": t_grad,
        "t_df_s": t_df,
    }
    return mol, calc, diagnostics
