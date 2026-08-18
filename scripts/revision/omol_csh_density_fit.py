#!/usr/bin/env python3
"""Fit an AO density matrix onto auxiliary-basis coefficients.

DenSNet trains on density-fitting coefficients, not AO density matrices, so a
CSH-derived P still has to be projected onto an auxiliary basis:

    (A|B) c_B = sum_{mu,nu} (mu nu|A) P_{mu,nu}

The paper pipeline (scripts/revision/generate_dft_labels.py) builds the whole
3-centre tensor in core, which is fine for aug-cc-pVDZ on ethanol but not for
def2-TZVPD on a 60-atom OMol structure - that tensor runs to hundreds of GB.
Here the right-hand side is accumulated over blocks of auxiliary shells
instead, so memory scales with the block size rather than with naux.

Fit quality is reported in the Coulomb metric, which is the norm the fit
actually minimises:

    dE = (rho|rho) - b . c  >= 0
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from pyscf import df, gto, lib


def auxmol_for(mol: gto.Mole, auxbasis: str) -> gto.Mole:
    return df.addons.make_auxmol(mol, auxbasis)


def _packed_density(dm: np.ndarray) -> np.ndarray:
    """Lower-triangle packing of dm with off-diagonal elements pre-doubled."""
    nao = dm.shape[0]
    packed = lib.pack_tril(dm + dm.conj().T)
    idx = np.arange(nao)
    packed[idx * (idx + 1) // 2 + idx] *= 0.5
    return packed


def density_fit_rhs(mol: gto.Mole, auxmol: gto.Mole, dm: np.ndarray, block: int = 40) -> np.ndarray:
    """b_A = sum_{mu,nu} (mu nu|A) P_{mu,nu}, accumulated over auxiliary shell blocks."""
    packed = _packed_density(dm)
    aux_loc = auxmol.ao_loc_nr()
    rhs = np.zeros(auxmol.nao)
    for start in range(0, auxmol.nbas, block):
        end = min(start + block, auxmol.nbas)
        ints = df.incore.aux_e2(
            mol,
            auxmol,
            intor="int3c2e",
            aosym="s2ij",
            shls_slice=(0, mol.nbas, 0, mol.nbas, start, end),
        )
        rhs[aux_loc[start] : aux_loc[end]] = packed @ ints
    return rhs


def density_fit_rhs_gpu(mol: gto.Mole, auxmol: gto.Mole, dm: np.ndarray) -> np.ndarray:
    """Same contraction on the GPU, for systems whose 3-centre tensor fits.

    gpu4pyscf builds the whole (nao, nao, naux) tensor at once, which is only
    viable for the smaller structures; the caller is responsible for the memory
    guard and for falling back to the blocked CPU path.
    """
    import cupy
    from gpu4pyscf.df import int3c2e

    ints = int3c2e.get_int3c2e(mol, auxmol)
    rhs = cupy.einsum("ijP,ij->P", ints, cupy.asarray(dm))
    del ints
    cupy.get_default_memory_pool().free_all_blocks()
    return cupy.asnumpy(rhs)


def gpu_tensor_bytes(mol: gto.Mole, auxmol: gto.Mole) -> int:
    return mol.nao * mol.nao * auxmol.nao * 8


def solve_df_coeffs(auxmol: gto.Mole, rhs: np.ndarray, lindep: float = 1e-10):
    """Solve (A|B) c = b, falling back to a truncated eigendecomposition."""
    j2c = auxmol.intor("int2c2e")
    try:
        coeffs = scipy.linalg.solve(j2c, rhs, assume_a="pos")
    except (scipy.linalg.LinAlgError, np.linalg.LinAlgError):
        val, vec = np.linalg.eigh(j2c)
        keep = val > lindep * val.max()
        coeffs = vec[:, keep] @ ((vec[:, keep].T @ rhs) / val[keep])
    return coeffs, j2c


def aux_charges(auxmol: gto.Mole, exponent: float = 1e-8) -> np.ndarray:
    """Integral of each auxiliary function over all space.

    Writing this analytically means committing to PySCF's internal contraction
    and solid-harmonic normalisation, which is easy to get wrong by a constant.
    Instead take the overlap with a single s Gaussian so diffuse it is
    effectively constant everywhere, and divide out its normalisation: as the
    exponent goes to zero the overlap tends to the plain integral of each
    auxiliary function, independently of where the probe is centred.
    """
    probe = gto.M(atom=[("H", (0.0, 0.0, 0.0))], basis={"H": [[0, [exponent, 1.0]]]}, spin=1)
    cross = gto.mole.intor_cross("int1e_ovlp", auxmol, probe).ravel()
    return cross / (2.0 * exponent / np.pi) ** 0.75


def fit_density(
    mol,
    dm,
    auxbasis: str,
    block: int = 40,
    coulomb_reference=None,
    device: str = "cpu",
    gpu_budget_bytes: int = 8_000_000_000,
):
    """Return DF coefficients plus fit diagnostics."""
    auxmol = auxmol_for(mol, auxbasis)
    used_gpu = False
    if device == "gpu" and gpu_tensor_bytes(mol, auxmol) <= gpu_budget_bytes:
        try:
            rhs = density_fit_rhs_gpu(mol, auxmol, dm)
            used_gpu = True
        except Exception:
            rhs = density_fit_rhs(mol, auxmol, dm, block=block)
    else:
        rhs = density_fit_rhs(mol, auxmol, dm, block=block)
    coeffs, _ = solve_df_coeffs(auxmol, rhs)

    charges = aux_charges(auxmol)
    n_elec_fit = float(coeffs @ charges)
    ovlp = mol.intor("int1e_ovlp")
    n_elec_exact = float(np.einsum("ij,ij->", ovlp, dm))

    info = {
        "naux": int(auxmol.nao),
        "nao": int(mol.nao),
        "auxbasis": auxbasis,
        "n_elec_exact": n_elec_exact,
        "n_elec_df": n_elec_fit,
        "n_elec_error": n_elec_fit - n_elec_exact,
        "used_gpu": used_gpu,
    }
    if coulomb_reference is not None:
        # (rho|rho) from the exact density; dE is the residual Coulomb error.
        self_energy = float(np.einsum("ij,ij->", coulomb_reference, dm))
        info["coulomb_self_energy"] = self_energy
        info["coulomb_fit_error"] = self_energy - float(rhs @ coeffs)
    return coeffs, auxmol, info
