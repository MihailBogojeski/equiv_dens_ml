#!/usr/bin/env python3
"""
Generate polythiophene dataset from trajectory.

Runs PBE/aug-cc-pVDZ DFT with D4 dispersion on each frame and produces
the dataset format expected by AtomsDensityData (dens_dataset + np_dataset).

Supports input: XYZ, single NPY, or per-frame NPY pattern (e.g. prefix_*.npy).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import ase
import ase.io
import numpy as np
from pyscf import gto, dft, df, lib
from pyscf.scf import hf
import scipy

# Add src to path for equiv_dens imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import equiv_dens.utils.base as utils
import dftd4.pyscf as d4disp

hf.MUTE_CHKFILE = True

BASIS = "augccpvdz"
AUXBASIS = "augccpvqzjkfit"


def _try_import_gpu4pyscf():
    """Try to import gpu4pyscf. Returns True if available."""
    try:
        import gpu4pyscf  # noqa: F401
        return True
    except ImportError:
        return False


def _to_numpy(arr):
    """Convert CuPy array to NumPy if needed (for gpu4pyscf)."""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def load_trajectory(trajectory_path: str, stride: int = 1):
    """
    Load trajectory from XYZ, single NPY, or per-frame NPY pattern.

    Returns
    -------
    list of tuples
        Each element is (positions, atom_numbers) for one frame.
        positions: (natoms, 3), atom_numbers: (natoms,)
    """
    path = Path(trajectory_path)

    if path.suffix.lower() == ".xyz":
        atoms_list = list(ase.io.iread(str(path)))
        frames = []
        for mol in atoms_list[::stride]:
            pos = mol.get_positions()
            anum = mol.get_atomic_numbers()
            frames.append((pos, anum))
        return frames

    if path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=True)

        # Single NPY: dict with positions (N, natoms, 3), atom_numbers
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item() if data.ndim == 0 else data

        if isinstance(data, dict):
            pos = data["positions"]
            anum = data["atom_numbers"]
            if anum.ndim == 1:
                anum = np.tile(anum, (pos.shape[0], 1))
            frames = []
            for i in range(0, len(pos), stride):
                p = pos[i]
                a = anum[i] if anum.ndim > 1 else anum
                nonzero = a > 0
                frames.append((p[nonzero], a[nonzero]))
            return frames

        # List of (mol.pack(), calc_dict) - pre-computed
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            first = data[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                if "mo_coeff" in first[1]:
                    return data  # Return as-is for aggregation
            # List of structures
            frames = []
            for i in range(0, len(data), stride):
                item = data[i]
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    mol = gto.Mole.unpack(item[0])
                    mol.build()
                    pos = mol.atom_coords()
                    anum = mol.atom_charges()
                    frames.append((pos, anum))
                else:
                    raise ValueError(f"Unexpected NPY structure at index {i}")
            return frames

    # Per-frame NPY pattern (e.g. prefix_0.npy, prefix_1.npy)
    if "*" in str(trajectory_path):
        pattern = trajectory_path
    else:
        # Assume directory or base path: look for *_0.npy, *_1.npy
        stem = path.stem
        parent = path.parent
        pattern = str(parent / f"{stem}_*.npy")

    files = sorted(glob.glob(pattern), key=lambda f: _natural_sort_key(f))
    if not files:
        raise FileNotFoundError(f"No files matching pattern: {pattern}")

    frames = []
    for f in files[::stride]:
        data = np.load(f, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item() if data.ndim == 0 else list(data)

        if isinstance(data, (list, tuple)) and len(data) >= 2:
            # (mol.pack(), calc_dict)
            if isinstance(data[0], dict) and "atom" in data[0]:
                mol = gto.Mole.unpack(data[0])
                mol.build()
                pos = mol.atom_coords()
                anum = np.array([a[0] for a in data[0]["atom"]])
                frames.append((pos, anum))
            elif isinstance(data[1], dict) and "mo_coeff" in data[1]:
                # Pre-computed: return as list of (mol.pack(), calc_dict)
                frames.append(data)
            else:
                raise ValueError(f"Cannot parse per-frame file: {f}")
        elif isinstance(data, dict) and "positions" in data:
            pos = data["positions"]
            anum = data["atom_numbers"]
            if pos.ndim == 3:
                pos = pos[0]
            if anum.ndim > 1:
                anum = anum[0]
            frames.append((pos, anum))
        else:
            raise ValueError(f"Unexpected format in {f}")

    return frames


def _natural_sort_key(s):
    """Sort key for natural ordering of filenames (e.g. prefix_2.npy before prefix_10.npy)."""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _is_precomputed_frame(frame):
    """Check if frame is pre-computed (mol.pack(), calc_dict) with mo_coeff."""
    if not isinstance(frame, (list, tuple)) or len(frame) != 2:
        return False
    return isinstance(frame[1], dict) and "mo_coeff" in frame[1]


def compute_frame(
    positions: np.ndarray,
    atom_numbers: np.ndarray,
    use_gpu: bool,
    use_df: bool,
) -> tuple:
    """
    Run PBE/aug-cc-pVDZ + D4 on one frame.

    Returns
    -------
    tuple
        (mol.pack(), calc_dict) with mo_coeff, mo_occ, energy, forces,
        and optionally df_coeff, auxbasis.
    """
    atom = [(int(anum), pos) for anum, pos in zip(atom_numbers, positions) if anum > 0]
    nelec = sum(gto.charge(anum) for anum, _ in atom)
    if nelec % 2 != 0:
        raise ValueError(
            f"Structure has odd electron count ({nelec}); only closed-shell systems "
            "are supported. Check input trajectory for errors."
        )
    mol = gto.M(atom=atom, basis=BASIS)
    mol.build()

    mf = dft.RKS(mol)
    mf.chkfile = None
    mf.xc = "pbe"
    mf.max_cycle = 1000

    if use_gpu and _try_import_gpu4pyscf():
        mf = mf.to_gpu()

    mf.kernel()

    # dftd4.pyscf requires standard PySCF SCF; convert GPU object to CPU
    if use_gpu:
        mf = mf.to_cpu()

    d4mf = d4disp.energy(mf).run()
    grad = d4mf.nuc_grad_method()
    gradients = grad.kernel()

    res = [mol.pack()]
    calc_dict = {
        "mo_coeff": _to_numpy(d4mf.mo_coeff),
        "mo_occ": _to_numpy(d4mf.mo_occ),
        "energy": float(d4mf.e_tot),
        "forces": _to_numpy(-gradients / ase.units.Bohr),
    }

    if use_df:
        dm1 = d4mf.make_rdm1(d4mf.mo_coeff, d4mf.mo_occ)
        dm1 = _to_numpy(dm1)
        auxmol = df.addons.make_auxmol(mol, AUXBASIS)
        ints_3c2e = df.incore.aux_e2(mol, auxmol, intor="int3c2e")
        ints_2c2e = auxmol.intor("int2c2e")
        nao = mol.nao
        naux = auxmol.nao
        df_coef = scipy.linalg.solve(
            ints_2c2e, ints_3c2e.reshape(nao * nao, naux).T
        )
        df_coef = df_coef.reshape(naux, nao, nao)
        df_basis = np.einsum("Pij,ij->P", df_coef, dm1)
        calc_dict["df_coeff"] = df_basis
        calc_dict["auxbasis"] = AUXBASIS

    res.append(calc_dict)
    return tuple(res)


def main():
    parser = argparse.ArgumentParser(
        description="Generate polythiophene dataset from trajectory (PBE/aug-cc-pVDZ + D4)"
    )
    parser.add_argument(
        "--trajectory",
        "-t",
        type=str,
        required=True,
        help="Path to trajectory: XYZ, NPY, or glob pattern (e.g. prefix_*.npy)",
    )
    parser.add_argument(
        "--output-prefix",
        "-o",
        type=str,
        default=None,
        help="Output prefix. Default: derived from trajectory path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory (default: current)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--df",
        action="store_true",
        help="Add density-fitting coeffs (aug-cc-pVQZ-JKfit) for training",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (use CPU PySCF only)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save results every N frames (default: 100)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    traj_path = Path(args.trajectory)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_prefix is not None:
        prefix = args.output_prefix
    else:
        prefix = traj_path.stem
        if "*" in str(traj_path):
            prefix = Path(glob.glob(str(traj_path))[0]).stem.rsplit("_", 1)[0]

    dens_path = out_dir / f"{prefix}_pyscf_augccpvdz_d4.npy"
    npy_path = out_dir / f"{prefix}_npy.npy"

    use_gpu = not args.no_gpu and _try_import_gpu4pyscf()
    if not args.quiet:
        print(f"GPU acceleration: {'enabled' if use_gpu else 'disabled'}")

    # Load trajectory
    frames = load_trajectory(args.trajectory, stride=args.stride)

    # Check if frames are pre-computed (list of (mol.pack(), calc_dict))
    precomputed = len(frames) > 0 and _is_precomputed_frame(frames[0])

    if precomputed:
        if not args.quiet:
            print(f"Aggregating {len(frames)} pre-computed frames")
        results = []
        for item in frames:
            mol_data, calc_dict = item
            calc_dict = dict(calc_dict)
            for key in ("mo_coeff", "mo_occ", "forces"):
                if key in calc_dict:
                    calc_dict[key] = _to_numpy(calc_dict[key])
            if "df_coeff" in calc_dict:
                calc_dict["df_coeff"] = _to_numpy(calc_dict["df_coeff"])
            results.append((mol_data, calc_dict))
    else:
        # Run DFT on each frame
        if os.path.exists(dens_path):
            results = np.load(dens_path, allow_pickle=True).tolist()
            start_idx = len(results)
            if not args.quiet:
                print(f"Resuming from frame {start_idx} (loaded {len(results)} existing)")
        else:
            results = []
            start_idx = 0

        try:
            from tqdm import tqdm
            iterator = tqdm(
                range(start_idx, len(frames)),
                desc="Computing DFT",
                disable=args.quiet,
            )
        except ImportError:
            iterator = range(start_idx, len(frames))

        for i in iterator:
            pos, anum = frames[i]
            res = compute_frame(pos, anum, use_gpu=use_gpu, use_df=args.df)
            results.append(res)

            if (len(results)) % args.save_interval == 0:
                np.save(dens_path, results, allow_pickle=True)
                if not args.quiet:
                    print(f"  Saved checkpoint at frame {len(results)}")

        np.save(dens_path, results, allow_pickle=True)

    # Build np_dataset (calc_dict_to_npy requires len>=2, so duplicate if single frame)
    data_for_npy = results if len(results) >= 2 else results + results
    npy_data = utils.calc_dict_to_npy(
        data_for_npy, convert_forces=False, compress_atoms=True
    )
    np.save(npy_path, npy_data, allow_pickle=True)

    if not args.quiet:
        print(f"Wrote density dataset: {dens_path}")
        print(f"Wrote structures: {npy_path}")
        print(f"Total frames: {len(results)}")


if __name__ == "__main__":
    main()
