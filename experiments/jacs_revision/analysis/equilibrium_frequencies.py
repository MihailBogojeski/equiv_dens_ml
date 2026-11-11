#!/usr/bin/env python3
"""DenSNet versus DFT minima, conformer energies and harmonic spectra (R2.11).

For each published small-molecule model, at the level of theory it was trained
on (ethanol PBE/cc-pVDZ; ethanethiol and resorcinol PBE/aug-cc-pVDZ, no
dispersion, which is what their stored labels contain):

1. ``validate``: DenSNet forces against DFT forces on held-out frames (MAE,
   slope of a through-origin fit, Pearson r). A model whose force scale or
   surface is off would make the rest meaningless, so this comes first.
2. ``run``: every conformer (ethanol and ethanethiol trans/gauche about the
   C-C-X-H torsion; resorcinol's three planar OH rotamers) is started from the
   same non-equilibrium geometry, a thermal training frame with the torsion
   set, and optimised separately with DFT and with DenSNet to max|F| < fmax.
   Both minima are compared (RMSD after superposition, internal coordinates),
   each method's residual force is evaluated at the other's minimum, and the
   conformer energy ordering is compared. Harmonic frequencies and IR
   intensities come from central finite differences of forces and dipoles
   with the same displacement for both methods; modes are paired by
   maximum mass-weighted eigenvector overlap (Hungarian).

DFT runs in GPU4PySCF (no density fitting, grids level 3, conv_tol 1e-11) and
falls back to PySCF on the CPU.

Usage:
  python experiments/jacs_revision/analysis/equilibrium_frequencies.py validate --molecule ethanethiol
  python experiments/jacs_revision/analysis/equilibrium_frequencies.py run --molecule ethanol
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0] = [str(REPO), str(REPO / "src")]
OUT = REPO / "results" / "revision" / "equilibrium_v2"

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903
EA_TO_DEBYE = 4.803204712570263
KMMOL_PER_D2_A2_AMU = 42.2561  # IR intensity prefactor for (dmu/dQ in D/(A amu^0.5))^2

MOLECULES = {
    "ethanol": {
        "template": ("datasets/ethanol_train_10.npy", 0),
        "basis": "cc-pvdz",
        # _ext, not the 96w7KyGG run of config/md/nn/ethanol.txt, whose forces do
        # not match PBE/cc-pVDZ with the current code (Pearson r 0.75 vs 0.9999).
        # Its training set is not on disk, so its force scale is fitted in validate.
        "run": "paper/models/ethanol/2024-04-18_5K9Ne7it_ext",
        "override": "config/md/nn/ethanol_ext_paths.txt",
        "loader": "spline",
        "train_np": None,
        "hetero": 8,
        "conformers": {"trans": (180.0,), "gauche": (60.0,)},
    },
    "ethanethiol": {
        "template": ("datasets/ethanethiol_combo_Aidx-1000.npy", 0),
        "basis": "aug-cc-pvdz",
        "run": "paper/models/ethanethiol/2024-02-29_NUmID4hT_ext4",
        "override": "config/md/nn/ethanethiol.txt",
        "loader": "calculator",
        "train_np": "datasets/ethanethiol_combo_Aidx-1000.npy",
        "test_np": "datasets/ethanethiol_rand-1000.npy",
        "hetero": 16,
        "conformers": {"trans": (180.0,), "gauche": (60.0,)},
    },
    "resorcinol": {
        "template": ("datasets/resorcinol_combo_kmeansidx-1000_train.npy", 0),
        "basis": "aug-cc-pvdz",
        "run": "paper/models/resorcinol/2024-03-18_Ozf6CkNF_ext1",
        "override": "config/md/nn/resorcinol.txt",
        "loader": "spline",
        "train_np": "datasets/resorcinol_combo_kmeansidx-1000_train.npy",
        "test_np": "datasets/resorcinol_augccpvdz_test.npy",
        "hetero": 8,
        # H-O-C-C2 torsions, C2 the ring carbon between the two C-O carbons.
        "conformers": {"syn_syn": (0.0, 0.0), "syn_anti": (0.0, 180.0), "anti_anti": (180.0, 180.0)},
    },
}


# ----------------------------------------------------------------------------- geometry
def frame(path: str, index: int):
    from ase import Atoms

    data = np.load(REPO / path, allow_pickle=True)
    data = data.item() if data.dtype == object and data.ndim == 0 else data
    z = np.asarray(data["atom_numbers"])
    z = np.asarray(z if z.ndim == 1 and z.dtype != object else z[index], dtype=int)
    r = np.asarray(data["positions"][index], dtype=float)
    return Atoms(numbers=z[z > 0], positions=r[: len(z)][z > 0])


def torsions(atoms, hetero: int):
    """(C_ref, C_X, X, H_X) quadruples for each X-H group (X = O or S)."""
    z = atoms.get_atomic_numbers()
    d = atoms.get_all_distances()
    carbons = np.flatnonzero(z == 6)
    quads = []
    xs = np.flatnonzero(z == hetero)
    c_x = {x: carbons[np.argmin(d[x, carbons])] for x in xs}
    for x in xs:
        hs = np.flatnonzero(z == 1)
        h = hs[np.argmin(d[x, hs])]
        cx = c_x[x]
        neighbours = [c for c in carbons if c != cx and d[cx, c] < 1.7]
        if len(xs) == 2:
            # resorcinol: the ring carbon bonded to both C-O carbons
            other = c_x[[y for y in xs if y != x][0]]
            ref = next(c for c in neighbours if d[c, other] < 1.7)
        else:
            ref = neighbours[0]
        quads.append((int(ref), int(cx), int(x), int(h)))
    return quads


def conformer_start(spec: dict, angles: tuple):
    atoms = frame(*spec["template"])
    for (a, b, c, h), angle in zip(torsions(atoms, spec["hetero"]), angles):
        atoms.set_dihedral(a, b, c, h, angle, indices=[h])
    return atoms


def kabsch_rmsd(p, q, idx=None):
    idx = np.arange(len(p)) if idx is None else idx
    p = p[idx] - p[idx].mean(0)
    q = q[idx] - q[idx].mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    s = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, s]) @ u.T
    return float(np.sqrt(((p @ r.T - q) ** 2).sum(1).mean()))


def internal_coordinates(atoms):
    """Bonds (< 1.9 A for heavy-heavy, < 1.25 A for X-H), bond angles, X-H torsions."""
    z = atoms.get_atomic_numbers()
    d = atoms.get_all_distances()
    n = len(atoms)
    bonds = [(i, j) for i in range(n) for j in range(i + 1, n)
             if d[i, j] < (1.25 if 1 in (z[i], z[j]) else 1.9)]
    nbr = {i: [j for b in bonds for j in b if i in b and j != i] for i in range(n)}
    angles = [(i, j, k) for j in range(n) for i in nbr[j] for k in nbr[j] if i < k]
    return bonds, angles


def compare_geometry(a, b, hetero):
    bonds, angles = internal_coordinates(a)
    heavy = np.flatnonzero(a.get_atomic_numbers() > 1)
    out = {
        "rmsd_all_A": kabsch_rmsd(a.get_positions(), b.get_positions()),
        "rmsd_heavy_A": kabsch_rmsd(a.get_positions(), b.get_positions(), heavy),
        "bond_mae_A": float(np.mean([abs(a.get_distance(i, j) - b.get_distance(i, j)) for i, j in bonds])),
        "bond_max_A": float(np.max([abs(a.get_distance(i, j) - b.get_distance(i, j)) for i, j in bonds])),
        "angle_mae_deg": float(np.mean([abs(a.get_angle(i, j, k) - b.get_angle(i, j, k)) for i, j, k in angles])),
        "angle_max_deg": float(np.max([abs(a.get_angle(i, j, k) - b.get_angle(i, j, k)) for i, j, k in angles])),
        "torsions_deg": [],
    }
    for q in torsions(a, hetero):
        ta, tb = a.get_dihedral(*q), b.get_dihedral(*q)
        out["torsions_deg"].append({"atoms": q, "a": ta, "b": tb, "diff": float((tb - ta + 180) % 360 - 180)})
    return out


# ----------------------------------------------------------------------------- calculators
class DFT:
    """PBE energy, forces (eV, eV/A) and dipole (e A) with density restarts."""

    def __init__(self, basis: str, xc: str = "pbe", gpu: bool = True):
        self.basis, self.xc, self.gpu = basis, xc, gpu
        self.dm, self.mol = None, None
        self.calls = 0

    def __call__(self, atoms):
        from pyscf import gto, scf

        mol = gto.M(atom=[(int(z), r) for z, r in zip(atoms.get_atomic_numbers(), atoms.get_positions())],
                    basis=self.basis, verbose=0)
        if self.gpu:
            from gpu4pyscf import dft
        else:
            from pyscf import dft
        mf = dft.RKS(mol, xc=self.xc)
        mf.grids.level = 3
        mf.conv_tol, mf.conv_tol_grad = 1e-11, 1e-7
        dm0 = None
        if self.dm is not None:
            dm0 = scf.addons.project_dm_nr2nr(self.mol, self.dm, mol) if self.mol.natm == mol.natm else None
        mf.kernel(dm0=dm0)
        if not mf.converged:
            mf.kernel()
        grad = mf.nuc_grad_method().kernel()
        grad = grad.get() if hasattr(grad, "get") else np.asarray(grad)
        try:
            dip = mf.dip_moment(unit="AU", verbose=0)
        except (AttributeError, NotImplementedError, TypeError):
            dip = mf.to_cpu().dip_moment(unit="AU", verbose=0)
        dip = dip.get() if hasattr(dip, "get") else np.asarray(dip)
        dm = mf.make_rdm1()
        self.dm, self.mol = (dm.get() if hasattr(dm, "get") else np.asarray(dm)), mol
        self.calls += 1
        return {
            "energy": float(mf.e_tot) * HARTREE_EV,
            "forces": -grad * HARTREE_EV / BOHR_A,
            "dipole": np.asarray(dip) * BOHR_A,
        }


class DenSNet:
    """Energy, forces (eV, eV/A) and dipole (e A) from a published DenSNet model."""

    def __init__(self, spec: dict, device: str = "cpu", np_dataset: str | None = None):
        import torch

        self.device = device
        if spec["loader"] == "calculator":
            from equiv_dens.md.dft_network_calculator import load_densnet_calculator

            self.calc = load_densnet_calculator(
                str(REPO / spec["run"]), args_file=str(REPO / spec["override"]),
                np_dataset=str(REPO / (np_dataset or spec["train_np"])), use_gpu=device == "cuda",
                repo=REPO, pyscf_grid=False,
            )
        else:
            from equiv_dens.data.density_dataset import AtomsDensityData
            from equiv_dens.md.dft_network_calculator import DenSNetCalculator
            from equiv_dens.training.model_loader import load_model
            from experiments.jacs_revision.analysis.rescore_full_density import checkpoint_args

            args = checkpoint_args(REPO / spec["run"], REPO / spec["override"])
            args.restart = str(REPO / spec["run"])
            args.use_gpu = device == "cuda"
            np_path = np_dataset or spec["train_np"]
            # With forces in the dataset, load_model rebuilds the training-set
            # force std that 2024 checkpoints do not store (scaling.std).
            # Without the training set (ethanol) nothing is rebuilt and the
            # validation reports the scale.
            dataset = AtomsDensityData(
                np_path=str(REPO / (np_path or spec["template"][0])), density_path=None,
                required_properties=["energy", "forces"] if np_path else [],
                orbitals_path=args.orbitals_file, radial_coeffs_file=args.radial_coeffs_file,
                atom_dens_path=args.atom_dens_path, atom_dens_type=args.atom_dens_type,
                cutoff=args.cutoff, dtype=torch.float32, df_loss_weights=args.df_loss_weights,
                density_n_samp=10**9,
            )
            if not np_path:
                args.ignore_missing_keywords = True
            model = load_model(args, dataset, train=False)
            import torch as _torch

            from experiments.jacs_revision.analysis.rescore_full_density import resolve_checkpoint

            saved = _torch.load(resolve_checkpoint(REPO / spec["run"]), map_location="cpu", weights_only=False)
            saved = saved.get("model_state_dict", saved)
            self.missing_keys = sorted(set(model.state_dict()) - set(saved))
            if set(self.missing_keys) - {"scaling.std"}:
                raise RuntimeError(f"checkpoint lacks {self.missing_keys[:5]}")
            self.calc = DenSNetCalculator(
                model, cutoff=args.cutoff, use_gpu=device == "cuda",
                energy_unit=getattr(args, "energy_unit_out", "kcal/mol"),
                grid_spec=dataset.grid_spec, grid_sampling_fn=dataset.sampling_fn, pyscf_grid=False,
                need_density_grid=False, atom_dens=dataset.atom_dens, atom_dens_type=args.atom_dens_type,
                remove_atom_density=args.remove_atom_density,
            )
        # The analytic dipole head (DipoleMomentIntorCalc) returns stale or wrong
        # dipoles after small displacements: it reuses int1e_r integrals within
        # 0.5 A (cache_integrals) and is unstable when the cache is off. Finite
        # differences need the quadrature dipole of the predicted density, which
        # DenSNetCalculator falls back to when the analytic head is absent.
        model = self.calc.model
        if "dipole_moment" in getattr(model, "property_models", {}):
            del model.property_models["dipole_moment"]
            model.no_force_props = [k for k in model.no_force_props if k != "dipole_moment"]
            model.force_props = [k for k in model.force_props if k != "dipole_moment"]
        self.scaling_std = float(self.calc.model.scaling.std) if hasattr(self.calc.model, "scaling") else None
        self.missing_keys = getattr(self, "missing_keys", [])
        # Multiplies energies and forces; 1 unless a force scale had to be fitted (ethanol).
        self.scale = float(spec.get("force_scale", 1.0))
        self.calls = 0

    def __call__(self, atoms, dipole: bool = True):
        from ase.calculators.calculator import all_changes

        props = ["energy", "forces"] + (["dipole"] if dipole else [])
        self.calc.calculate(atoms, props, all_changes)
        self.calls += 1
        res = {"energy": self.scale * float(self.calc.results["energy"]),
               "forces": self.scale * np.asarray(self.calc.results["forces"])}
        if dipole:
            res["dipole"] = np.asarray(self.calc.results["dipole"], dtype=float)
        return res


class Wrapped:
    """ASE calculator around a DFT or DenSNet callable (energy and forces)."""

    def __new__(cls, fn):
        from ase.calculators.calculator import Calculator, all_changes

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                res = fn(self.atoms) if not isinstance(fn, DenSNet) else fn(self.atoms, dipole=False)
                self.results = {"energy": res["energy"], "forces": res["forces"]}

        return _Calc()


# ----------------------------------------------------------------------------- vibrations
def finite_difference(fn, atoms, delta=0.005):
    """Hessian (eV/A^2) and dipole derivatives (e) by central differences."""
    n = len(atoms)
    ref = atoms.get_positions().copy()
    hess = np.zeros((3 * n, 3 * n))
    dmu = np.zeros((3 * n, 3))
    for i in range(n):
        for k in range(3):
            res = []
            for sign in (1, -1):
                pos = ref.copy()
                pos[i, k] += sign * delta
                atoms.set_positions(pos)
                res.append(fn(atoms))
            hess[3 * i + k] = -(res[0]["forces"] - res[1]["forces"]).ravel() / (2 * delta)
            dmu[3 * i + k] = (res[0]["dipole"] - res[1]["dipole"]) / (2 * delta)
    atoms.set_positions(ref)
    return 0.5 * (hess + hess.T), dmu


def harmonic(atoms, hess, dmu):
    """Frequencies (cm-1), mass-weighted modes, IR intensities (km/mol); rigid-body modes projected out."""
    from ase import units

    masses = atoms.get_masses()
    n = len(atoms)
    m = np.repeat(masses, 3)
    hmw = hess / np.sqrt(np.outer(m, m))
    # Translation and rotation vectors (mass-weighted), orthonormalised and projected out.
    x = atoms.get_positions() - atoms.get_center_of_mass()
    vecs = []
    for k in range(3):
        t = np.zeros((n, 3))
        t[:, k] = 1.0
        vecs.append((t * np.sqrt(masses)[:, None]).ravel())
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1.0
        vecs.append((np.cross(axis, x) * np.sqrt(masses)[:, None]).ravel())
    q, _ = np.linalg.qr(np.array(vecs).T)
    q = q[:, np.linalg.norm(np.array(vecs).T, axis=0) > 1e-8][:, :6]
    proj = np.eye(3 * n) - q @ q.T
    evals, evecs = np.linalg.eigh(proj @ hmw @ proj)
    order = np.argsort(np.abs(evals))[6:]
    evals, evecs = evals[order], evecs[:, order]
    idx = np.argsort(evals)
    evals, evecs = evals[idx], evecs[:, idx]
    # eV/(A^2 amu) -> cm-1
    conv = np.sqrt(units._e / units._amu) * 1e10 / (2 * np.pi * units._c * 100)
    freqs = np.sign(evals) * np.sqrt(np.abs(evals)) * conv
    # dmu/dQ (e A per A amu^0.5) for mass-weighted normal coordinate Q
    cart_modes = evecs / np.sqrt(m)[:, None]
    dmu_dq = cart_modes.T @ dmu
    intensity = KMMOL_PER_D2_A2_AMU * (EA_TO_DEBYE ** 2) * (dmu_dq ** 2).sum(1)
    return freqs, evecs, intensity


def pair_modes(modes_a, modes_b):
    from scipy.optimize import linear_sum_assignment

    overlap = np.abs(modes_a.T @ modes_b)
    rows, cols = linear_sum_assignment(-overlap)
    return cols, overlap[rows, cols]


# ----------------------------------------------------------------------------- commands
def validate(molecule: str, n_frames: int, device: str, gpu_dft: bool) -> dict:
    spec = MOLECULES[molecule]
    model = DenSNet(spec, device)
    rng = np.random.default_rng(0)
    if spec.get("test_np"):
        data = np.load(REPO / spec["test_np"], allow_pickle=True).item()
        idx = rng.choice(len(data["positions"]), n_frames, replace=False)
        frames = [frame(spec["test_np"], int(i)) for i in idx]
        ref_forces = [np.asarray(data["forces"][int(i)], dtype=float) * HARTREE_EV for i in idx]  # Ha/A -> eV/A
        source = f"{spec['test_np']} stored PBE/{spec['basis']} forces (Hartree/A)"
    else:
        from ase import Atoms

        md17 = np.load(REPO / "datasets/external/md17/md17_ethanol.npz")
        test_idx = np.load(REPO / "results/revision/overlap_v2/md17_ethanol_seed42_test_idx.npy")
        idx = rng.choice(test_idx, n_frames, replace=False)
        frames = [Atoms(numbers=md17["z"], positions=md17["R"][int(i)]) for i in idx]
        dft = DFT(spec["basis"], gpu=gpu_dft)
        ref_forces = [dft(a)["forces"] for a in frames]
        source = f"PySCF PBE/{spec['basis']} on MD17 frames (reconstructed test split)"
    pred = [model(a, dipole=False)["forces"] for a in frames]
    p, r = np.concatenate([x.ravel() for x in pred]), np.concatenate([x.ravel() for x in ref_forces])
    report = {
        "molecule": molecule, "n_frames": n_frames, "reference": source,
        "model": spec["run"], "scaling_std": model.scaling_std, "missing_keys": model.missing_keys,
        "force_scale_applied": model.scale,
        "force_mae_eV_A": float(np.abs(p - r).mean()),
        "force_mae_kcal_mol_A": float(np.abs(p - r).mean() * 23.060548),
        "force_rmse_eV_A": float(np.sqrt(((p - r) ** 2).mean())),
        "fit_slope": float(p @ r / (r @ r)),
        "force_mae_after_fit_kcal_mol_A": float(np.abs(p / (p @ r / (r @ r)) - r).mean() * 23.060548),
        "pearson_r": float(np.corrcoef(p, r)[0, 1]),
        "ref_force_std_eV_A": float(r.std()),
    }
    print(json.dumps(report, indent=1), flush=True)
    return report


def optimise(fn, atoms, fmax, steps, log):
    from ase.optimize import BFGS

    atoms = atoms.copy()
    atoms.calc = Wrapped(fn)
    opt = BFGS(atoms, logfile=str(log), maxstep=0.1)
    converged = opt.run(fmax=fmax, steps=steps)
    fmax_reached = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    return atoms, {"converged": bool(converged), "steps": int(opt.get_number_of_steps()),
                   "fmax_eV_A": fmax_reached, "energy_eV": float(atoms.get_potential_energy())}


def run(molecule: str, device: str, gpu_dft: bool, fmax: float, delta: float) -> dict:
    from ase.io import write

    spec = MOLECULES[molecule]
    out = OUT / molecule
    out.mkdir(parents=True, exist_ok=True)
    dft = DFT(spec["basis"], gpu=gpu_dft)
    model = DenSNet(spec, device)
    report = {"molecule": molecule, "level": f"PBE/{spec['basis']}", "model": spec["run"],
              "fmax_eV_A": fmax, "fd_delta_A": delta, "conformers": {}}
    if "scaling.std" in model.missing_keys:
        # No training set to rebuild the force std from: use the slope fitted
        # against DFT forces in `validate` for this same checkpoint.
        val = json.loads((OUT / f"validate_{molecule}.json").read_text())
        if val["model"] != spec["run"] or val.get("force_scale_applied", 1.0) != 1.0:
            raise SystemExit(f"validate_{molecule}.json is for another model or already scaled; rerun validate")
        model.scale = 1.0 / val["fit_slope"]
        report["force_scale_fitted"] = {"scale": model.scale, "from": f"validate_{molecule}.json",
                                        "pearson_r": val["pearson_r"], "n_frames": val["n_frames"]}
    for name, angles in spec["conformers"].items():
        t0 = time.time()
        start = conformer_start(spec, angles)
        write(out / f"{name}_start.xyz", start)
        d_min, d_opt = optimise(dft, start, fmax, 400, out / f"{name}_dft_opt.log")
        m_min, m_opt = optimise(model, start, fmax, 1000, out / f"{name}_densnet_opt.log")
        write(out / f"{name}_dft_min.xyz", d_min)
        write(out / f"{name}_densnet_min.xyz", m_min)
        entry = {
            "start_torsions_deg": list(angles),
            "dft_opt": d_opt, "densnet_opt": m_opt,
            "geometry": compare_geometry(d_min, m_min, spec["hetero"]),
            "densnet_fmax_at_dft_min_eV_A": float(np.linalg.norm(model(d_min, dipole=False)["forces"], axis=1).max()),
            "dft_fmax_at_densnet_min_eV_A": float(np.linalg.norm(dft(m_min)["forces"], axis=1).max()),
        }
        hd, mud = finite_difference(dft, d_min.copy(), delta)
        hm, mum = finite_difference(model, m_min.copy(), delta)
        fd, vd, idf = harmonic(d_min, hd, mud)
        fm, vm, idm = harmonic(m_min, hm, mum)
        perm, ovl = pair_modes(vd, vm)
        diff = fm[perm] - fd
        entry["frequencies"] = {
            "dft_cm1": fd.tolist(), "densnet_cm1_paired": fm[perm].tolist(), "overlap": ovl.tolist(),
            "dft_ir_km_mol": idf.tolist(), "densnet_ir_km_mol_paired": idm[perm].tolist(),
            "mae_cm1": float(np.abs(diff).mean()), "max_abs_cm1": float(np.abs(diff).max()),
            "mae_below_2000_cm1": float(np.abs(diff[fd < 2000]).mean()),
            "mae_above_2000_cm1": float(np.abs(diff[fd >= 2000]).mean()) if (fd >= 2000).any() else None,
            "mean_signed_cm1": float(diff.mean()),
            "sorted_mae_cm1": float(np.abs(np.sort(fm) - np.sort(fd)).mean()),
            "min_overlap": float(ovl.min()),
            "n_imaginary_dft": int((fd < 0).sum()), "n_imaginary_densnet": int((fm < 0).sum()),
        }
        np.savez(out / f"{name}_hessians.npz", dft_hessian=hd, dft_dmu=mud, densnet_hessian=hm, densnet_dmu=mum)
        entry["wall_s"] = time.time() - t0
        report["conformers"][name] = entry
        print(f"{molecule} {name}: rmsd heavy {entry['geometry']['rmsd_heavy_A']:.4f} all {entry['geometry']['rmsd_all_A']:.4f} A, "
              f"freq MAE {entry['frequencies']['mae_cm1']:.1f} cm-1 ({entry['wall_s']:.0f} s)", flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    # Conformer energy ordering relative to each method's own lowest conformer.
    names = list(report["conformers"])
    e_d = np.array([report["conformers"][c]["dft_opt"]["energy_eV"] for c in names])
    e_m = np.array([report["conformers"][c]["densnet_opt"]["energy_eV"] for c in names])
    report["conformer_energies_kcal_mol"] = {
        c: {"dft": float((e_d[i] - e_d.min()) * 23.060548), "densnet": float((e_m[i] - e_m.min()) * 23.060548)}
        for i, c in enumerate(names)
    }
    report["ordering_agrees"] = bool(list(np.argsort(e_d)) == list(np.argsort(e_m)))
    report["dft_calls"], report["densnet_calls"] = dft.calls, model.calls
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["conformer_energies_kcal_mol"], indent=1))
    return report


def redo_ir(molecule: str, device: str, delta: float) -> None:
    """Recompute DenSNet finite differences (forces, quadrature dipoles) at the saved minima."""
    from ase.io import read

    spec = MOLECULES[molecule]
    out = OUT / molecule
    report = json.loads((out / "report.json").read_text())
    model = DenSNet(spec, device)
    if report.get("force_scale_fitted"):
        model.scale = report["force_scale_fitted"]["scale"]
    for name, entry in report["conformers"].items():
        d_min = read(out / f"{name}_dft_min.xyz")
        m_min = read(out / f"{name}_densnet_min.xyz")
        old = np.load(out / f"{name}_hessians.npz")
        hm, mum = finite_difference(model, m_min.copy(), delta)
        fd, vd, idf = harmonic(d_min, old["dft_hessian"], old["dft_dmu"])
        fm, vm, idm = harmonic(m_min, hm, mum)
        perm, ovl = pair_modes(vd, vm)
        diff = fm[perm] - fd
        entry["frequencies"].update({
            "densnet_cm1_paired": fm[perm].tolist(), "overlap": ovl.tolist(),
            "densnet_ir_km_mol_paired": idm[perm].tolist(),
            "mae_cm1": float(np.abs(diff).mean()), "max_abs_cm1": float(np.abs(diff).max()),
            "mae_below_2000_cm1": float(np.abs(diff[fd < 2000]).mean()),
            "mae_above_2000_cm1": float(np.abs(diff[fd >= 2000]).mean()) if (fd >= 2000).any() else None,
            "mean_signed_cm1": float(diff.mean()), "min_overlap": float(ovl.min()),
            "n_imaginary_densnet": int((fm < 0).sum()),
            "densnet_dipole": "quadrature of the predicted density (analytic head bypassed)",
        })
        np.savez(out / f"{name}_hessians.npz", dft_hessian=old["dft_hessian"], dft_dmu=old["dft_dmu"],
                 densnet_hessian=hm, densnet_dmu=mum)
        print(f"{molecule} {name}: freq MAE {entry['frequencies']['mae_cm1']:.1f} cm-1, "
              f"IR sum DFT {idf.sum():.1f} DenSNet {idm[perm].sum():.1f} km/mol", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["validate", "run", "redo-ir"])
    ap.add_argument("--molecule", required=True, choices=list(MOLECULES))
    ap.add_argument("--device", default="cpu", help="DenSNet device")
    ap.add_argument("--cpu-dft", action="store_true", help="PySCF on the CPU instead of GPU4PySCF")
    ap.add_argument("--n-frames", type=int, default=20)
    ap.add_argument("--fmax", type=float, default=1e-3, help="eV/A")
    ap.add_argument("--delta", type=float, default=0.005, help="finite-difference step, A")
    cli = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if cli.command == "validate":
        rep = validate(cli.molecule, cli.n_frames, cli.device, not cli.cpu_dft)
        (OUT / f"validate_{cli.molecule}.json").write_text(json.dumps(rep, indent=1) + "\n")
    elif cli.command == "redo-ir":
        redo_ir(cli.molecule, cli.device, cli.delta)
    else:
        run(cli.molecule, cli.device, not cli.cpu_dft, cli.fmax, cli.delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
