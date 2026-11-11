#!/usr/bin/env python3
"""Per-step cost of DenSNet against ML force fields, semi-empirical methods and DFT.

Response to R1.6 / R2.2 / R3.2 / R3.4. Every method is timed on the same
geometries (ethanol 9, resorcinol 14, water n=6 18, water n=12 36, thiophene
6-mer 44 and 12-mer 86 atoms) with the same protocol: at least 10 warm-up calls,
then at least 100 timed calls, each a full energy + force evaluation from new
coordinates, synchronised on the GPU. Median and interquartile range are
reported; the mean is kept only for reference.

Each (system, method, device) runs in its own process (``worker``), started by
``drive`` with the thread count pinned through OMP/MKL/OpenBLAS/torch/XLA
settings and ``taskset``, so JAX, PyTorch and the Fortran codes cannot share or
oversubscribe cores.

DenSNet is timed three ways:
  densnet          energy + forces (what an MLFF step costs)
  densnet_dipole   energy + forces + dipole (an IR step)
  densnet_parts    the IR step split into input/neighbour list, SAD prior,
                   density network, energy head with forces, dipole head, output
The water systems add ``densnet_direct`` (the direct-learning model, no SAD
prior), which is the R3.4 direct-versus-delta cost comparison.

DFT reference: ``dft_gpu`` is PBE/aug-cc-pVDZ + D4 with density fitting
(aug-cc-pVDZ-JKFIT) in GPU4PySCF, SCF and gradient on the GPU, D4 from dftd4,
with the AIMD script's thresholds (conv_tol 1e-5, conv_tol_grad 1e-3) and the
previous step's density as the guess, on geometries displaced by 0.005 A per
call as in an MD step. ``aimd_paper`` times scripts/md/aimd_gpu4pyscf.py as the
paper used it (its D4 wrapper re-runs the SCF and the gradient on the CPU).
Both use fewer calls than the ML methods (a single call takes seconds to
minutes); the counts are recorded.

Usage:
  python experiments/jacs_revision/analysis/benchmark_step_timing.py drive \\
      --devices cuda,cpu --threads 16 --out results/revision/timing_v2/timing.json
  python experiments/jacs_revision/analysis/benchmark_step_timing.py worker \\
      --system water_n6 --method densnet_parts --device cuda --threads 16 --out /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

# (geometry source, selector): an index, or ("natoms", n) for the first frame of that size.
SYSTEMS = {
    "ethanol": ("datasets/ethanol_train_10.xyz", 0),
    "resorcinol": ("datasets/resorcinol_augccpvdz_test.npy", 0),
    "water_n6": ("datasets/revision/water_clusters/id_test.xyz", ("natoms", 18)),
    "water_n12": ("datasets/revision/water_clusters/ood_size.xyz", ("natoms", 36)),
    "thiophene_6mer": ("datasets/thiophene6mer_test_d4.npy", 0),
    "thiophene_12mer": ("datasets/thiophene12mer_test.xyz", 0),
    # Larger water systems, DenSNet only, to show how the stage costs scale past
    # the ~100 atoms where the GPU stops being launch-latency bound.
    "water_n24": ("datasets/revision/water_ood/ood_order.xyz", ("natoms", 72)),
    "water_n30": ("datasets/revision/water_ood_rackers/ood_rackers.xyz", ("natoms", 90)),
    # four n=30 clusters 30 A apart (timing only; no cross-cluster neighbours)
    "water_4x30": ("datasets/revision/water_ood_rackers/ood_rackers.xyz", ("tile", 90, 4, 30.0)),
}

_WATER = ("results/revision/water_pbe_d4_avdz_energy/2026-09-14_Dw6lXobF", None,
          "datasets/revision/water_labels/water_train_pbe_d4_avdz_base.npy")
_WATER_DIRECT = ("results/revision/water_direct_pbe_d4_avdz_energy/2026-09-14_tSBmCFsJ", None,
                 "datasets/revision/water_labels/water_train_pbe_d4_avdz_base.npy")
_THIOPHENE = ("paper/models/polythiophene/2024-03-23_1XDL67zp_ext", "config/md/nn/polythiophene_12mer.txt",
              "datasets/thiophene_all_train_d4.npy")
# system -> (run directory, MD args file or None for the run's own args.txt, training np_dataset)
DENSNET = {
    # The 96w7KyGG run referenced by config/md/nn/ethanol.txt does not reproduce
    # PBE/cc-pVDZ forces with the current code (Pearson r 0.75); the _ext run does
    # (r 0.9999, see results/revision/equilibrium_v2/validate_ethanol.json).
    "ethanol": ("paper/models/ethanol/2024-04-18_5K9Ne7it_ext", "config/md/nn/ethanol_ext_paths.txt", None),
    "resorcinol": ("paper/models/resorcinol/2024-03-18_Ozf6CkNF_ext1", "config/md/nn/resorcinol.txt",
                   "datasets/resorcinol_combo_kmeansidx-1000_train.npy"),
    "water_n6": _WATER,
    "water_n12": _WATER,
    "thiophene_6mer": _THIOPHENE,
    "thiophene_12mer": _THIOPHENE,
    "water_n24": _WATER,
    "water_n30": _WATER,
    "water_4x30": _WATER,
}
DENSNET_DIRECT = {k: _WATER_DIRECT for k in ("water_n6", "water_n12", "water_n24", "water_n30", "water_4x30")}

ML_METHODS = ["densnet", "densnet_dipole", "densnet_parts", "densnet_direct", "densnet_direct_parts",
              "density_grid", "density_grid_direct", "so3lr", "maceoff", "macepolar", "aimnet2"]
CPU_ONLY = ["gfn2xtb", "gxtb", "dftb"]
GPU_ONLY = ["dft_gpu"]


# ----------------------------------------------------------------------------- helpers
def pin_threads(n: int) -> None:
    """Thread counts for every runtime a method might use; call before heavy imports."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"
    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={n}"
    )
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
    os.environ.setdefault("WANDB_MODE", "offline")


def load_system(name: str):
    import numpy as np
    from ase import Atoms
    from ase.io import read

    path, sel = SYSTEMS[name]
    path = REPO / path
    if path.suffix == ".xyz":
        frames = read(path, index=":")
    else:
        data = np.load(path, allow_pickle=True)
        data = data.item() if data.dtype == object and data.ndim == 0 else data
        frames = []
        for i in range(min(len(data["positions"]), 50)):
            z = np.asarray(data["atom_numbers"][i] if np.ndim(data["atom_numbers"]) > 1 else data["atom_numbers"], dtype=int)
            r = np.asarray(data["positions"][i], dtype=float)[: len(z)]
            frames.append(Atoms(numbers=z[z > 0], positions=r[z > 0]))
    if isinstance(sel, tuple) and sel[0] == "tile":
        _, natoms, copies, spacing = sel
        unit = next(a for a in frames if len(a) == natoms)
        atoms = unit.copy()
        for k in range(1, copies):
            shifted = unit.copy()
            shifted.positions += np.array([k * spacing, 0.0, 0.0])
            atoms += shifted
    elif isinstance(sel, tuple):
        atoms = next(a for a in frames if len(a) == sel[1])
    else:
        atoms = frames[sel]
    atoms = Atoms(numbers=atoms.get_atomic_numbers(), positions=atoms.get_positions())
    atoms.info.update(charge=0, spin=1, external_field=[0.0, 0.0, 0.0])
    return atoms


def summarise(times: list[float]) -> dict:
    import numpy as np

    t = np.asarray(times)
    return {
        "median_s": float(np.median(t)),
        "p25_s": float(np.percentile(t, 25)),
        "p75_s": float(np.percentile(t, 75)),
        "iqr_s": float(np.percentile(t, 75) - np.percentile(t, 25)),
        "mean_s": float(t.mean()),
        "std_s": float(t.std()),
        "min_s": float(t.min()),
        "max_s": float(t.max()),
        "n_timed": int(len(t)),
    }


class Stepper:
    """Moves the atoms a little each call so no calculator can reuse its last answer."""

    def __init__(self, atoms, amplitude=0.005, seed=0):
        import numpy as np

        self.atoms = atoms
        self.ref = atoms.get_positions().copy()
        self.rng = np.random.default_rng(seed)
        self.amplitude = amplitude

    def next(self):
        self.atoms.set_positions(self.ref + self.amplitude * self.rng.standard_normal(self.ref.shape))
        return self.atoms


def time_calls(fn, stepper, n_warmup, n_timed, sync, budget_s=900.0, min_timed=5):
    """Warm-up, then timed calls. Calls taking seconds to minutes (g-xTB and DFT
    on the 12-mer) stop at the time budget once ``min_timed`` calls are in;
    the achieved counts are returned with the times."""
    t_start = time.perf_counter()
    done_warmup = 0
    for _ in range(n_warmup):
        fn(stepper.next())
        sync()
        done_warmup += 1
        if time.perf_counter() - t_start > budget_s / 3:
            break
    times = []
    t_start = time.perf_counter()
    for _ in range(n_timed):
        atoms = stepper.next()
        sync()
        t0 = time.perf_counter()
        fn(atoms)
        sync()
        times.append(time.perf_counter() - t0)
        if len(times) >= min_timed and time.perf_counter() - t_start > budget_s:
            break
    return times, done_warmup


# ----------------------------------------------------------------------------- methods
def make_ase_step(calc, properties=("energy", "forces")):
    from ase.calculators.calculator import all_changes

    def step(atoms):
        calc.calculate(atoms, list(properties), all_changes)
        return calc.results

    return step


# load_densnet_calculator cannot build the 2024 ethanol (_ext) and resorcinol
# models (their free-atom prior files do not fit its energy-head construction).
# They are loaded the way the full-density re-score loads them (architecture
# from the checkpoint's pickled args, which build no analytic dipole head), and
# their dipole is the quadrature of the predicted density on the level-2 grid.
SPLINE_PRIOR_MODELS = {"ethanol", "resorcinol"}


def _equiv_dens_path():
    import equiv_dens
    return str(Path(equiv_dens.__file__).resolve().parent)


def _apply_compile_mode(calc):
    """DENSNET_COMPILE=dynamic: torch.compile of the representation and force region with dynamic shapes
    (DFTNetwork.enable_compile); static: fixed shapes with CUDA graphs and padded neighbor pairs, what
    --static_compile does in MD. Unset / empty: eager. The first warm-up calls carry the compilation.
    EQUIV_DENS_SRC selects the source tree (default: this repo's src); the run's JSON records which one loaded."""
    mode = os.environ.get("DENSNET_COMPILE", "").strip().lower()
    if mode == "dynamic":
        calc.model.enable_compile(static=False)
    elif mode == "static":
        calc.static_compile = True
        calc.model.enable_compile(static=True)
    elif mode:
        raise SystemExit(f"DENSNET_COMPILE must be dynamic, static or empty, not {mode!r}")
    return mode or "eager"


def densnet_calculator_spline(system: str, device: str):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, os.environ.get("EQUIV_DENS_SRC", str(REPO / "src")))
    import torch

    from equiv_dens.md.dft_network_calculator import DenSNetCalculator
    from equiv_dens.training.model_loader import load_model
    from experiments.jacs_revision.analysis.rescore_full_density import checkpoint_args, make_dataset

    run, override, _ = DENSNET[system]
    args = checkpoint_args(REPO / run, REPO / override)
    args.restart = str(REPO / run)
    args.use_gpu = device == "cuda"
    args.ignore_missing_keywords = True
    geometry = {"ethanol": "datasets/revision/ood_geom_labels/ood_modes_ethanol_pbe_d4_avdz",
                "resorcinol": "datasets/revision/ood_geom_labels/ood_modes_resorcinol_pbe_d4_avdz"}[system]
    dataset = make_dataset(args, str(REPO / f"{geometry}_base.npy"), str(REPO / f"{geometry}.npy"), "auto", False, 2)
    model = load_model(args, dataset, train=False)
    grid_spec = dataset.grid_spec
    if device == "cuda":
        grid_spec = {k: (v[0].cuda(), v[1].cuda()) for k, v in grid_spec.items()}
    calc = DenSNetCalculator(
        model, cutoff=args.cutoff, use_gpu=device == "cuda",
        energy_unit=getattr(args, "energy_unit_out", "kcal/mol"),
        grid_spec=grid_spec, grid_sampling_fn=dataset.sampling_fn, pyscf_grid=False,
        need_density_grid=False, atom_dens=dataset.atom_dens, atom_dens_type=args.atom_dens_type,
        remove_atom_density=args.remove_atom_density,
    )
    mode = _apply_compile_mode(calc)
    info = {"compile": mode, "equiv_dens": _equiv_dens_path(),
            "model": run, "args_file": f"checkpoint args + {override}", "remove_atom_density": calc.remove_atom_density,
            "atom_dens_type": calc.atom_dens_type, "cutoff": calc.cutoff,
            "dipole_head": "dipole_moment" in getattr(calc.model, "property_models", {}),
            "note": "checkpoint-args loader: dipole from the density on the level-2 grid"}
    return calc, info


def densnet_calculator(system: str, device: str, direct: bool = False):
    sys.path.insert(0, os.environ.get("EQUIV_DENS_SRC", str(REPO / "src")))
    from equiv_dens.md.dft_network_calculator import load_densnet_calculator

    if system in SPLINE_PRIOR_MODELS and not direct:
        return densnet_calculator_spline(system, device)
    run, args_file, np_dataset = (DENSNET_DIRECT if direct else DENSNET)[system]
    calc = load_densnet_calculator(
        str(REPO / run),
        args_file=str(REPO / args_file) if args_file else None,
        np_dataset=str(REPO / np_dataset) if np_dataset else None,
        use_gpu=device == "cuda",
        repo=REPO,
        pyscf_grid=False,
    )
    mode = _apply_compile_mode(calc)
    info = {"model": run, "args_file": args_file, "remove_atom_density": calc.remove_atom_density,
            "atom_dens_type": calc.atom_dens_type, "cutoff": calc.cutoff,
            "dipole_head": "dipole_moment" in getattr(calc.model, "property_models", {}),
            "compile": mode, "equiv_dens": _equiv_dens_path()}
    return calc, info


def densnet_parts_step(calc, device):
    """The DenSNetCalculator IR step, stage by stage (mirrors DFTNetwork.forward)."""
    import numpy as np
    import torch

    from equiv_dens.md.dft_network_calculator import geometry_to_model_inputs
    from equiv_dens.nn.dft_network import _skip_grid_property
    from equiv_dens.utils import orbitals

    cuda = device == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    model = calc.model
    parts: dict[str, list[float]] = {k: [] for k in
                                     ("inputs", "sad_prior", "density_network", "energy_forward", "force_backward",
                                      "dipole", "output")}

    def step(atoms):
        marks = [time.perf_counter()]

        def mark():
            sync()
            marks.append(time.perf_counter())

        atom_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)[None, :]
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32,
                                 device="cuda" if cuda else "cpu").unsqueeze(0)
        inputs = geometry_to_model_inputs(atom_numbers, positions, calc.cutoff, cuda)
        mark()
        if calc.remove_atom_density and calc.atom_dens is not None and calc.atom_dens_type in {"mo_coeffs", "df_coeffs"}:
            key = "atom_" + calc.atom_dens_type
            inputs[key], inputs[key + "_occ"], inputs[key + "_basis"] = orbitals.join_atom_coeffs(
                torch.as_tensor(atom_numbers, dtype=torch.long), calc.atom_dens, calc.atom_dens_type
            )
        mark()
        data = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
        if model.calculate_forces:
            data["positions"].requires_grad_(True)
        data = model.density_repr_model(data)
        mark()
        # The energy head normally takes -dE/dR inside its own forward. Switch that
        # off to time the forward alone, then take the same gradient as its own
        # stage. The backward runs through the energy head and, because the
        # coefficients depend on the positions, through the density network too.
        heads = [model.property_models[key] for key in model.force_props]
        saved = [head.calculate_forces for head in heads]
        for head in heads:
            head.calculate_forces = False
        try:
            for key in model.force_props:
                data = model.property_models[key](data)
        finally:
            for head, flag in zip(heads, saved):
                head.calculate_forces = flag
        mark()
        data["forces"] = -torch.autograd.grad(torch.sum(data["energy"]), data["positions"])[0]
        mark()
        data["positions"].requires_grad_(False)
        grid_dipole = "dipole_moment" not in model.property_models
        if grid_dipole:
            # No analytic head (spline prior): attach the level-2 grid and the
            # free-atom density on it, expand the density there, integrate r*rho.
            grid = calc._add_density_grid({}, atom_numbers, positions)
            data.update(grid)
        with torch.no_grad():
            for key in model.no_force_props:
                if _skip_grid_property(key, data):
                    continue
                data = model.property_models[key](data)
                if key == "density" and model.remove_atom_density:
                    data["density"] = data["density"] + data["atom_density"]
            if grid_dipole:
                packed = orbitals.calc_dipole_moment({
                    "density": data["density"], "coord_weights": data["coord_weights"], "coords": data["coords"],
                    "batch_positions": data["batch_positions"], "batch_atom_numbers": data["batch_atom_numbers"],
                })
                data["dipole_moment"] = packed["dipole_moment"]
        mark()
        data = model.scaling.transform_back(data)
        data = model.conversions_out(data)
        out = {"energy": data["energy"].detach().cpu().numpy(), "forces": data["forces"].detach().cpu().numpy()}
        if "dipole_moment" in data:
            out["dipole"] = data["dipole_moment"].detach().cpu().numpy()
        mark()
        for name, a, b in zip(parts, marks[:-1], marks[1:]):
            parts[name].append(b - a)
        return out

    return step, parts


def density_grid_step(calc, device):
    """Real-space density on the level-2 grid: grid, SAD on grid, network, density head.

    An MD step needs no grid (energies, forces and analytic dipoles come from
    coefficients), so the SAD prior there is a coefficient lookup. Producing
    rho(r) itself is where a delta model pays for its prior; this measures it.
    """
    import numpy as np
    import torch

    from equiv_dens.md.dft_network_calculator import geometry_to_model_inputs
    from equiv_dens.utils import orbitals

    cuda = device == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    model = calc.model
    if "density" not in model.property_models:
        raise RuntimeError("model has no density head")
    parts: dict[str, list[float]] = {k: [] for k in ("grid", "sad_on_grid", "density_network", "density_head")}
    info = {}

    def step(atoms):
        atom_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)[None, :]
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32,
                                 device="cuda" if cuda else "cpu").unsqueeze(0)
        inputs = geometry_to_model_inputs(atom_numbers, positions, calc.cutoff, cuda)
        if calc.remove_atom_density and calc.atom_dens is not None and calc.atom_dens_type in {"mo_coeffs", "df_coeffs"}:
            key = "atom_" + calc.atom_dens_type
            inputs[key], inputs[key + "_occ"], inputs[key + "_basis"] = orbitals.join_atom_coeffs(
                torch.as_tensor(atom_numbers, dtype=torch.long), calc.atom_dens, calc.atom_dens_type)
        sync()
        t = [time.perf_counter()]
        coords, weights = calc.grid_sampling_fn(calc.grid_spec, 10_000_000_000, atom_numbers, positions)
        coords, weights = coords.to(positions.device), weights.to(positions.device)
        sync()
        t.append(time.perf_counter())
        atom_density = None
        if calc.remove_atom_density and calc.atom_dens is not None:
            atom_density, _ = orbitals.sample_atom_density(
                positions=positions.view(-1, atom_numbers.shape[1], 3).cpu(),
                atom_numbers=torch.tensor(atom_numbers).type(torch.long), coords=coords.cpu(),
                atom_dens_type=calc.atom_dens_type, atom_dens_dict=calc.atom_dens)
            atom_density = atom_density.to(positions)
        sync()
        t.append(time.perf_counter())
        with torch.no_grad():
            data = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}
            data = model.density_repr_model(data)
            sync()
            t.append(time.perf_counter())
            data["coords"], data["coord_weights"] = coords, weights
            if atom_density is not None:
                data["atom_density"] = atom_density
            data = model.property_models["density"](data)
            rho = data["density"] + (data["atom_density"] if model.remove_atom_density else 0)
            rho = rho.cpu()
        sync()
        t.append(time.perf_counter())
        for name, a, b in zip(parts, t[:-1], t[1:]):
            parts[name].append(b - a)
        info["n_grid"] = int(coords.shape[1])
        return {"density": rho}

    return step, parts, info


def mlip_step(method: str, device: str):
    import numpy as np

    info = {}
    if method == "so3lr":
        if device == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
        import jax
        from so3lr import So3lrCalculator

        info["jax_devices"] = [str(d) for d in jax.devices()]
        if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError(f"JAX has no GPU backend here: {jax.devices()}")
        calc = So3lrCalculator(calculate_stress=False, lr_cutoff=1000.0, dtype=np.float64)
        info["settings"] = "lr_cutoff=1000 A, float64 (as in the first timing)"
    elif method == "maceoff":
        from mace.calculators import mace_off

        calc = mace_off(model="medium", device=device)
        info["settings"] = "MACE-OFF23 medium, default dtype"
    elif method == "macepolar":
        from mace.calculators import mace_polar

        calc = mace_polar(model="polar-1-m", device=device, default_dtype="float64")
        info["settings"] = "MACE-POLAR-1-M, float64"
    elif method == "aimnet2":
        sys.path.insert(0, str(REPO))
        from experiments.jacs_revision.lib.triton_headers import make_aimnet2_calculator

        calc = make_aimnet2_calculator(device)
        info["settings"] = "AIMNet2, no torch.compile"
    else:
        raise ValueError(method)
    return make_ase_step(calc), info


def semiempirical_step(method: str, workdir: Path):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, os.environ.get("EQUIV_DENS_SRC", str(REPO / "src")))
    info = {}
    if method == "gfn2xtb":
        from tblite.ase import TBLite

        calc = TBLite(method="GFN2-xTB", verbosity=0)
        info["settings"] = "tblite GFN2-xTB, in process"
    elif method == "gxtb":
        from equiv_dens.md.gxtb_calculator import GxTBCalculator

        calc = GxTBCalculator(gxtb_command=str(REPO / "g-xtb/binary/gxtb"))
        info["settings"] = "g-xTB binary, one process per call (includes start-up)"
    elif method == "dftb":
        from experiments.jacs_revision.lib.dftb_calculator import default_dftb_paths, make_dftb3_3ob_calculator

        dftb_bin, sk_dir = default_dftb_paths(REPO / "experiments/jacs_revision/analysis")
        os.chdir(workdir)
        calc = None

        def build(atoms):
            return make_dftb3_3ob_calculator(atoms, command=f"{Path(dftb_bin).resolve()} > PREFIX.out", sk_dir=sk_dir)

        info["settings"] = "DFTB+ DFTB3/3ob, one process per call (includes start-up and file I/O)"
        state = {}

        def step(atoms):
            if "calc" not in state:
                state["calc"] = build(atoms)
            from ase.calculators.calculator import all_changes

            state["calc"].calculate(atoms, ["energy", "forces"], all_changes)
            return state["calc"].results

        return step, info
    else:
        raise ValueError(method)
    return make_ase_step(calc), info


def dft_gpu_step():
    """GPU4PySCF PBE/aug-cc-pVDZ(+JKFIT) SCF and gradient on the GPU, D4 from dftd4."""
    import numpy as np
    from dftd4.interface import DampingParam, DispersionModel
    from gpu4pyscf import dft
    from pyscf import gto, scf

    state = {"dm": None, "mol": None}
    info = {"settings": "gpu4pyscf RKS PBE/aug-cc-pVDZ, DF aug-cc-pVDZ-JKFIT, conv_tol 1e-5, conv_tol_grad 1e-3, "
                        "previous density as guess; D4(PBE) energy+gradient from dftd4", "scf_cycles": []}

    def step(atoms):
        mol = gto.M(atom=[(int(z), p) for z, p in zip(atoms.get_atomic_numbers(), atoms.get_positions())],
                    basis="augccpvdz", verbose=0)
        mf = dft.RKS(mol, xc="pbe").density_fit(auxbasis="augccpvdzjkfit")
        mf.conv_tol, mf.conv_tol_grad = 1e-5, 1e-3
        dm0 = None
        if state["dm"] is not None:
            dm0 = scf.addons.project_dm_nr2nr(state["mol"], state["dm"], mol)
        cycles = {"n": 0}

        def count(envs):
            cycles["n"] += 1

        mf.callback = count
        mf.kernel(dm0=dm0)
        grad = mf.nuc_grad_method().kernel()
        grad = grad.get() if hasattr(grad, "get") else np.asarray(grad)
        disp = DispersionModel(atoms.get_atomic_numbers(), mol.atom_coords()).get_dispersion(
            DampingParam(method="pbe"), grad=True
        )
        dm = mf.make_rdm1()
        state["dm"], state["mol"] = (dm.get() if hasattr(dm, "get") else np.asarray(dm)), mol
        info["scf_cycles"].append(cycles["n"])
        return {"energy": float(mf.e_tot) + disp["energy"], "gradient": grad + disp["gradient"]}

    return step, info


def aimd_paper_step():
    sys.path.insert(0, str(REPO / "scripts/md"))
    from aimd_gpu4pyscf import PySCFCalculator

    calc = PySCFCalculator(use_gpu=True, verbose=0)
    info = {"settings": "scripts/md/aimd_gpu4pyscf.py PySCFCalculator as used for the paper's AIMD timings "
                        "(GPU SCF, then dftd4.pyscf re-runs SCF and gradient on the CPU)"}
    return make_ase_step(calc), info


# ----------------------------------------------------------------------------- worker / driver
def worker(cli) -> int:
    # absolute before the DFTB branch changes directory, or the JSON lands in its work dir
    cli.out = str(Path(cli.out).resolve())
    pin_threads(cli.threads)
    import numpy as np
    import torch

    torch.set_num_threads(cli.threads)
    torch.set_num_interop_threads(1)
    cuda = cli.device == "cuda"
    if cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    atoms = load_system(cli.system)
    stepper = Stepper(atoms)
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    result = {"system": cli.system, "n_atoms": len(atoms), "method": cli.method, "device": cli.device,
              "threads": cli.threads, "n_warmup": cli.n_warmup, "host": platform.node()}
    if cuda:
        result["gpu"] = torch.cuda.get_device_name(0)
    t_setup = time.perf_counter()
    parts = None
    m = cli.method
    if m in ("densnet", "densnet_dipole", "densnet_direct"):
        calc, info = densnet_calculator(cli.system, cli.device, direct=m == "densnet_direct")
        props = ("energy", "forces", "dipole") if m == "densnet_dipole" else ("energy", "forces")
        step = make_ase_step(calc, props)
    elif m in ("densnet_parts", "densnet_direct_parts"):
        calc, info = densnet_calculator(cli.system, cli.device, direct=m == "densnet_direct_parts")
        step, parts = densnet_parts_step(calc, cli.device)
    elif m in ("density_grid", "density_grid_direct"):
        calc, info = densnet_calculator(cli.system, cli.device, direct=m == "density_grid_direct")
        step, parts, grid_info = density_grid_step(calc, cli.device)
        info = {**info, "grid_info": grid_info}
    elif m in ("so3lr", "maceoff", "macepolar", "aimnet2"):
        step, info = mlip_step(m, cli.device)
    elif m in CPU_ONLY:
        work = Path(cli.out).with_suffix("").parent / f"work_{cli.system}_{m}"
        work.mkdir(parents=True, exist_ok=True)
        step, info = semiempirical_step(m, work)
    elif m == "dft_gpu":
        step, info = dft_gpu_step()
    elif m == "aimd_paper":
        step, info = aimd_paper_step()
    else:
        raise SystemExit(f"unknown method {m}")
    result["setup_s"] = time.perf_counter() - t_setup
    result.update(info)
    times, n_warm = time_calls(step, stepper, cli.n_warmup, cli.n_timed, sync, budget_s=cli.budget)
    result["n_warmup_done"] = n_warm
    result.update(summarise(times))
    result["times_s"] = times
    if parts is not None:
        # Drop the warm-up calls from the stage records.
        result["parts"] = {k: summarise(v[n_warm:]) for k, v in parts.items()}
        result["parts_sum_median_s"] = float(sum(p["median_s"] for p in result["parts"].values()))
    if cuda:
        result["max_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("times_s",)}, indent=1))
    return 0


def plan(devices, systems, include_aimd_paper):
    runs = []
    for system in systems:
        for device in devices:
            for m in ML_METHODS:
                if (m.startswith("densnet_direct") or m == "density_grid_direct") and system not in DENSNET_DIRECT:
                    continue
                runs.append((system, m, device))
            if device == "cpu":
                runs += [(system, m, "cpu") for m in CPU_ONLY]
            if device == "cuda":
                runs.append((system, "dft_gpu", "cuda"))
                if include_aimd_paper:
                    runs.append((system, "aimd_paper", "cuda"))
    return runs


def counts(method: str, system: str, cli):
    if method == "dft_gpu":
        return 2, 10 if system != "thiophene_12mer" else 5
    if method == "aimd_paper":
        return 0, 2 if system != "thiophene_12mer" else 1
    return cli.n_warmup, cli.n_timed


def drive(cli) -> int:
    out = Path(cli.out)
    runs_dir = out.parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    devices = cli.devices.split(",")
    systems = cli.systems.split(",") if cli.systems else list(SYSTEMS)
    runs = plan(devices, systems, cli.aimd_paper)
    if cli.methods:
        keep = set(cli.methods.split(","))
        runs = [r for r in runs if r[1] in keep]
    # The first `threads` cores of this job's own allocation (Slurm cgroups do
    # not necessarily start at core 0).
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < cli.threads:
        raise SystemExit(f"only {len(allowed)} cores allocated, {cli.threads} requested")
    cores = ",".join(str(c) for c in allowed[: cli.threads])
    results = []
    for system, method, device in runs:
        target = runs_dir / f"{system}__{method}__{device}.json"
        if target.is_file() and not cli.force:
            results.append(json.loads(target.read_text()))
            continue
        n_w, n_t = counts(method, system, cli)
        cmd = ["taskset", "-c", cores, sys.executable, "-B", __file__, "worker", "--system", system,
               "--method", method, "--device", device, "--threads", str(cli.threads),
               "--n-warmup", str(n_w), "--n-timed", str(n_t), "--budget", str(cli.budget), "--out", str(target)]
        env = dict(os.environ)
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["JAX_PLATFORMS"] = "cpu"
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=cli.timeout)
        log = runs_dir / f"{system}__{method}__{device}.log"
        log.write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:])
        if proc.returncode == 0 and target.is_file():
            res = json.loads(target.read_text())
            print(f"{system:16s} {method:22s} {device:5s} median {res['median_s'] * 1e3:10.2f} ms "
                  f"IQR {res['iqr_s'] * 1e3:8.2f} ms  ({time.time() - t0:.0f} s)", flush=True)
        else:
            res = {"system": system, "method": method, "device": device,
                   "error": (proc.stderr or proc.stdout)[-1500:]}
            print(f"{system:16s} {method:22s} {device:5s} FAILED (see {log})", flush=True)
        results.append(res)
    summary = {"host": platform.node(), "threads": cli.threads, "devices": devices,
               "protocol": f">= {cli.n_warmup} warm-up then >= {cli.n_timed} timed calls (DFT: fewer, recorded)",
               "runs": [{k: v for k, v in r.items() if k != "times_s"} for r in results]}
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--system", required=True, choices=list(SYSTEMS))
    w.add_argument("--method", required=True)
    w.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    w.add_argument("--threads", type=int, default=16)
    w.add_argument("--n-warmup", type=int, default=10)
    w.add_argument("--n-timed", type=int, default=100)
    w.add_argument("--budget", type=float, default=900.0, help="seconds of timed calls before stopping early")
    w.add_argument("--out", required=True)
    d = sub.add_parser("drive")
    d.add_argument("--devices", default="cuda,cpu")
    d.add_argument("--systems", default="")
    d.add_argument("--methods", default="", help="comma-separated subset of methods (default: all)")
    d.add_argument("--threads", type=int, default=16)
    d.add_argument("--n-warmup", type=int, default=10)
    d.add_argument("--n-timed", type=int, default=100)
    d.add_argument("--aimd-paper", action="store_true", help="also time the paper's AIMD calculator")
    d.add_argument("--timeout", type=int, default=7200)
    d.add_argument("--budget", type=float, default=900.0)
    d.add_argument("--force", action="store_true")
    d.add_argument("--out", required=True)
    cli = ap.parse_args()
    return worker(cli) if cli.cmd == "worker" else drive(cli)


if __name__ == "__main__":
    raise SystemExit(main())
