#!/usr/bin/env python3
"""Figure 2 timing table: DenSNet / MACE-OFF / AIMNet2 / xTB / AIMD (R1.6, R3.2, R3.4).

Times a single-point energy+force call (and optional dipole) so the 755 ms
vs 5 min number can be regenerated from a script instead of copied twice.
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/analysis/benchmark_figure2.py'


import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
import sys

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
from experiments.jacs_revision.lib.dftb_calculator import default_dftb_paths, make_dftb3_3ob_calculator  # noqa: E402
from experiments.jacs_revision.lib.paths import repo_root  # noqa: E402
from experiments.jacs_revision.lib.triton_headers import (  # noqa: E402
    ensure_triton_python_headers,
    make_aimnet2_calculator,
    mlip_device,
)

ensure_triton_python_headers(start=_SCRIPT_DIR)


def _load_atoms(path: Path) -> Atoms:
    if path.suffix.lower() == ".xyz":
        return read(path, index=0)
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item() if data.ndim == 0 else data
    pos = np.asarray(data["positions"])
    anum = np.asarray(data["atom_numbers"])
    if pos.ndim == 3:
        pos = pos[0]
    if anum.ndim > 1:
        anum = anum[0]
    return Atoms(numbers=anum, positions=pos)


def _uncache(atoms):
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return
    if hasattr(calc, "reset"):
        calc.reset()
    elif hasattr(calc, "results"):
        calc.results = {}


def _time_call(fn, atoms=None, n_warm=1, n_rep=5):
    for _ in range(n_warm):
        if atoms is not None:
            _uncache(atoms)
        fn()
    ts = []
    for _ in range(n_rep):
        if atoms is not None:
            _uncache(atoms)
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"mean_s": float(np.mean(ts)), "std_s": float(np.std(ts)), "n": n_rep}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, default=Path("datasets/ethanol_train_10.xyz"))
    parser.add_argument("--methods", default="gxtb")
    parser.add_argument("--gxtb-bin", default="g-xtb/binary/gxtb")
    dftb_bin_default, dftb_sk_default = default_dftb_paths(_SCRIPT_DIR)
    parser.add_argument("--dftb-bin", type=Path, default=dftb_bin_default)
    parser.add_argument("--dftb-sk", type=Path, default=dftb_sk_default)
    parser.add_argument("--out", type=Path, default=Path("results/revision/figure2_timing.json"))
    parser.add_argument("--n-rep", type=int, default=3)
    parser.add_argument("--densnet-model", default="paper/models/ethanol/2024-03-22_96w7KyGG")
    parser.add_argument("--densnet-args", default="config/md/nn/ethanol_500ps.txt")
    parser.add_argument(
        "--densnet-np-dataset",
        default=None,
        help="Training-set npy used to size the orbital basis (must include every element)",
    )
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    atoms = _load_atoms(args.structure)
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)
    atoms.info.setdefault("external_field", [0.0, 0.0, 0.0])
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = {
        "structure": str(args.structure),
        "n_atoms": int(len(atoms)),
        "methods": {},
    }

    for method in methods:
        try:
            if method == "gxtb":
                sys.path.insert(0, str(repo_root() / "src"))
                from equiv_dens.md.gxtb_calculator import GxTBCalculator

                calc = GxTBCalculator(gxtb_command=args.gxtb_bin)
                atoms.calc = calc

                def _run():
                    atoms.get_potential_energy()
                    atoms.get_forces()

                results["methods"][method] = _time_call(_run, atoms=atoms, n_rep=args.n_rep)

            elif method in ("gfn2xtb", "tblite"):
                calc = None
                err = None
                try:
                    from tblite.ase import TBLite

                    calc = TBLite(method="GFN2-xTB")
                except Exception as exc:
                    err = f"tblite: {exc}"
                    try:
                        from xtb.ase.calculator import XTB

                        calc = XTB(method="GFN2-xTB")
                        err = None
                    except Exception as exc2:
                        err = f"{err}; xtb-python: {exc2}"
                if calc is None:
                    results["methods"][method] = {"error": err}
                    continue
                atoms.calc = calc

                def _run():
                    atoms.get_potential_energy()
                    atoms.get_forces()

                results["methods"][method] = _time_call(_run, atoms=atoms, n_rep=args.n_rep)

            elif method == "maceoff":
                from mace.calculators import mace_off

                device = mlip_device(args.use_gpu)
                atoms.calc = mace_off(model="medium", device=device)

                def _run():
                    atoms.get_potential_energy()
                    atoms.get_forces()

                results["methods"][method] = _time_call(_run, atoms=atoms, n_rep=args.n_rep)

            elif method == "macepolar":
                from mace.calculators import mace_polar

                device = mlip_device(args.use_gpu)
                atoms.calc = mace_polar(
                    model="polar-1-m",
                    device=device,
                    default_dtype="float64",
                )

                def _run():
                    atoms.get_potential_energy()
                    atoms.get_forces()

                results["methods"][method] = _time_call(_run, atoms=atoms, n_rep=args.n_rep)

            elif method == "aimd":
                results["methods"][method] = {
                    "note": "Use scripts/md/aimd_gpu4pyscf.py --steps 20 and divide wall time",
                }
            elif method == "densnet":
                import os
                import sys

                if not args.use_gpu:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                sys.path.insert(0, str(repo_root() / "src"))
                from equiv_dens.md.dft_network_calculator import load_densnet_calculator

                atoms.calc = load_densnet_calculator(
                    args.densnet_model,
                    args_file=args.densnet_args,
                    np_dataset=args.densnet_np_dataset,
                    use_gpu=args.use_gpu,
                )

                def _run():
                    atoms.get_potential_energy()
                    atoms.get_forces()

                results["methods"][method] = _time_call(_run, atoms=atoms, n_rep=args.n_rep)
                results["methods"][method]["model"] = args.densnet_model
            elif method == "so3lr":
                import os

                if not args.use_gpu:
                    os.environ.setdefault("JAX_PLATFORMS", "cpu")
                from so3lr import So3lrCalculator

                atoms.calc = So3lrCalculator(
                    calculate_stress=False,
                    lr_cutoff=1000.0,
                    dtype=np.float64,
                )

                def _run():
                    atoms.get_potential_energy()
                    atoms.get_forces()

                results["methods"][method] = _time_call(_run, atoms=atoms, n_rep=args.n_rep)
            elif method == "aimnet2":
                import os

                os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
                os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
                device = mlip_device(args.use_gpu)
                atoms.calc = make_aimnet2_calculator(device)

                def _run():
                    atoms.get_potential_energy()
                    atoms.get_forces()

                results["methods"][method] = _time_call(_run, atoms=atoms, n_rep=args.n_rep)
            elif method == "dftb":
                dftb_bin = Path(args.dftb_bin)
                if not dftb_bin.is_file():
                    results["methods"][method] = {
                        "error": f"DFTB+ binary not found: {dftb_bin}",
                    }
                    continue
                sk_dir = Path(args.dftb_sk)
                if not sk_dir.is_dir():
                    results["methods"][method] = {
                        "error": f"Slater–Koster directory not found: {sk_dir}",
                    }
                    continue
                command = f"{dftb_bin.resolve()} > PREFIX.out"
                atoms.calc = make_dftb3_3ob_calculator(atoms, command=command, sk_dir=sk_dir)

                def _run():
                    atoms.get_potential_energy()
                    atoms.get_forces()

                results["methods"][method] = _time_call(_run, atoms=atoms, n_rep=args.n_rep)
            else:
                results["methods"][method] = {"error": f"unknown method {method}"}
        except Exception as exc:
            results["methods"][method] = {"error": str(exc)}

        if method in results["methods"] and "mean_s" in results["methods"][method]:
            ms = results["methods"][method]["mean_s"]
            results["methods"][method]["ms_per_step"] = ms * 1000.0
            results["methods"][method]["ns_per_day_0.5fs"] = (0.0005 / ms) * 86400.0 / 1000.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
