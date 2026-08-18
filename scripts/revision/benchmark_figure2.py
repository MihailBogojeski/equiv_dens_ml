#!/usr/bin/env python3
"""Figure 2 timing table: DenSNet / MACE-OFF / AIMNet2 / xTB / AIMD (R1.6, R3.2, R3.4).

Times a single-point energy+force call (and optional dipole) so the 755 ms
vs 5 min number can be regenerated from a script instead of copied twice.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read


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
    parser.add_argument("--out", type=Path, default=Path("results/revision/figure2_timing.json"))
    parser.add_argument("--n-rep", type=int, default=3)
    parser.add_argument("--densnet-model", default="paper/models/ethanol/2024-03-22_96w7KyGG")
    parser.add_argument("--densnet-args", default="config/md/nn/ethanol_500ps.txt")
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    atoms = _load_atoms(args.structure)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = {
        "structure": str(args.structure),
        "n_atoms": int(len(atoms)),
        "methods": {},
    }

    for method in methods:
        try:
            if method == "gxtb":
                import sys

                sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
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

                device = "cuda" if args.use_gpu else "cpu"
                atoms.calc = mace_off(model="medium", device=device)

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
                    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
                from equiv_dens.md.dft_network_calculator import load_densnet_calculator

                atoms.calc = load_densnet_calculator(
                    args.densnet_model,
                    args_file=args.densnet_args,
                    use_gpu=args.use_gpu,
                )

                def _run():
                    atoms.get_potential_energy()
                    atoms.get_forces()

                results["methods"][method] = _time_call(_run, atoms=atoms, n_rep=args.n_rep)
                results["methods"][method]["model"] = args.densnet_model
            elif method == "so3lr":
                from so3lr import So3lrCalculator

                device = "cuda" if args.use_gpu else "cpu"
                atoms.calc = So3lrCalculator(device=device)

                def _run():
                    atoms.get_potential_energy()
                    atoms.get_forces()

                results["methods"][method] = _time_call(_run, atoms=atoms, n_rep=args.n_rep)
            elif method == "aimnet2":
                import os

                os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
                os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
                try:
                    from aimnet.calculators import AIMNet2ASE

                    atoms.calc = AIMNet2ASE("aimnet2")
                except Exception:
                    try:
                        from aimnet.calculators.ase import AIMNet2ASE

                        atoms.calc = AIMNet2ASE("aimnet2")
                    except Exception as exc:
                        results["methods"][method] = {"error": str(exc)}
                        continue

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
