#!/usr/bin/env python3
"""DenSNet minima hopping on a polythiophene oligomer (SO3krates protocol).

Usage:
  python scripts/revision/thiophene/minima_hopping_densnet.py \\
    --structure datasets/thiophene3mer_init.npy \\
    --model paper/models/polythiophene/2024-03-23_1XDL67zp_ext \\
    --np-dataset datasets/thiophene_all_train_d4.npy \\
    --n-escapes 10000 --out results/revision/thiophene_mh/minima.json
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/thiophene/minima_hopping_densnet.py'


import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)
_REPO_ROOT = repo_root()
sys.path.insert(0, str(_REPO_ROOT / "src"))

from experiments.jacs_revision.thiophene.minima_hopping import run_minima_hopping  # noqa: E402
from experiments.jacs_revision.thiophene.thiophene_collective_variables import (  # noqa: E402
    symbols_from_z,
    terthiophene_theta12_theta23,
)


def load_structure(path: Path) -> Atoms:
    if path.suffix.lower() == ".xyz":
        return read(path, index=0)
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item() if data.ndim == 0 else data
    if not isinstance(data, dict):
        raise ValueError(f"unsupported structure file {path}")
    pos = np.asarray(data["positions"], dtype=float)
    z = np.asarray(data["atom_numbers"])
    if pos.ndim == 3:
        pos = pos[0]
    if z.ndim == 2:
        z = z[0]
    mask = z > 0
    return Atoms(symbols=symbols_from_z(z), positions=pos[mask])


def serialize_minima(history: list, symbols: list[str]) -> list[dict]:
    rows = []
    for row in history:
        pos = np.asarray(row["positions"], dtype=float)
        try:
            t12, t23 = terthiophene_theta12_theta23(pos, symbols)
        except ValueError:
            t12, t23 = float("nan"), float("nan")
        rows.append(
            {
                "energy": float(row["energy"]),
                "step": int(row["step"]),
                "symbols": symbols,
                "positions": pos.tolist(),
                "theta12": t12,
                "theta23": t23,
            }
        )
    return rows


def write_minima_payload(path: Path, payload: dict, symbols: list[str], history: list) -> None:
    serializable = serialize_minima(history, symbols)
    payload = dict(payload)
    payload["n_unique"] = len(serializable)
    payload["minima"] = serializable
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)
    write(path.with_suffix(".xyz"), [Atoms(symbols=row["symbols"], positions=row["positions"]) for row in serializable])


def attach_densnet(atoms: Atoms, args: argparse.Namespace) -> None:
    from equiv_dens.md.dft_network_calculator import load_densnet_calculator

    calc = load_densnet_calculator(
        str(args.model),
        args_file=str(args.args_file) if args.args_file else None,
        np_dataset=str(args.np_dataset) if args.np_dataset else None,
        use_gpu=args.use_gpu,
        pyscf_grid=False,
    )
    atoms.calc = calc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=_REPO_ROOT / "paper/models/polythiophene/2024-03-23_1XDL67zp_ext",
    )
    parser.add_argument(
        "--args-file",
        type=Path,
        default=_REPO_ROOT / "config/md/nn/polythiophene_2mer_500ps.txt",
        help="DenSNet args (defaults to the working polythiophene MD config)",
    )
    parser.add_argument(
        "--np-dataset",
        type=Path,
        default=_REPO_ROOT / "datasets/thiophene_all_train_d4.npy",
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--n-escapes", type=int, default=10)
    parser.add_argument("--escape-steps", type=int, default=2000)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--temperature-k", type=float, default=1000.0)
    parser.add_argument("--e-diff-ev", type=float, default=2.0)
    parser.add_argument("--fmax", type=float, default=1e-4)
    parser.add_argument("--relax-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "results/revision/thiophene_mh/minima.json",
    )
    parser.add_argument(
        "--mock-harmonic",
        action="store_true",
        help="use a harmonic well instead of DenSNet (tests / dry runs)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    atoms = load_structure(args.structure)
    if args.mock_harmonic:
        from ase.calculators.calculator import Calculator, all_changes

        ref = atoms.get_positions().copy()

        class _Harmonic(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                delta = atoms.get_positions() - ref
                self.results = {
                    "energy": 0.5 * 8.0 * float(np.sum(delta * delta)),
                    "forces": -8.0 * delta,
                }

        atoms.calc = _Harmonic()
    else:
        attach_densnet(atoms, args)

    symbols = atoms.get_chemical_symbols()
    start_step = 0
    history = None
    temperature_k = args.temperature_k
    e_diff_ev = args.e_diff_ev
    prior_log: list = []
    if args.resume and args.out.is_file():
        saved = json.loads(args.out.read_text())
        history = [
            {
                "positions": np.asarray(row["positions"], dtype=float),
                "energy": float(row["energy"]),
                "step": int(row["step"]),
            }
            for row in saved.get("minima", [])
        ]
        prior_log = list(saved.get("log", []))
        start_step = int(prior_log[-1]["step"]) if prior_log else 0
        temperature_k = float(saved.get("final_temperature_k", args.temperature_k))
        e_diff_ev = float(saved.get("final_e_diff_ev", args.e_diff_ev))
        print(f"resuming from step {start_step} with {len(history)} unique minima")

    def _checkpoint(step, hist, log, t_k, e_diff):
        if step and args.checkpoint_every and step % args.checkpoint_every != 0:
            return
        write_minima_payload(
            args.out,
            {
                "structure": str(args.structure),
                "final_temperature_k": t_k,
                "final_e_diff_ev": e_diff,
                "log": prior_log + log,
            },
            symbols,
            hist,
        )
        print(f"checkpoint step {step}: {len(hist)} unique minima", flush=True)

    result = run_minima_hopping(
        atoms,
        n_escapes=args.n_escapes,
        escape_steps=args.escape_steps,
        timestep_fs=args.timestep_fs,
        temperature_k=temperature_k,
        e_diff_ev=e_diff_ev,
        fmax=args.fmax,
        relax_steps=args.relax_steps,
        seed=args.seed,
        start_step=start_step,
        history=history,
        on_step=_checkpoint,
    )
    write_minima_payload(
        args.out,
        {
            "structure": str(args.structure),
            "final_temperature_k": result["final_temperature_k"],
            "final_e_diff_ev": result["final_e_diff_ev"],
            "log": prior_log + result["log"],
        },
        symbols,
        result["minima"],
    )
    print(f"wrote {args.out} ({result['n_unique']} unique minima)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
