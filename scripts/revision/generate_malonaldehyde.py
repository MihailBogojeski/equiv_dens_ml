#!/usr/bin/env python3
"""Build the malonaldehyde proton-transfer splits.

The co-author's note singles this molecule out because one calculation answers
two reviewer concerns at once: it has a strong intramolecular hydrogen bond with
large density redistribution, and its proton-transfer coordinate is a physically
meaningful reaction coordinate along which "distance from the training set" is
something a chemist can read off a plot rather than an abstract descriptor
distance.

The design is a deliberate extrapolation. Training samples the enol basin at
300 K, where the transferring proton stays bound to one oxygen. The test set is
a relaxed scan that walks the proton across to the other oxygen, through the
symmetric transition state that the training set never visits. Error against the
scan coordinate

    delta = r(O_donor-H) - r(H-O_acceptor)

is then a direct readout of how the model degrades as configurations leave the
training distribution, with delta = 0 the symmetric shared-proton structure.

Geometries come from GFN2-xTB, which is only ever used to *place* atoms; every
frame is labelled afterwards at the campaign's DFT level.

Usage:
  python scripts/revision/generate_malonaldehyde.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from generate_ood_water import write_xyz  # noqa: E402

SMILES_ENOL = "OC=CC=O"


def _calculator():
    from tblite.ase import TBLite

    return TBLite(method="GFN2-xTB", verbosity=0)


def build_enol():
    """Optimised cis-enol malonaldehyde, plus the proton-transfer atom indices."""
    from ase import Atoms
    from ase.optimize import BFGS
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES_ENOL))
    params = AllChem.ETKDGv3()
    params.randomSeed = 20260819
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.calc = _calculator()
    BFGS(atoms, logfile=None).run(fmax=0.02, steps=500)

    idx = proton_transfer_indices(atoms)
    return atoms, idx


def proton_transfer_indices(atoms) -> dict[str, int]:
    """Locate (donor O, transferring H, acceptor O) from the geometry itself.

    Reading these off the RDKit atom order would silently break if the embedding
    ever reordered atoms, so they are identified structurally: the hydroxyl
    hydrogen is the only H bonded to an O, and the acceptor is the other oxygen.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    oxygens = [i for i, s in enumerate(symbols) if s == "O"]
    hydrogens = [i for i, s in enumerate(symbols) if s == "H"]
    if len(oxygens) != 2:
        raise ValueError(f"expected 2 oxygens, found {len(oxygens)}")

    best = None
    for h in hydrogens:
        for o in oxygens:
            dist = float(np.linalg.norm(positions[h] - positions[o]))
            if dist < 1.2 and (best is None or dist < best[0]):
                best = (dist, h, o)
    if best is None:
        raise ValueError("no hydroxyl hydrogen found; is this the enol tautomer?")
    _, h_idx, donor = best
    acceptor = oxygens[0] if oxygens[1] == donor else oxygens[1]
    return {"donor": donor, "hydrogen": h_idx, "acceptor": acceptor}


def delta_coordinate(positions: np.ndarray, idx: dict[str, int]) -> float:
    r_donor = np.linalg.norm(positions[idx["hydrogen"]] - positions[idx["donor"]])
    r_acceptor = np.linalg.norm(positions[idx["hydrogen"]] - positions[idx["acceptor"]])
    return float(r_donor - r_acceptor)


def sample_enol_basin(
    atoms,
    idx: dict[str, int],
    n_frames: int,
    temperature_k: float,
    steps_between: int,
    seed: int,
    delta_max: float,
    max_sweeps: int = 40,
):
    """Langevin sampling restricted to one enol well.

    The barrier is only about 150 meV, so at 300 K the proton transfers on a
    picosecond timescale and an unfiltered trajectory visits both wells. That
    would be fatal here: the training set would cover the very region the test
    set is supposed to probe, and the experiment would measure nothing. Frames
    with delta above `delta_max` are therefore discarded, leaving a training set
    confined to the reactant basin and a clean gap up to the transition state.

    Returns the kept frames and the count discarded, so the filtering is
    reported rather than hidden.
    """
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    work = atoms.copy()
    work.calc = _calculator()
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(work, temperature_K=temperature_k, rng=rng)
    dyn = Langevin(work, 0.5 * units.fs, temperature_K=temperature_k, friction=0.02, rng=rng)
    dyn.run(2000)  # discard the initial transient before collecting

    frames: list[tuple[list[str], np.ndarray]] = []
    n_rejected = 0
    for _ in range(max_sweeps * n_frames):
        if len(frames) >= n_frames:
            break
        dyn.run(steps_between)
        positions = work.get_positions().copy()
        if delta_coordinate(positions, idx) <= delta_max:
            frames.append((work.get_chemical_symbols(), positions))
        else:
            n_rejected += 1
    if len(frames) < n_frames:
        raise RuntimeError(
            f"only {len(frames)}/{n_frames} frames stayed below delta={delta_max}; "
            "lower --temperature or raise --train-delta-max"
        )
    return frames, n_rejected


def relaxed_scan(atoms, idx: dict[str, int], targets: np.ndarray):
    """Relax the molecule at each fixed donor-H distance.

    Constraining one bond and relaxing everything else is what makes this a
    reaction path rather than a straight-line interpolation: the heavy-atom
    frame contracts as the proton moves, which is the structural response that
    makes the transition state interesting in the first place.
    """
    from ase.constraints import FixBondLength
    from ase.optimize import BFGS

    frames, records = [], []
    work = atoms.copy()
    for target in targets:
        positions = work.get_positions()
        vec = positions[idx["hydrogen"]] - positions[idx["donor"]]
        current = np.linalg.norm(vec)
        positions[idx["hydrogen"]] = positions[idx["donor"]] + vec / current * target
        work.set_positions(positions)

        work.calc = _calculator()
        work.set_constraint(FixBondLength(idx["donor"], idx["hydrogen"]))
        BFGS(work, logfile=None).run(fmax=0.05, steps=300)
        work.set_constraint()

        snapshot = work.get_positions().copy()
        frames.append((work.get_chemical_symbols(), snapshot))
        records.append(
            {
                "r_donor_h": float(target),
                "delta": delta_coordinate(snapshot, idx),
                "energy_gfn2_ev": float(work.get_potential_energy()),
            }
        )
    return frames, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "datasets/revision/malonaldehyde")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--steps-between", type=int, default=50)
    parser.add_argument(
        "--train-delta-max",
        type=float,
        default=-0.40,
        help="keep only training frames with delta below this, i.e. one basin",
    )
    parser.add_argument("--scan-min", type=float, default=0.98)
    parser.add_argument("--scan-max", type=float, default=1.60)
    parser.add_argument("--scan-points", type=int, default=25)
    parser.add_argument("--scan-jitter-per-point", type=int, default=4)
    parser.add_argument("--scan-jitter-sigma", type=float, default=0.03)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("optimising the enol minimum with GFN2-xTB")
    atoms, idx = build_enol()
    d0 = delta_coordinate(atoms.get_positions(), idx)
    print(f"  donor={idx['donor']} H={idx['hydrogen']} acceptor={idx['acceptor']}  delta0={d0:+.3f} A")

    manifest: dict = {"seed": args.seed, "indices": idx, "delta_minimum": d0, "splits": {}}

    print(f"sampling the enol basin at {args.temperature:.0f} K, keeping delta <= {args.train_delta_max:+.2f}")
    pool, n_rejected = sample_enol_basin(
        atoms,
        idx,
        args.n_train + args.n_val,
        args.temperature,
        args.steps_between,
        args.seed,
        args.train_delta_max,
    )
    print(f"  discarded {n_rejected} frames that had crossed the barrier")
    manifest["train_delta_max"] = args.train_delta_max
    manifest["n_rejected_crossed_barrier"] = n_rejected
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))
    train = [pool[i] for i in order[: args.n_train]]
    val = [pool[i] for i in order[args.n_train :]]

    for name, frames in (("train", train), ("val", val)):
        path = args.output_dir / f"{name}.xyz"
        deltas = [delta_coordinate(np.asarray(p), idx) for _, p in frames]
        write_xyz(
            path,
            frames,
            [f"malonaldehyde {name} delta={d:+.3f}" for d in deltas],
        )
        manifest["splits"][name] = {
            "xyz": str(path),
            "n_frames": len(frames),
            "delta_min": min(deltas),
            "delta_max": max(deltas),
        }
        print(f"  {name}: {len(frames)} frames, delta in [{min(deltas):+.3f}, {max(deltas):+.3f}]")

    print(f"relaxed scan over r(O-H) in [{args.scan_min}, {args.scan_max}]")
    targets = np.linspace(args.scan_min, args.scan_max, args.scan_points)
    scan_frames, scan_records = relaxed_scan(atoms, idx, targets)

    # A bare scan is a one-dimensional line through configuration space; a few
    # jittered copies per point give the test set some width so the reported
    # error at each delta is an average rather than a single lucky geometry.
    frames, comments, deltas = [], [], []
    for (symbols, positions), record in zip(scan_frames, scan_records):
        frames.append((symbols, positions))
        comments.append(f"malonaldehyde scan delta={record['delta']:+.3f}")
        deltas.append(record["delta"])
        for _ in range(args.scan_jitter_per_point):
            jittered = positions + rng.normal(0.0, args.scan_jitter_sigma, size=positions.shape)
            delta = delta_coordinate(jittered, idx)
            frames.append((symbols, jittered))
            comments.append(f"malonaldehyde scan-jitter delta={delta:+.3f}")
            deltas.append(delta)

    path = args.output_dir / "ood_proton_transfer.xyz"
    write_xyz(path, frames, comments)
    manifest["splits"]["ood_proton_transfer"] = {
        "xyz": str(path),
        "n_frames": len(frames),
        "delta_min": min(deltas),
        "delta_max": max(deltas),
        "scan": scan_records,
    }
    print(f"  scan: {len(frames)} frames, delta in [{min(deltas):+.3f}, {max(deltas):+.3f}]")

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
