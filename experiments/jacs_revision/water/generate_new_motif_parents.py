#!/usr/bin/env python3
"""Phase B: new water/ice/droplet/malonaldehyde parents, then the same MD 10×.

Water-cluster isomers are not an RDKit problem. New parents come from an
expanded analytic MOTIFS table, from GFN2 (or fake) random-pack + LBFGS
minima, and from extra genice2 polymorphs / carve seeds. Malonaldehyde adds
more ETKDG embeddings and a denser relaxed scan. CREST is used if it is on
``$PATH`` and never required.

Writes ``*_newmotifs/`` so the original 3k campaign and the Phase A ``*_md10x``
trees stay intact. Dedup is heavy-atom RMSD plus a CV-space floor.

Usage:
  python scripts/revision/water/generate_new_motif_parents.py --all \\
    --water-out datasets/revision/water_clusters_newmotifs
"""

from __future__ import annotations

from pathlib import Path as _CompatPath
_LEGACY_FILE = next(p for p in _CompatPath(__file__).resolve().parents
                    if (p / "pyproject.toml").is_file()) / 'scripts/revision/water/generate_new_motif_parents.py'


import argparse
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path


_SCRIPT_DIR = _LEGACY_FILE.resolve().parent
_LIB_DIR = _SCRIPT_DIR if _SCRIPT_DIR.name == "lib" else _SCRIPT_DIR.parent / "lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from experiments.jacs_revision.lib.paths import ensure_revision_imports, repo_root  # noqa: E402

ensure_revision_imports(_SCRIPT_DIR)

import numpy as np

from experiments.jacs_revision.water.generate_ood_water import (  # noqa: E402
    build_droplet_split,
    build_ice_split,
    write_xyz,
)
from experiments.jacs_revision.water.generate_water_cluster_geoms import (  # noqa: E402
    EXTRA_MOTIFS,
    MOTIFS,
    OO,
    _valid,
    _vec,
    _water_hydrogens,
    _write_xyz,
)
from experiments.jacs_revision.water.md_sampling import Frame, choose_calculator, dedup_frames, heavy_atom_rmsd  # noqa: E402

_REPO_ROOT = repo_root()


def extra_motifs() -> dict[int, dict[str, object]]:
    """Analytic isomers that are not in the original campaign ``MOTIFS`` table."""
    return EXTRA_MOTIFS


def n_waters_fallback(symbols: list[str]) -> int:
    """Oxygen count when a frame meta does not record n."""
    n_o = sum(1 for s in symbols if s.upper() == "O")
    return n_o if n_o else max(1, len(symbols) // 3)


def _ideal_water(oxygen, rng: np.random.Generator) -> list[tuple[str, tuple[float, float, float]]]:
    # Random donation axis; second H at the experimental HOH angle.
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    up = rng.normal(size=3)
    o = (float(oxygen[0]), float(oxygen[1]), float(oxygen[2]))
    h1, h2 = _water_hydrogens(o, tuple(axis), tuple(up))
    return [("O", o), ("H", h1), ("H", h2)]


def random_pack_waters(
    n: int,
    rng: np.random.Generator | None = None,
    radius: float | None = None,
    min_oo: float = 2.45,
    max_tries: int = 4000,
) -> tuple[list[str], np.ndarray]:
    """Packmol-like random waters in a sphere. No DFT; used as an opt starting point."""
    rng = rng if rng is not None else np.random.default_rng()
    if n < 1:
        raise ValueError("n must be >= 1")
    if radius is None:
        radius = max(2.4, 1.7 * n ** (1.0 / 3.0) * (OO / 2.0))
    oxygens: list[np.ndarray] = []
    tries = 0
    while len(oxygens) < n and tries < max_tries:
        tries += 1
        direction = rng.normal(size=3)
        direction = direction / (np.linalg.norm(direction) or 1.0)
        trial = direction * (radius * rng.random() ** (1.0 / 3.0))
        if any(float(np.linalg.norm(trial - other)) < min_oo for other in oxygens):
            continue
        oxygens.append(trial)
    if len(oxygens) < n:
        raise RuntimeError(f"could only pack {len(oxygens)}/{n} waters in r={radius:.2f} A")
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for oxygen in oxygens:
        atoms.extend(_ideal_water(oxygen, rng))
    symbols = [s for s, _ in atoms]
    coords = np.asarray([p for _, p in atoms], dtype=float)
    return symbols, coords


def keep_distinct_minima(
    frames: list[Frame],
    rmsd_cutoff: float = 0.35,
    energy_tol: float = 1e-3,
) -> list[Frame]:
    """Keep packings that differ in heavy-atom RMSD or in energy."""
    kept: list[Frame] = []
    for frame in frames:
        energy = float(frame.meta.get("energy", math.nan))
        duplicate = False
        for other in kept:
            if len(other.symbols) != len(frame.symbols):
                continue
            rmsd = heavy_atom_rmsd(frame.coords, other.coords, frame.symbols)
            other_e = float(other.meta.get("energy", math.nan))
            same_energy = np.isfinite(energy) and np.isfinite(other_e) and abs(energy - other_e) <= energy_tol
            if rmsd < rmsd_cutoff and (same_energy or not np.isfinite(energy)):
                duplicate = True
                break
        if not duplicate:
            kept.append(frame)
    return kept


def _optimize_frame(symbols: list[str], coords: np.ndarray, calculator, fmax: float = 0.05) -> tuple[np.ndarray, float]:
    from ase import Atoms
    from ase.optimize import LBFGS
    from experiments.jacs_revision.water.md_sampling import attach_calculator

    atoms = Atoms(symbols=symbols, positions=coords)
    attach_calculator(atoms, calculator)
    LBFGS(atoms, logfile=None).run(fmax=fmax, steps=400)
    energy = float(atoms.get_potential_energy())
    return atoms.get_positions().copy(), energy


def _jitter_atoms(atoms, sigma: float, rng: random.Random):
    out = []
    for symbol, pos in atoms:
        out.append(
            (
                symbol,
                (
                    pos[0] + rng.gauss(0.0, sigma),
                    pos[1] + rng.gauss(0.0, sigma),
                    pos[2] + rng.gauss(0.0, sigma),
                ),
            )
        )
    return out


def _sample_table(table: dict, n: int, n_frames: int, sigmas: tuple[float, ...], rng: random.Random) -> list[Frame]:
    motifs = table.get(n, {})
    if not motifs:
        return []
    names = list(motifs)
    frames: list[Frame] = []
    attempts = 0
    while len(frames) < n_frames and attempts < n_frames * 40:
        attempts += 1
        name = names[len(frames) % len(names)]
        base = motifs[name]()
        sigma = sigmas[len(frames) % len(sigmas)]
        trial = _jitter_atoms(base, sigma, rng)
        if not _valid(trial):
            continue
        symbols = [s for s, _ in trial]
        coords = np.asarray([p for _, p in trial], dtype=float)
        frames.append(Frame(symbols, coords, f"n={n} {name} sigma={sigma}", {"n": n, "isomer": name, "sigma": sigma}))
    return frames


def _maybe_crest_on_xyz(xyz_path: Path, workdir: Path) -> list[Frame]:
    crest = shutil.which("crest")
    if crest is None:
        return []
    workdir.mkdir(parents=True, exist_ok=True)
    dest = workdir / "parent.xyz"
    dest.write_text(xyz_path.read_text().split("\n\n")[0] if False else "".join(xyz_path.read_text().splitlines(True)[:]))
    # CREST wants a single-frame XYZ. Take the first frame.
    lines = xyz_path.read_text().splitlines()
    nat = int(lines[0].strip())
    (workdir / "crest_in.xyz").write_text("\n".join(lines[: nat + 2]) + "\n")
    proc = subprocess.run(
        [crest, "crest_in.xyz", "--gfn2", "--noreftopo"],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    ensemble = workdir / "crest_conformers.xyz"
    if proc.returncode != 0 or not ensemble.is_file():
        return []
    from experiments.jacs_revision.water.md_sampling import read_xyz_symbol_frames

    return [Frame(s, c, comment, {"source": "crest"}) for s, c, comment in read_xyz_symbol_frames(ensemble)]


def _build_malon_parents(n_etkdg: int, scan_points: int, jitter_per_point: int, seed: int, optimize: bool) -> dict[str, list[Frame]]:
    from experiments.jacs_revision.water.generate_malonaldehyde import (
        SMILES_ENOL,
        delta_coordinate,
        proton_transfer_indices,
        relaxed_scan,
        sample_enol_basin,
    )

    frames_train: list[Frame] = []
    frames_scan: list[Frame] = []
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from ase import Atoms
    except ImportError:
        return {"train": [], "val": [], "ood_proton_transfer": []}

    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES_ENOL))
    embeddings = []
    for i in range(n_etkdg):
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed + i)
        if AllChem.EmbedMolecule(mol, params) != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        positions = np.array([list(conf.GetAtomPosition(j)) for j in range(mol.GetNumAtoms())], dtype=float)
        embeddings.append((symbols, positions))

    if optimize:
        try:
            calc = choose_calculator("gfn2", "malon_train", 9)
        except Exception:
            calc = None
            optimize = False
    else:
        calc = None

    from ase.optimize import BFGS
    from experiments.jacs_revision.water.md_sampling import attach_calculator

    for i, (symbols, positions) in enumerate(embeddings):
        atoms = Atoms(symbols=symbols, positions=positions)
        if calc is not None:
            attach_calculator(atoms, calc)
            BFGS(atoms, logfile=None).run(fmax=0.05, steps=250)
        try:
            idx = proton_transfer_indices(atoms)
        except ValueError:
            continue
        coords = atoms.get_positions().copy()
        frames_train.append(
            Frame(
                list(atoms.get_chemical_symbols()),
                coords,
                f"malon etkdg={i} delta={delta_coordinate(coords, idx):+.3f}",
                {"source": "etkdg", "delta": delta_coordinate(coords, idx)},
            )
        )

    # Denser scan from the first successful embedding, when GFN2 is available.
    if embeddings and calc is not None:
        atoms = Atoms(symbols=embeddings[0][0], positions=embeddings[0][1])
        attach_calculator(atoms, calc)
        BFGS(atoms, logfile=None).run(fmax=0.05, steps=250)
        idx = proton_transfer_indices(atoms)
        targets = np.linspace(0.98, 1.60, scan_points)
        scan_frames, records = relaxed_scan(atoms, idx, targets)
        rng = np.random.default_rng(seed)
        for (symbols, positions), record in zip(scan_frames, records):
            frames_scan.append(
                Frame(symbols, positions, f"malon scan delta={record['delta']:+.3f}", {"source": "scan", **record})
            )
            for _ in range(jitter_per_point):
                jittered = positions + rng.normal(0.0, 0.03, size=positions.shape)
                frames_scan.append(
                    Frame(
                        symbols,
                        jittered,
                        f"malon scan-jitter delta={delta_coordinate(jittered, idx):+.3f}",
                        {"source": "scan_jitter"},
                    )
                )
        try:
            basin, _ = sample_enol_basin(atoms, idx, n_frames=max(20, n_etkdg), temperature_k=300.0, steps_between=40, seed=seed, delta_max=-0.40)
            for symbols, positions in basin:
                frames_train.append(
                    Frame(symbols, positions, "malon basin-md", {"source": "basin_md", "delta": delta_coordinate(positions, idx)})
                )
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frames_train))
    n_val = max(1, len(frames_train) // 6) if frames_train else 0
    val = [frames_train[i] for i in order[:n_val]]
    train = [frames_train[i] for i in order[n_val:]]
    return {"train": train, "val": val, "ood_proton_transfer": frames_scan}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water-out", type=Path, default=_REPO_ROOT / "datasets/revision/water_clusters_newmotifs")
    parser.add_argument("--ood-out", type=Path, default=_REPO_ROOT / "datasets/revision/water_ood_newmotifs")
    parser.add_argument("--malon-out", type=Path, default=_REPO_ROOT / "datasets/revision/malonaldehyde_newmotifs")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--water", action="store_true")
    parser.add_argument("--ice", action="store_true")
    parser.add_argument("--droplet", action="store_true")
    parser.add_argument("--malon", action="store_true")
    parser.add_argument("--train-per-size", type=int, default=250)
    parser.add_argument("--val-per-size", type=int, default=50)
    parser.add_argument("--id-test-per-size", type=int, default=50)
    parser.add_argument("--ood-per-size", type=int, default=100)
    parser.add_argument("--n-pack-per-size", type=int, default=40, help="random-pack attempts per n, before dedup")
    parser.add_argument("--optimize", choices=("none", "gfn2", "fake"), default="none")
    parser.add_argument("--polymorphs", default="Ih,XI,Ic,II,III,V,VI,IX")
    parser.add_argument("--ice-sizes", default="12,16,20,24")
    parser.add_argument("--ice-temperatures", default="50,150")
    parser.add_argument("--ice-per-combo", type=int, default=16)
    parser.add_argument("--ice-rep", type=int, default=4)
    parser.add_argument("--droplet-sizes", default="16,20,24")
    parser.add_argument("--droplet-densities", default="0.8,0.6,0.4,0.2")
    parser.add_argument("--droplet-per-combo", type=int, default=20)
    parser.add_argument("--droplet-melt-steps", type=int, default=2000)
    parser.add_argument("--n-etkdg", type=int, default=12)
    parser.add_argument("--scan-points", type=int, default=40)
    parser.add_argument("--scan-jitter-per-point", type=int, default=8)
    parser.add_argument("--genice-bin", default=str(_REPO_ROOT / ".venv/bin/genice2"))
    parser.add_argument("--rmsd-cutoff", type=float, default=0.20)
    parser.add_argument("--cv-floor", type=float, default=0.04)
    args = parser.parse_args()

    do_water = args.water or args.all
    do_ice = args.ice or args.all
    do_droplet = args.droplet or args.all
    do_malon = args.malon or args.all
    if not (do_water or do_ice or do_droplet or do_malon):
        raise SystemExit("nothing to do: pass --all or --water/--ice/--droplet/--malon")

    rng = random.Random(args.seed)
    manifest: dict = {"seed": args.seed, "splits": {}}

    if do_water:
        # Original isomers plus the new table, then random-pack minima.
        combined = {n: dict(MOTIFS.get(n, {})) for n in (2, 3, 4, 5, 6, 8, 10, 12)}
        for n, extra in EXTRA_MOTIFS.items():
            combined.setdefault(n, {}).update(extra)

        split_frames: dict[str, list[Frame]] = {"train": [], "val": [], "id_test": [], "ood_size": []}
        for n in (2, 3, 4, 5, 6):
            for split, count in (
                ("train", args.train_per_size),
                ("val", args.val_per_size),
                ("id_test", args.id_test_per_size),
            ):
                split_frames[split].extend(_sample_table(combined, n, count, (0.04, 0.06, 0.08), rng))
        for n in (8, 10, 12):
            split_frames["ood_size"].extend(_sample_table(combined, n, args.ood_per_size, (0.05, 0.07, 0.10), rng))

        calc = None
        if args.optimize != "none":
            try:
                calc = choose_calculator(args.optimize, "water", 18)
            except Exception as exc:
                print(f"optimize={args.optimize} unavailable ({exc}); packing without DFT")
                calc = None

        packed: list[Frame] = []
        for n in (2, 3, 4, 5, 6, 8, 10, 12):
            for k in range(args.n_pack_per_size):
                try:
                    symbols, coords = random_pack_waters(n, rng=np.random.default_rng(args.seed + 1000 * n + k))
                except RuntimeError:
                    continue
                energy = float("nan")
                if calc is not None:
                    try:
                        coords, energy = _optimize_frame(symbols, coords, calc)
                    except Exception:
                        continue
                packed.append(Frame(symbols, coords, f"random-pack n={n} k={k}", {"n": n, "source": "pack", "energy": energy}))
        packed = keep_distinct_minima(packed, rmsd_cutoff=0.35)
        # Packs are deduped against each other. Motif-jitter frames are not:
        # they follow the original generator (250 thermal draws per size) and
        # sit on new isomers. A global CV floor would collapse them back to
        # one frame per isomer.
        for frame in packed:
            n = int(frame.meta["n"])
            dest = "ood_size" if n >= 8 else "train"
            split_frames[dest].append(frame)

        args.water_out.mkdir(parents=True, exist_ok=True)
        for split, frames in split_frames.items():
            motif_frames = [f for f in frames if f.meta.get("source") != "pack"]
            pack_frames = [f for f in frames if f.meta.get("source") == "pack"]
            kept_packs, rejected_packs = dedup_frames(
                pack_frames,
                rmsd_cutoff=args.rmsd_cutoff,
                cv_floor=args.cv_floor,
                existing=motif_frames,
            )
            kept = motif_frames + kept_packs
            xyz = args.water_out / f"{split}.xyz"
            _write_xyz(xyz, [list(zip(f.symbols, [tuple(p) for p in f.coords])) for f in kept], f"new motifs {split}")
            by_n: dict[str, int] = {}
            for frame in kept:
                key = str(int(frame.meta.get("n") or n_waters_fallback(frame.symbols)))
                by_n[key] = by_n.get(key, 0) + 1
            manifest["splits"][f"water_{split}"] = {
                "xyz": str(xyz),
                "n_frames": len(kept),
                "by_n": dict(sorted(by_n.items(), key=lambda kv: int(kv[0]))),
                "n_motif": len(motif_frames),
                "n_pack": len(kept_packs),
                "n_rejected_pack_dedup": len(rejected_packs),
            }
            print(
                f"  water {split}: {len(kept)} parents "
                f"({len(motif_frames)} motif + {len(kept_packs)} pack, {len(rejected_packs)} pack-deduped)"
            )

        crest_frames = _maybe_crest_on_xyz(args.water_out / "train.xyz", args.water_out / "crest")
        if crest_frames:
            print(f"  CREST added {len(crest_frames)} conformers (not merged; inspect crest/)")

    if do_ice:
        polymorphs = [p.strip() for p in args.polymorphs.split(",") if p.strip()]
        sizes = [int(v) for v in args.ice_sizes.split(",")]
        temps = [float(v) for v in args.ice_temperatures.split(",")]
        print(f"building extra ice parents: {polymorphs} sizes={sizes}")
        frames, comments, meta = [], [], []
        for polymorph in polymorphs:
            try:
                part_f, part_c, part_m = build_ice_split(
                    [polymorph], sizes, temps, args.ice_per_combo, args.ice_rep, args.seed, args.genice_bin
                )
            except Exception as exc:
                print(f"  skip ice {polymorph}: {exc}")
                continue
            frames.extend(part_f)
            comments.extend(part_c)
            meta.extend(part_m)
        if frames:
            path = args.ood_out / "ood_order.xyz"
            write_xyz(path, frames, comments)
            manifest["splits"]["ood_order"] = {"xyz": str(path), "n_frames": len(frames), "n_meta": len(meta)}
            print(f"  ice: {len(frames)} parents -> {path}")

    if do_droplet:
        sizes = [int(v) for v in args.droplet_sizes.split(",")]
        densities = [float(v) for v in args.droplet_densities.split(",")]
        print(f"building extra droplets: sizes={sizes} densities={densities}")
        frames, comments, meta = build_droplet_split(
            sizes,
            densities,
            args.droplet_per_combo,
            args.droplet_melt_steps,
            330.0,
            args.seed,
            args.ice_rep,
            args.genice_bin,
        )
        path = args.ood_out / "ood_density.xyz"
        write_xyz(path, frames, comments)
        manifest["splits"]["ood_density"] = {"xyz": str(path), "n_frames": len(frames), "n_meta": len(meta)}
        print(f"  droplets: {len(frames)} parents -> {path}")

    if do_malon:
        print("building extra malonaldehyde embeddings / scan")
        groups = _build_malon_parents(
            args.n_etkdg,
            args.scan_points,
            args.scan_jitter_per_point,
            args.seed,
            optimize=args.optimize == "gfn2",
        )
        args.malon_out.mkdir(parents=True, exist_ok=True)
        for split, frames in groups.items():
            kept, rejected = dedup_frames(frames, rmsd_cutoff=0.08, cv_floor=None)
            path = args.malon_out / f"{split}.xyz"
            write_xyz(path, [(f.symbols, f.coords) for f in kept], [f.comment for f in kept])
            manifest["splits"][f"malon_{split}"] = {
                "xyz": str(path),
                "n_frames": len(kept),
                "n_rejected_dedup": len(rejected),
            }
            print(f"  malon {split}: {len(kept)} parents")

    for out_dir in (args.water_out, args.ood_out, args.malon_out):
        if out_dir.exists():
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifests under {args.water_out.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
