#!/usr/bin/env python3
"""Build the out-of-distribution water splits.

The existing size hold-out (train on n=2-6, test on n=8-12) is a weak notion of
out-of-distribution: the test clusters are the same motifs with more molecules,
built by the same generator, and Reviewers 1 and 2 both said so. These splits
move away from the training distribution along axes the training set has no
access to at all.

Tier 2, order. Clusters carved from real ice lattices (genice2): Ih and Ic are
proton-disordered, XI is the proton-ordered form of Ih, II is a denser
polymorph. Training sees only thermally jittered gas-phase motifs, so the
tetrahedral order parameter and the ring statistics differ in kind, not degree.
Fluctuations about the lattice minimum are added as rigid-molecule displacements
with amplitudes taken from equipartition against the measured Einstein and
librational frequencies of ice, which keeps the crystalline oxygen network
intact while making the frames genuinely thermal.

Tier 3, density. Droplets: a compact cluster at roughly bulk density is expanded
affinely, moving whole rigid molecules apart without touching internal geometry,
down to a fifth of bulk density. This is deliberately not a bulk system -- it is
large, tenuous, and mostly surface, which is the regime where a model trained on
compact hydrogen-bonded motifs has the least support. Evaporation frames put one
molecule well outside the droplet.

Usage:
  python scripts/revision/generate_ood_water.py --all
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

# Physical constants in the units used here (Angstrom, K, amu, cm^-1).
KB_EV = 8.617333262e-5
AMU_EV_PER_A2_PER_C2 = 1.0364e-4  # amu * (cm^-1)^2 * A^2 -> eV, see _sigma_from_mode
M_WATER = 18.01528
I_WATER = np.array([1.0220, 1.9187, 2.9407])  # principal moments, amu A^2

# Ice Ih: translational (Einstein) band near 230 cm^-1, librational band near
# 600-900 cm^-1. Used only to set displacement amplitudes, so the band centre is
# enough; the point is that the jitter is thermal in magnitude, not arbitrary.
OMEGA_TRANS_CM = 230.0
OMEGA_LIBR_CM = 700.0
OMEGA_INTRA_CM = 1600.0

BULK_DENSITY_G_CM3 = 0.997
ROH = 0.9572
HOH_DEG = 104.52


HBAR = 1.054571817e-34
AMU_KG = 1.66053907e-27


def _mean_square_amplitude(temperature_k: float, inertia_si: float, omega_cm: float) -> float:
    """<u^2> for a quantum harmonic oscillator, in SI units of `inertia_si`.

    <u^2> = (hbar / 2 I w) coth(hbar w / 2 kT), which reduces to the classical
    kT / (I w^2) at high temperature but keeps the zero-point amplitude at 50 K,
    where a purely classical estimate would make the crystal unphysically rigid.
    `inertia_si` is a mass in kg for a translation and a moment of inertia in
    kg m^2 for a libration, which is what keeps the two cases dimensionally
    honest: the first returns m^2, the second rad^2.
    """
    omega_rad = omega_cm * 2.0 * math.pi * 2.99792458e10
    x = (omega_cm * 1.239841984e-4) / (2.0 * KB_EV * max(temperature_k, 1.0))
    coth = 1.0 / math.tanh(x) if x < 50 else 1.0
    return HBAR / (2.0 * inertia_si * omega_rad) * coth


def translational_sigma_a(temperature_k: float, mass_amu: float, omega_cm: float) -> float:
    """RMS centre-of-mass displacement per Cartesian component, in Angstrom."""
    return math.sqrt(_mean_square_amplitude(temperature_k, mass_amu * AMU_KG, omega_cm)) * 1e10


def librational_sigma_rad(temperature_k: float, inertia_amu_a2: float, omega_cm: float) -> float:
    """RMS librational angle, in radians."""
    inertia_si = inertia_amu_a2 * AMU_KG * 1e-20
    return math.sqrt(_mean_square_amplitude(temperature_k, inertia_si, omega_cm))


def _random_rotation(max_angle_rad: float, rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.normal(0.0, max_angle_rad)
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) * c + s * k + (1 - c) * np.outer(axis, axis)


# --------------------------------------------------------------------------
# Molecule bookkeeping
# --------------------------------------------------------------------------


def split_molecules(symbols: list[str], coords: np.ndarray) -> list[np.ndarray]:
    """Group an O,H,H-ordered frame into per-molecule coordinate blocks."""
    mols = []
    i = 0
    while i < len(symbols):
        if symbols[i] != "O":
            raise ValueError(f"expected O at position {i}, found {symbols[i]}")
        mols.append(coords[i : i + 3].copy())
        i += 3
    return mols


def molecules_to_frame(mols: list[np.ndarray]) -> tuple[list[str], np.ndarray]:
    symbols: list[str] = []
    rows: list[np.ndarray] = []
    for mol in mols:
        symbols.extend(["O", "H", "H"])
        rows.append(mol)
    return symbols, np.vstack(rows)


def write_xyz(path: Path, frames: list[tuple[list[str], np.ndarray]], comments: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for (symbols, coords), comment in zip(frames, comments):
            fh.write(f"{len(symbols)}\n{comment}\n")
            for sym, xyz in zip(symbols, coords):
                fh.write(f"{sym:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}\n")


# --------------------------------------------------------------------------
# Tier 2: crystalline cutouts
# --------------------------------------------------------------------------


def genice_frame(polymorph: str, rep: int, seed: int, genice_bin: str) -> tuple[list[str], np.ndarray]:
    cmd = [genice_bin, polymorph, "-r", str(rep), str(rep), str(rep), "-w", "tip3p", "-f", "xyz", "--seed", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"genice2 failed for {polymorph}: {proc.stderr[-400:]}")
    lines = [ln for ln in proc.stdout.splitlines() if not ln.startswith("#")]
    nat = int(lines[0].strip())
    symbols, coords = [], []
    for row in lines[2 : 2 + nat]:
        parts = row.split()
        symbols.append(parts[0])
        coords.append([float(v) for v in parts[1:4]])
    return symbols, np.asarray(coords, dtype=float)


def carve_cluster(mols: list[np.ndarray], n_waters: int) -> list[np.ndarray]:
    """The `n_waters` molecules closest to the most interior molecule.

    Seeding from the most interior molecule rather than from the geometric
    centre keeps the cutout away from the supercell faces, so the cluster is
    carved out of bulk-coordinated ice rather than out of a slab surface.
    """
    oxygens = np.array([m[0] for m in mols])
    centre = oxygens.mean(axis=0)
    seed_idx = int(np.argmin(np.linalg.norm(oxygens - centre, axis=1)))
    order = np.argsort(np.linalg.norm(oxygens - oxygens[seed_idx], axis=1))
    picked = [mols[i].copy() for i in order[:n_waters]]
    com = np.vstack(picked).mean(axis=0)
    return [m - com for m in picked]


# Matches sigmas_id in generate_water_cluster_geoms.py so the intramolecular
# marginal of these frames is drawn from the same distribution as training.
TRAIN_INTRA_SIGMAS = (0.04, 0.06, 0.08)


def molecules_are_valid(mols: list[np.ndarray], min_oo: float = 2.40, oh_range=(0.75, 1.25)) -> bool:
    """Reject frames where thermal noise has produced unphysical contacts."""
    oxygens = np.array([m[0] for m in mols])
    if len(oxygens) > 1:
        dist = np.linalg.norm(oxygens[:, None] - oxygens[None], axis=-1)
        np.fill_diagonal(dist, np.inf)
        if dist.min() < min_oo:
            return False
    for mol in mols:
        for h in mol[1:]:
            if not (oh_range[0] <= np.linalg.norm(h - mol[0]) <= oh_range[1]):
                return False
    return True


def thermal_displace(
    mols: list[np.ndarray],
    temperature_k: float,
    rng: np.random.Generator,
    intra_sigma: float = 0.03,
) -> list[np.ndarray]:
    """Rigid-molecule translation and libration, plus a small internal jitter.

    Molecules move as rigid bodies on purpose. The claim this split supports is
    that the *network* is out of distribution, so the intramolecular geometry is
    deliberately kept inside the training distribution: `intra_sigma` matches the
    0.04-0.08 Angstrom jitter that generate_water_cluster_geoms.py applies. If
    bond lengths drifted too, an error on these frames could be blamed on
    stretched monomers rather than on the crystalline environment.
    """
    sigma_t = translational_sigma_a(temperature_k, M_WATER, OMEGA_TRANS_CM)
    sigma_rot = librational_sigma_rad(temperature_k, float(I_WATER.mean()), OMEGA_LIBR_CM)

    out = []
    for mol in mols:
        com = mol.mean(axis=0)
        local = mol - com
        rot = _random_rotation(sigma_rot, rng)
        moved = local @ rot.T + com + rng.normal(0.0, sigma_t, size=3)
        moved = moved + rng.normal(0.0, intra_sigma, size=moved.shape)
        out.append(moved)
    return out


def build_ice_split(
    polymorphs: list[str],
    sizes: list[int],
    temperatures: list[float],
    per_combo: int,
    rep: int,
    seed: int,
    genice_bin: str,
) -> tuple[list, list, list]:
    rng = np.random.default_rng(seed)
    frames, comments, meta = [], [], []
    for polymorph in polymorphs:
        # One lattice realisation per polymorph; the proton disorder in Ih/Ic
        # already differs between genice2 seeds, and re-running it per frame
        # would dominate the runtime for no extra structural variety.
        symbols, coords = genice_frame(polymorph, rep, seed, genice_bin)
        all_mols = split_molecules(symbols, coords)
        for n in sizes:
            if n > len(all_mols):
                continue
            base = carve_cluster(all_mols, n)
            for temperature in temperatures:
                for k in range(per_combo):
                    intra = float(rng.choice(TRAIN_INTRA_SIGMAS))
                    mols = None
                    for _ in range(200):
                        trial = (
                            base
                            if temperature <= 0
                            else thermal_displace(base, temperature, rng, intra_sigma=intra)
                        )
                        if molecules_are_valid(trial):
                            mols = trial
                            break
                    if mols is None:
                        raise RuntimeError(
                            f"no valid {polymorph} n={n} frame at {temperature} K after 200 tries"
                        )
                    frames.append(molecules_to_frame(mols))
                    comments.append(f"ice{polymorph} n={n} T={temperature:.0f}K rep={k}")
                    meta.append(
                        {"tier": "ood_order", "polymorph": polymorph, "n": n, "T": temperature, "intra_sigma": intra}
                    )
    return frames, comments, meta


# --------------------------------------------------------------------------
# Tier 3: droplets
# --------------------------------------------------------------------------


def ideal_water_geometry(rng: np.random.Generator) -> np.ndarray:
    half = math.radians(HOH_DEG) / 2.0
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [ROH * math.sin(half), 0.0, ROH * math.cos(half)],
            [-ROH * math.sin(half), 0.0, ROH * math.cos(half)],
        ]
    )
    return local @ _random_rotation(math.pi, rng).T


def droplet_radius(n_waters: int, density_g_cm3: float) -> float:
    """Radius of a sphere holding `n_waters` at `density_g_cm3`, in Angstrom."""
    mass_g = n_waters * M_WATER / 6.02214076e23
    volume_cm3 = mass_g / density_g_cm3
    volume_a3 = volume_cm3 * 1e24
    return (3.0 * volume_a3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def _spherical_wall_calculator(radius: float, spring: float = 5.0):
    """A soft spherical wall, as an ASE calculator.

    A free cluster at 320 K evaporates long before it finishes disordering, so
    the melt needs a container. The wall is flat inside `radius` and harmonic
    outside, which holds the droplet together without biasing its internal
    structure.
    """
    from ase.calculators.calculator import Calculator, all_changes

    class SphericalWall(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            positions = atoms.get_positions()
            centre = positions.mean(axis=0)
            delta = positions - centre
            dist = np.linalg.norm(delta, axis=1)
            excess = np.clip(dist - radius, 0.0, None)
            energy = 0.5 * spring * float((excess**2).sum())
            forces = np.zeros_like(positions)
            active = excess > 0
            if active.any():
                unit = delta[active] / dist[active][:, None]
                forces[active] = -spring * excess[active][:, None] * unit
            self.results = {"energy": energy, "forces": forces}

    return SphericalWall()


def melt_droplet(
    mols: list[np.ndarray],
    steps: int,
    temperature_k: float,
    seed: int,
    n_samples: int = 1,
) -> list[list[np.ndarray]]:
    """Disorder an ice cutout into liquid-like droplets with GFN2-xTB dynamics.

    Starting from ice rather than from a random packing is what gives the
    droplet a real hydrogen-bond network: random placement produces the right
    density but no network at all, and a density label computed on that would
    describe a packing artefact rather than water. Melting then destroys the
    crystalline order, which is what separates this tier from the ice tier --
    here the frames are disordered and the out-of-distribution axis is density,
    not order.

    Returns `n_samples` frames spread over the second half of the trajectory,
    once the initial lattice memory has been lost.
    """
    from ase import Atoms, units
    from ase.calculators.mixing import SumCalculator
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from tblite.ase import TBLite

    symbols, coords = molecules_to_frame(mols)
    atoms = Atoms(symbols=symbols, positions=coords)
    radius = droplet_radius(len(mols), BULK_DENSITY_G_CM3) + 1.5
    atoms.calc = SumCalculator([TBLite(method="GFN2-xTB", verbosity=0), _spherical_wall_calculator(radius)])

    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k, rng=rng)
    dyn = Langevin(atoms, 0.5 * units.fs, temperature_K=temperature_k, friction=0.02, rng=rng)

    samples: list[list[np.ndarray]] = []
    first_sample = steps // 2
    stride = max(1, (steps - first_sample) // max(1, n_samples))
    for step in range(steps):
        dyn.run(1)
        if step >= first_sample and (step - first_sample) % stride == 0 and len(samples) < n_samples:
            samples.append(split_molecules(symbols, atoms.get_positions()))
    if not samples:
        samples.append(split_molecules(symbols, atoms.get_positions()))
    return samples


def expand_droplet(mols: list[np.ndarray], factor: float) -> list[np.ndarray]:
    """Move rigid molecules apart by `factor` about the droplet centre.

    Intramolecular geometry is untouched, so the only thing that changes is the
    intermolecular environment: the same molecules at a lower density.
    """
    centre = np.vstack(mols).mean(axis=0)
    out = []
    for mol in mols:
        com = mol.mean(axis=0)
        out.append(mol - com + (centre + (com - centre) * factor))
    return out


def evaporate(mols: list[np.ndarray], distance: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Pull the outermost molecule `distance` further out along its own radius."""
    centre = np.vstack(mols).mean(axis=0)
    coms = np.array([m.mean(axis=0) for m in mols])
    idx = int(np.argmax(np.linalg.norm(coms - centre, axis=1)))
    direction = coms[idx] - centre
    direction = direction / (np.linalg.norm(direction) or 1.0)
    out = [m.copy() for m in mols]
    out[idx] = out[idx] + direction * distance + rng.normal(0.0, 0.05, size=3)
    return out


def build_droplet_split(
    sizes: list[int],
    densities: list[float],
    per_combo: int,
    melt_steps: int,
    melt_temperature: float,
    seed: int,
    rep_supercell: int,
    genice_bin: str,
) -> tuple[list, list, list]:
    rng = np.random.default_rng(seed)
    frames, comments, meta = [], [], []
    for n in sizes:
        # Each melt is expensive, so one trajectory per size supplies several
        # independent starting droplets from its second half.
        symbols, coords = genice_frame("Ih", rep_supercell, seed + n, genice_bin)
        cutout = carve_cluster(split_molecules(symbols, coords), n)
        print(f"  melting n={n} for {melt_steps} steps at {melt_temperature:.0f} K")
        bases = melt_droplet(cutout, melt_steps, melt_temperature, seed + n, n_samples=per_combo)
        for rep, base in enumerate(bases):
            for density in densities:
                factor = (BULK_DENSITY_G_CM3 / density) ** (1.0 / 3.0)
                mols = base if abs(factor - 1.0) < 1e-9 else expand_droplet(base, factor)
                frames.append(molecules_to_frame(mols))
                comments.append(f"droplet n={n} rho={density:.2f} rep={rep}")
                meta.append({"tier": "ood_density", "n": n, "density": density, "kind": "expanded"})
            evaporated = evaporate(base, float(rng.uniform(4.0, 10.0)), rng)
            frames.append(molecules_to_frame(evaporated))
            comments.append(f"droplet-evaporation n={n} rep={rep}")
            meta.append({"tier": "ood_density", "n": n, "density": BULK_DENSITY_G_CM3, "kind": "evaporation"})
    return frames, comments, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "datasets/revision/water_ood")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ice", action="store_true")
    parser.add_argument("--droplet", action="store_true")
    parser.add_argument("--polymorphs", default="Ih,XI,Ic,II")
    parser.add_argument("--ice-sizes", default="12,16,20,24")
    parser.add_argument("--ice-temperatures", default="50,150")
    parser.add_argument("--ice-per-combo", type=int, default=8)
    parser.add_argument("--ice-rep", type=int, default=3)
    parser.add_argument("--droplet-sizes", default="16,20,24")
    parser.add_argument("--droplet-densities", default="0.8,0.6,0.4,0.2")
    parser.add_argument("--droplet-per-combo", type=int, default=10)
    parser.add_argument("--droplet-melt-steps", type=int, default=4000)
    parser.add_argument("--droplet-melt-temperature", type=float, default=330.0)
    parser.add_argument("--genice-bin", default=str(_REPO_ROOT / ".venv/bin/genice2"))
    args = parser.parse_args()

    do_ice = args.ice or args.all
    do_droplet = args.droplet or args.all
    if not (do_ice or do_droplet):
        raise SystemExit("nothing to do: pass --all, --ice, or --droplet")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"seed": args.seed, "splits": {}}

    if do_ice:
        polymorphs = [p.strip() for p in args.polymorphs.split(",") if p.strip()]
        sizes = [int(v) for v in args.ice_sizes.split(",")]
        temps = [float(v) for v in args.ice_temperatures.split(",")]
        print(f"building ice cutouts: {polymorphs} sizes={sizes} T={temps}")
        frames, comments, meta = build_ice_split(
            polymorphs, sizes, temps, args.ice_per_combo, args.ice_rep, args.seed, args.genice_bin
        )
        path = args.output_dir / "ood_order.xyz"
        write_xyz(path, frames, comments)
        manifest["splits"]["ood_order"] = {"xyz": str(path), "n_frames": len(frames), "meta": meta}
        print(f"  wrote {len(frames)} frames -> {path}")

    if do_droplet:
        sizes = [int(v) for v in args.droplet_sizes.split(",")]
        densities = [float(v) for v in args.droplet_densities.split(",")]
        print(f"building droplets: sizes={sizes} densities={densities}")
        frames, comments, meta = build_droplet_split(
            sizes,
            densities,
            args.droplet_per_combo,
            args.droplet_melt_steps,
            args.droplet_melt_temperature,
            args.seed,
            args.ice_rep,
            args.genice_bin,
        )
        path = args.output_dir / "ood_density.xyz"
        write_xyz(path, frames, comments)
        manifest["splits"]["ood_density"] = {"xyz": str(path), "n_frames": len(frames), "meta": meta}
        print(f"  wrote {len(frames)} frames -> {path}")

    manifest_path = args.output_dir / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"splits": {}}
    existing.setdefault("splits", {}).update(manifest["splits"])
    existing["seed"] = args.seed
    manifest_path.write_text(json.dumps(existing, indent=2))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
