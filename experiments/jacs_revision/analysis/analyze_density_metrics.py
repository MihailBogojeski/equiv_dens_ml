#!/usr/bin/env python3
"""How big is the SAD correction next to the density itself? (R2.8, R3.4)

Reviewer 3 asked what delta-learning buys over predicting the density directly,
and Reviewer 2 asked whether a softplus output is consistent with a correction
that must change sign. Both are questions about the same object: the difference
between the reference density and the superposition-of-atomic-densities prior,

    delta_rho(r) = rho_ref(r) - rho_SAD(r)

so this integrates both on a real grid rather than inferring anything from
fitting coefficients. An earlier version reported the mean absolute value of the
DF coefficient vector, which is not a density and is not comparable between
molecules: the coefficients live in an over-complete auxiliary basis where large
cancelling components can leave the density untouched.

The two densities are expanded in different bases and that is not incidental.
The reference is a density fit in the molecular auxiliary basis
(aug-cc-pVQZ-JKFIT, 142 functions on O and 76 on H). The prior stores each free
atom in its own compact, spherically symmetric s-only basis (11 on O, 5 on H),
so it is evaluated atom by atom and summed. Treating the prior's coefficients as
if they indexed the molecular auxiliary basis would silently misplace it.

Reported per frame and aggregated:

  * ``int |delta_rho| / int |rho_ref|`` -- the fraction of the density the
    network has to produce. This is the number that says whether the correction
    is small.
  * ``int |delta_rho| / N_elec`` -- the same quantity on the scale the paper
    reports density errors, so a model error can be put beside it.
  * a signed, volume-weighted histogram of ``delta_rho``, which is what shows
    the correction takes both signs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from equiv_dens.utils.scipy_compat import enable_legacy_bspline_unpickling


def _load_obj(path: Path):
    # Priors from the published models hold BSplines pickled under an older
    # SciPy; without this the load raises before anything can be measured.
    enable_legacy_bspline_unpickling()
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        return data.item() if data.ndim == 0 else list(data)
    return data


def sad_density_on_grid(mol, sad, coords):
    """Sum the free-atom prior densities at their nuclear positions."""
    from pyscf import gto

    rho = np.zeros(len(coords))
    per_atom_electrons = []
    for i in range(mol.natm):
        z = int(mol.atom_charge(i))
        entry = sad.get(z)
        if entry is None or "df_coeffs" not in entry:
            raise KeyError(f"SAD prior has no df_coeffs for Z={z}")
        basis = entry["df_basis"][z] if isinstance(entry["df_basis"], dict) else entry["df_basis"]
        atom_mol = gto.Mole()
        atom_mol.atom = [[int(z), tuple(mol.atom_coord(i, unit="Angstrom"))]]
        atom_mol.basis = {gto.mole._symbol(int(z)): basis}
        atom_mol.unit = "Angstrom"
        atom_mol.spin = None
        atom_mol.charge = 0
        atom_mol.build(False, False)
        ao = atom_mol.eval_gto("GTOval_sph", coords)
        c = np.asarray(entry["df_coeffs"]).ravel()
        if ao.shape[1] != c.size:
            raise ValueError(
                f"Z={z}: prior has {c.size} coefficients but its basis spans {ao.shape[1]}"
            )
        contribution = ao @ c
        rho += contribution
        per_atom_electrons.append(float(z))
    return rho, per_atom_electrons


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dens-ref", type=Path, required=True, help="DF dataset npy")
    ap.add_argument("--atom-dens", type=Path, required=True, help="SAD prior npy")
    ap.add_argument("--out", type=Path, default=Path("results/revision/density_metrics.json"))
    ap.add_argument("--max-frames", type=int, default=25)
    ap.add_argument("--grid-level", type=int, default=1)
    ap.add_argument("--hist-bins", type=int, default=61)
    ap.add_argument("--hist-range", type=float, default=0.5,
                    help="signed delta_rho histogram spans +/- this, in e/a0^3")
    cli = ap.parse_args()

    from pyscf import df as pyscf_df
    from pyscf import gto
    from pyscf.dft import gen_grid

    frames = _load_obj(cli.dens_ref)
    if not isinstance(frames, list):
        raise SystemExit(f"expected a list of (mol.pack(), calc) in {cli.dens_ref}")
    sad = _load_obj(cli.atom_dens)
    if not isinstance(sad, dict):
        raise SystemExit("SAD prior must be a dict keyed by atomic number")

    edges = np.linspace(-cli.hist_range, cli.hist_range, cli.hist_bins + 1)
    pooled = np.zeros(cli.hist_bins)
    below = above = 0.0
    neg_weight = all_weight = 0.0
    rows = []
    sad_seconds = 0.0

    for k, item in enumerate(frames[: cli.max_frames]):
        pack, calc = item[0], item[1]
        if "df_coeff" not in calc:
            continue
        mol = gto.Mole.unpack(pack)
        mol.build(False, False)

        grids = gen_grid.Grids(mol)
        grids.level = cli.grid_level
        grids.build()
        w = grids.weights

        auxmol = pyscf_df.addons.make_auxmol(mol, calc["auxbasis"])
        ao_aux = auxmol.eval_gto("GTOval_sph", grids.coords)
        rho_ref = ao_aux @ np.asarray(calc["df_coeff"]).ravel()

        t0 = time.perf_counter()
        rho_sad, _ = sad_density_on_grid(mol, sad, grids.coords)
        sad_seconds += time.perf_counter() - t0

        delta = rho_ref - rho_sad
        n_elec = float(mol.nelectron)
        int_ref = float(np.dot(w, rho_ref))
        int_sad = float(np.dot(w, rho_sad))
        int_abs_ref = float(np.dot(w, np.abs(rho_ref)))
        int_abs_delta = float(np.dot(w, np.abs(delta)))

        pooled += np.histogram(delta, bins=edges, weights=w)[0]
        # np.histogram drops everything outside the range. The correction is
        # sharpest at the nuclei, so the discarded tail is precisely the large
        # positive part; without tracking it, the fraction below zero would be
        # computed against a denominator missing its positive extreme and would
        # come out near unity for a correction that is not.
        below += float(np.dot(w, (delta < edges[0]).astype(float)))
        above += float(np.dot(w, (delta > edges[-1]).astype(float)))
        # Accumulated from the values, not from the bins. The central bin
        # straddles zero and holds most of the diffuse region, so summing bins
        # with a negative left edge counted that entire bin as negative and put
        # the figure at 0.998 where the true share is 0.84.
        neg_weight += float(np.dot(w, (delta < 0).astype(float)))
        all_weight += float(w.sum())
        rows.append({
            "frame": k,
            "n_atoms": int(mol.natm),
            "n_elec": n_elec,
            "integrated_rho_ref": int_ref,
            "integrated_rho_sad": int_sad,
            # A fit that integrates to the wrong electron count would make every
            # ratio below meaningless, so it is reported rather than assumed.
            "electron_count_error_ref": int_ref - n_elec,
            "abs_delta_over_abs_rho": int_abs_delta / int_abs_ref if int_abs_ref else None,
            "abs_delta_per_electron": int_abs_delta / n_elec if n_elec else None,
            "fraction_negative_delta_by_volume": float(np.dot(w, (delta < 0).astype(float)) / w.sum()),
            "max_abs_delta": float(np.abs(delta).max()),
        })
        print(f"  [{k + 1}] n_atoms={mol.natm} "
              f"|dRho|/|rho|={rows[-1]['abs_delta_over_abs_rho']:.4f} "
              f"e-count err={rows[-1]['electron_count_error_ref']:+.2e}", flush=True)

    if not rows:
        raise SystemExit("no frames carried df_coeff; nothing to analyse")

    def agg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals))}

    total = pooled.sum() + below + above
    summary = {
        "dens_ref": str(cli.dens_ref),
        "atom_dens": str(cli.atom_dens),
        "n_frames": len(rows),
        "grid_level": cli.grid_level,
        "density_unit": "e/a0^3",
        "abs_delta_over_abs_rho": agg("abs_delta_over_abs_rho"),
        "abs_delta_per_electron": agg("abs_delta_per_electron"),
        "electron_count_error_ref": agg("electron_count_error_ref"),
        "fraction_negative_delta_by_volume": agg("fraction_negative_delta_by_volume"),
        "sad_seconds_per_frame": sad_seconds / len(rows),
        "signed_delta_histogram": {
            "unit": "e/a0^3",
            "edges": edges.tolist(),
            "volume_weighted_counts": pooled.tolist(),
            "volume_weighted_underflow": below,
            "volume_weighted_overflow": above,
            "fraction_below_zero": (neg_weight / all_weight) if all_weight else None,
        },
        "interpretation": (
            "abs_delta_over_abs_rho is the share of the density the network must "
            "supply; the rest is the free-atom prior, which costs a table lookup. "
            "fraction_below_zero shows the correction is signed, so the softplus "
            "constrains rho_SAD + delta_rho rather than delta_rho itself."
        ),
        "per_frame": rows,
    }
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(summary, indent=2) + "\n")
    brief = {k: v for k, v in summary.items()
             if k not in ("per_frame", "signed_delta_histogram")}
    brief["fraction_of_delta_below_zero"] = summary["signed_delta_histogram"]["fraction_below_zero"]
    print()
    print(json.dumps(brief, indent=2))
    print(f"\nwrote {cli.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
