#!/usr/bin/env python3
"""
Fit free-atom densities with many s-type Gaussians.

For every element in a free-atom file (mo_basis, mo_coeff, mo_occ, ...) the coefficients c of
rho~(r) = sum_k c_k phi_k(r), phi_k = (2 a_k / pi)^(3/4) exp(-a_k r^2) (pyscf-normalized s shell),
minimize

    J[rho - rho~] + w_V (V[rho] - V[rho~])^2   subject to   int rho~ = N,

J is the Coulomb self-energy of the error density, V = -Z int rho / r. This is a linear (KKT)
system, solved with an eigenvalue cutoff. The exponents are either even-tempered ("et") or the
l=0 exponents of the jkfit basis with the largest log-gaps filled up ("jk"); "best" fits both
and keeps the one with the lower relative L1 density error. The output has the same format as the input, with
df_basis / df_coeffs replaced by the s basis / coefficients, so it works with
--atom_dens_type=df_coeffs.

--scan N1 N2 ... only prints the errors for different numbers of s functions.
"""
import argparse

import numpy as np
import torch
from pyscf import gto, dft, df

HA2KCAL = 627.5096080305927


def s_basis(a):
    return [[0, [float(x), 1.0]] for x in a]


def atom_mol(Z, basis):
    return gto.M(atom=[[Z, (0, 0, 0)]], basis={Z: basis}, spin=Z % 2, verbose=0)


def exponent_range(mo_basis, lo, hi):
    prims = np.array([pr[0] for sh in mo_basis for pr in sh[1:]])
    return lo * prims.min(), hi * prims.max()


def et_exponents(a_lo, a_hi, n):
    return np.geomspace(a_lo, a_hi, n)


def jk_exponents(jk_basis, a_lo, a_hi, n):
    """jkfit l=0 exponents, then bisect the largest log-gap until there are n."""
    a = [e[1][0] for e in jk_basis if e[0] == 0]
    a = sorted(set(a + [a_lo, a_hi]))
    while len(a) < n:
        gaps = np.diff(np.log(a))
        i = np.argmax(gaps)
        a.insert(i + 1, float(np.sqrt(a[i] * a[i + 1])))
    return np.array(a)


class Reference:
    def __init__(self, Z, d):
        self.Z = Z
        self.mol = atom_mol(Z, d["mo_basis"][Z])
        self.dm = (d["mo_coeff"] * d["mo_occ"]) @ d["mo_coeff"].T
        self.rho_rho = np.einsum("ij,ijkl,kl->", self.dm, self.mol.intor("int2e"), self.dm)
        self.v_ref = np.einsum("ij,ij->", self.dm, self.mol.intor("int1e_nuc"))
        grids = dft.gen_grid.Grids(self.mol)
        grids.level = 7
        grids.build()
        self.coords, self.weights = grids.coords, grids.weights
        self.r = np.linalg.norm(self.coords, axis=1)
        ao = dft.numint.eval_ao(self.mol, self.coords)
        self.rho = np.einsum("pi,ij,pj->p", ao, self.dm, ao)
        self.v_grid = -Z * np.sum(self.rho * self.weights / self.r)

    def three_center(self, aux):
        return np.einsum("ijP,ij->P", df.incore.aux_e2(self.mol, aux, intor="int3c2e"), self.dm)


def fit_coeffs(ref, a, w_V, cutoffs):
    Z = ref.Z
    aux = atom_mol(Z, s_basis(a))
    A = aux.intor("int2c2e")
    b = ref.three_center(aux)
    norm = (2 * a / np.pi) ** 0.75
    n = norm * (np.pi / a) ** 1.5  # int phi
    v = -Z * norm * 2 * np.pi / a  # -Z int phi / r
    M = A + w_V * np.outer(v, v)
    rhs = b + w_V * ref.v_ref * v
    ev, U = np.linalg.eigh(M)
    fits = []
    for cutoff in cutoffs:
        keep = ev > cutoff * ev.max()
        Minv = (U[:, keep] / ev[keep]) @ U[:, keep].T
        lam = (n @ Minv @ rhs - Z) / (n @ Minv @ n)
        fits.append((Minv @ (rhs - lam * n), cutoff))
    return fits


def evaluate(ref, rho_fit, j_err=np.nan):
    """errors of a fitted density sampled on the reference grid"""
    diff = rho_fit - ref.rho
    near = ref.r < 0.1
    tail = (ref.r > 3) & (ref.r < 10)
    return dict(
        N=np.sum(rho_fit * ref.weights),
        J=j_err * HA2KCAL,
        V=(-ref.Z * np.sum(rho_fit * ref.weights / ref.r) - ref.v_grid) * HA2KCAL,
        L1=np.sum(np.abs(diff) * ref.weights) / np.sum(ref.rho * ref.weights),
        near=np.max(np.abs(diff[near])) / ref.rho[near].max(),
        tail=np.max(np.abs(diff[tail]) / ref.rho[tail]),
    )


def evaluate_df(ref, aux_basis, c):
    aux = atom_mol(ref.Z, aux_basis)
    c = np.asarray(c, dtype=float).reshape(-1)
    j_err = ref.rho_rho - 2 * c @ ref.three_center(aux) + c @ aux.intor("int2c2e") @ c
    return evaluate(ref, dft.numint.eval_ao(aux, ref.coords) @ c, j_err)


def show(name, m):
    print(f"    {name:16s} N={m['N']:10.6f}  J {m['J']:9.4f}  V {m['V']:9.4f} kcal/mol  "
          f"relL1 {m['L1']:.2e}  nucleus {m['near']:.2e}  tail {m['tail']:.2e}")


def fit_best(ref, d, jk_basis, n, args):
    a_lo, a_hi = exponent_range(d["mo_basis"][ref.Z], args.lo, args.hi)
    results = {}
    for kind, a in [("et", et_exponents(a_lo, a_hi, n)), ("jk", jk_exponents(jk_basis, a_lo, a_hi, n))]:
        # the normal matrix is ill-conditioned beyond ~1e-12; pick the cutoff with the best density
        evals = [(evaluate_df(ref, s_basis(a), c), c, cutoff) for c, cutoff in fit_coeffs(ref, a, args.w_V, args.cutoffs)]
        m, c, cutoff = min(evals, key=lambda e: e[0]["L1"])
        results[kind] = (m, a, c, cutoff)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/workspace/datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy")
    p.add_argument("--minimized", default="/workspace/datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy",
                   help="jkfit l=0-only fit: baseline and source of the jkfit l=0 exponents")
    p.add_argument("--output", default=None)
    p.add_argument("--n_s", type=int, default=50)
    p.add_argument("--scan", type=int, nargs="*", default=None)
    p.add_argument("--exps", choices=["et", "jk", "best"], default="best")
    p.add_argument("--w_V", type=float, default=1.0)
    p.add_argument("--cutoffs", type=float, nargs="+", default=[1e-14, 1e-13, 1e-12, 1e-11, 1e-10],
                   help="relative eigenvalue cutoffs of the normal matrix to try (best relative L1 is kept)")
    p.add_argument("--hi", type=float, default=2.0, help="tightest exponent = hi * tightest MO primitive")
    p.add_argument("--lo", type=float, default=0.3, help="most diffuse exponent = lo * most diffuse MO primitive")
    args = p.parse_args()

    data = np.load(args.input, allow_pickle=True).item()
    mini = np.load(args.minimized, allow_pickle=True).item()
    out = {}
    for Z, d in data.items():
        ref = Reference(Z, d)
        jk_basis = mini[Z]["df_basis"][Z]
        print(f"Z={Z}")
        if args.scan is None:
            show("full jkfit df", evaluate_df(ref, d["df_basis"][Z], d["df_coeffs"].numpy()))
        show(f"jkfit l=0 ({len([e for e in jk_basis if e[0] == 0])})", evaluate_df(ref, jk_basis, mini[Z]["df_coeffs"].numpy()))
        if args.scan is None:
            spl = d["spline_interp"](np.log(ref.r)).clip(0)
            show("spline", evaluate(ref, spl))
        for n in args.scan or [args.n_s]:
            res = fit_best(ref, d, jk_basis, n, args)
            for kind, (m, a, c, cutoff) in res.items():
                show(f"{kind} {n} (cut {cutoff:.0e})", m)
            if args.scan is None:
                kind = min(res, key=lambda k: res[k][0]["L1"]) if args.exps == "best" else args.exps
                print(f"    -> using {kind}")
                m, a, c, _ = res[kind]
                out[Z] = dict(d)
                out[Z]["df_basis"] = {Z: s_basis(a)}
                out[Z]["df_coeffs"] = torch.tensor(c, dtype=torch.float64).reshape(1, -1)
    if args.scan is None:
        path = args.output or f"/workspace/datasets/free_atom_densities_augccpvdz_sgauss{args.n_s}.npy"
        np.save(path, out, allow_pickle=True)
        print("saved", path)


if __name__ == "__main__":
    main()
