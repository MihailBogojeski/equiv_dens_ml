#!/usr/bin/env python3
"""
Fit free-atom densities with many s-type Gaussians.

For every element in a free-atom file (mo_basis, mo_coeff, mo_occ, ...) the coefficients c of
rho~(r) = sum_k c_k phi_k(r), phi_k = (2 a_k / pi)^(3/4) exp(-a_k r^2) (pyscf-normalized s shell),
minimize the Coulomb self-energy of the error density J[rho - rho~] (+ mu times its L2 norm) subject
to the electron count int rho~ = N and, by default, the exact nuclear attraction V[rho~] = V[rho]
(V = -Z int rho / r). The exponents are even-tempered between 0.3x the most diffuse and 2x the
tightest MO primitive.

The Coulomb matrix of that many Gaussians has condition number ~1e12-1e15, so the double precision
errors of the integrals were amplified into the fit. Here the reference density is expanded
analytically in the MO primitives and every integral and the KKT solve are done in 60-digit
mpmath arithmetic, which removes the noise floor: the error keeps falling with the number of
functions and the coefficients stay O(1-10).

The output has the same format as the input, with df_basis / df_coeffs replaced by the s basis /
coefficients, so it works with --atom_dens_type=df_coeffs.
--scan N1 N2 ... only prints the errors for different numbers of s functions.
"""
import argparse

import numpy as np
import torch
from mpmath import mp
from pyscf import gto, dft, df

mp.dps = 60

HA2KCAL = 627.5096080305927

# recommended number of s functions per element (J < 1e-2 kcal/mol and relL1 < 1e-4)
N_S_DEFAULT = {1: 18, 6: 36, 7: 36, 8: 36, 9: 40, 16: 44, 17: 44}


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


def radial_terms(ref, mo_basis):
    """rho(r) = sum_t w_t r^(2 l_t) exp(-g_t r^2) for the (spherical) reference density, as an exact
    mpmath expansion over products of MO primitives. Returns {(l, g): w}."""
    mol, dm = ref.mol, ref.dm
    loc = mol.ao_loc_nr()
    shells = []
    for sh in mo_basis:
        l = sh[0]
        a = [mp.mpf(float(pr[0])) for pr in sh[1:]]
        cs = np.array([pr[1:] for pr in sh[1:]], dtype=float)
        cols = []
        for c in range(cs.shape[1]):
            # unit-normalized radial function: int R^2 r^2 dr = 1
            cp = [mp.mpf(float(cs[p, c])) * mp.sqrt(2 * (2 * a[p]) ** (l + mp.mpf(3) / 2) / mp.gamma(l + mp.mpf(3) / 2))
                  for p in range(len(a))]
            S = sum(cp[p] * cp[q] * mp.gamma(l + mp.mpf(3) / 2) / (2 * (a[p] + a[q]) ** (l + mp.mpf(3) / 2))
                    for p in range(len(a)) for q in range(len(a)))
            cols.append([x / mp.sqrt(S) for x in cp])
        shells.append((l, a, cols))
    terms = {}
    for ib, (l, a, cols) in enumerate(shells):
        for jb, (l2, a2, cols2) in enumerate(shells):
            if l2 != l:
                continue
            for c, cc in enumerate(cols):
                for c2, cc2 in enumerate(cols2):
                    D = mp.mpf(float(np.mean([dm[loc[ib] + c * (2 * l + 1) + m, loc[jb] + c2 * (2 * l + 1) + m]
                                              for m in range(2 * l + 1)])))
                    for p in range(len(a)):
                        for q in range(len(a2)):
                            key = (l, a[p] + a2[q])
                            terms[key] = terms.get(key, 0) + D * cc[p] * cc2[q] * (2 * l + 1) / (4 * mp.pi)
    return terms


def erf_moment(l, g, a):
    """int_0^inf r^(2l+1) exp(-g r^2) erf(sqrt(a) r) dr for l = 0, 1, 2"""
    h = g + a
    s = mp.sqrt(a) / 2
    if l == 0:
        return s / (g * mp.sqrt(h))
    if l == 1:
        return s * (1 / (g ** 2 * mp.sqrt(h)) + 1 / (2 * g * h ** 1.5))
    return s * (2 / (g ** 3 * mp.sqrt(h)) + 1 / (g ** 2 * h ** 1.5) + 3 / (4 * g * h ** 2.5))


def fit_exact(Z, terms, a, mu, exact_V):
    """minimize J[rho - rho~] + mu int (rho - rho~)^2 with int rho~ = Z (and V[rho~] = V[rho] if
    exact_V) in 60-digit arithmetic. Returns float coefficients."""
    K = len(a)
    a = [mp.mpf(float(x)) for x in a]
    pi = mp.pi
    N = [(2 * x / pi) ** mp.mpf(0.75) for x in a]
    A = mp.matrix(K, K)
    S = mp.matrix(K, K)
    for k in range(K):
        for m in range(K):
            A[k, m] = N[k] * N[m] * 2 * pi ** 2.5 / (a[k] * a[m] * mp.sqrt(a[k] + a[m]))
            S[k, m] = N[k] * N[m] * (pi / (a[k] + a[m])) ** 1.5
    b = mp.matrix(K, 1)
    sv = mp.matrix(K, 1)
    for k in range(K):
        b[k] = sum(w * 4 * pi * N[k] * (pi / a[k]) ** 1.5 * erf_moment(l, g, a[k]) for (l, g), w in terms.items())
        sv[k] = sum(w * N[k] * 4 * pi * mp.gamma(l + 1.5) / (2 * (g + a[k]) ** (l + 1.5)) for (l, g), w in terms.items())
    v_ref = -Z * sum(w * 4 * pi * mp.gamma(l + 1) / (2 * g ** (l + 1)) for (l, g), w in terms.items())
    rows = [[N[k] * (pi / a[k]) ** 1.5 for k in range(K)]]
    d = [mp.mpf(Z)]
    if exact_V:
        rows.append([-Z * N[k] * 2 * pi / a[k] for k in range(K)])
        d.append(v_ref)
    m = len(rows)
    KKT = mp.matrix(K + m, K + m)
    rhs = mp.matrix(K + m, 1)
    for k in range(K):
        for j in range(K):
            KKT[k, j] = A[k, j] + mu * S[k, j]
        rhs[k] = b[k] + mu * sv[k]
        for i in range(m):
            KKT[k, K + i] = rows[i][k]
            KKT[K + i, k] = rows[i][k]
    for i in range(m):
        rhs[K + i] = d[i]
    sol = mp.lu_solve(KKT, rhs)
    return np.array([float(sol[k]) for k in range(K)])


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
    m = evaluate(ref, dft.numint.eval_ao(aux, ref.coords) @ c, j_err)
    # analytic N and V of the s shells (higher l integrate to zero): int phi = (pi/a)^(3/2) n, int phi/r = 2 pi/a n
    s = [(sh[1][0], ci) for sh, ci in zip(aux_basis, c) if sh[0] == 0]
    a, cs = np.array([x[0] for x in s]), np.array([x[1] for x in s])
    norm = cs * (2 * a / np.pi) ** 0.75
    m["N_an"] = np.sum(norm * (np.pi / a) ** 1.5) - ref.Z
    m["V_an"] = (-ref.Z * np.sum(norm * 2 * np.pi / a) - ref.v_ref) * HA2KCAL
    return m


def show(name, m):
    print(f"    {name:16s} N={m['N']:10.6f}  J {m['J']:9.2e}  V {m['V']:9.2e} kcal/mol  "
          f"relL1 {m['L1']:.2e}  nucleus {m['near']:.2e}  tail {m['tail']:.2e}"
          + (f"  analytic: dN {m['N_an']:9.2e}  dV {m['V_an']:9.2e} kcal/mol" if "V_an" in m else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="/workspace/datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf.npy")
    p.add_argument("--minimized", default="/workspace/datasets/free_atom_densities_augccpvdz_augccpvqzjkfit_pyscf_minimized.npy",
                   help="jkfit l=0-only fit: baseline and source of the jkfit l=0 exponents")
    p.add_argument("--output", default=None)
    p.add_argument("--n_s", type=int, default=None, help="uniform number of s functions (default: N_S_DEFAULT per element)")
    p.add_argument("--n_s_per_element", nargs="*", default=[], help="Z:n pairs overriding the defaults, e.g. 1:12 6:40")
    p.add_argument("--elements", type=int, nargs="*", default=None, help="only these Z (default: all in the input)")
    p.add_argument("--scan", type=int, nargs="*", default=None)
    p.add_argument("--exps", choices=["et", "jk"], default="et", help="even-tempered, or jkfit l=0 exponents plus gap filling")
    p.add_argument("--mu", type=float, default=0.0, help="weight of the L2 density error added to J")
    p.add_argument("--no_exact_V", action="store_true", help="do not constrain V[rho~] = V[rho]")
    p.add_argument("--hi", type=float, default=2.0, help="tightest exponent = hi * tightest MO primitive")
    p.add_argument("--lo", type=float, default=0.3, help="most diffuse exponent = lo * most diffuse MO primitive")
    args = p.parse_args()

    data = np.load(args.input, allow_pickle=True).item()
    mini = np.load(args.minimized, allow_pickle=True).item()
    n_per = {int(z): int(n) for z, n in (x.split(":") for x in args.n_s_per_element)}
    if args.n_s is None and not n_per:
        n_per = N_S_DEFAULT
    out = {}
    for Z, d in data.items():
        if args.elements is not None and Z not in args.elements:
            continue
        ref = Reference(Z, d)
        terms = radial_terms(ref, d["mo_basis"][Z])
        jk_basis = mini[Z]["df_basis"][Z]
        a_lo, a_hi = exponent_range(d["mo_basis"][Z], args.lo, args.hi)
        print(f"Z={Z}")
        if args.scan is None:
            show("full jkfit df", evaluate_df(ref, d["df_basis"][Z], d["df_coeffs"].numpy()))
        show(f"jkfit l=0 ({len([e for e in jk_basis if e[0] == 0])})", evaluate_df(ref, jk_basis, mini[Z]["df_coeffs"].numpy()))
        if args.scan is None:
            show("spline", evaluate(ref, d["spline_interp"](np.log(ref.r)).clip(0)))
        for n in args.scan or [n_per.get(Z, args.n_s or 60)]:
            a = et_exponents(a_lo, a_hi, n) if args.exps == "et" else jk_exponents(jk_basis, a_lo, a_hi, n)
            c = fit_exact(Z, terms, a, args.mu, not args.no_exact_V)
            show(f"{args.exps} {n}", evaluate_df(ref, s_basis(a), c))
            out[Z] = dict(d)
            out[Z]["df_basis"] = {Z: s_basis(a)}
            out[Z]["df_coeffs"] = torch.tensor(c, dtype=torch.float64).reshape(1, -1)
    if args.scan is None:
        path = args.output or "/workspace/datasets/free_atom_densities_augccpvdz_sgauss_pe.npy"
        np.save(path, out, allow_pickle=True)
        print("saved", path)


if __name__ == "__main__":
    main()
