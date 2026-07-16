"""
GPU dipole moment integrals (int1e_r) using CuPy.

Paper-critical path for ML-MD dipole evaluation on long trajectories. This
module accelerates the dominant cost in analytic dipole moments when the
basis is uncontracted (nctr=1); contracted shells fall back to PySCF.

Port of libcint's int1e_r algorithm:
- Overlap CINTg1e_ovlp + position recurrence CINTx1i_1e
- Returns ⟨i|r - R_origin|j⟩ in Bohr (same as PySCF intor_cross('int1e_r'))

Uses NumPy when CuPy is unavailable (xp=np). Install optional dep
``equiv-dens[cupy-dipole]`` (or cupy-cuda12x) for GPU acceleration.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto

def _check_cupy():
    try:
        import cupy as _cp
        return True, _cp
    except Exception:
        return False, None


_CUPY_AVAILABLE, _cp = _check_cupy()

# PySCF/libcint basis layout (from pyscf.gto.mole)
ANG_OF = 1
NPRIM_OF = 2
NCTR_OF = 3
PTR_EXP = 5
PTR_COEFF = 6
BAS_SLOTS = 8
PTR_COORD = 1
ATM_SLOTS = 6
ATOM_OF = 0


def _ncart(l: int) -> int:
    """Number of Cartesian functions for angular momentum l."""
    return (l + 1) * (l + 2) // 2


def _cart_comp(l: int) -> list[tuple[int, int, int]]:
    """Return [(nx, ny, nz), ...] for all Cartesian functions with nx+ny+nz=l."""
    out = []
    for nx in range(l, -1, -1):
        for ny in range(l - nx, -1, -1):
            nz = l - nx - ny
            out.append((nx, ny, nz))
    return out


def _overlap_primitive(
    xp,
    ai: float,
    aj: float,
    ri: tuple[float, float, float],
    rj: tuple[float, float, float],
    ci: float,
    cj: float,
    li: int,
    lj: int,
) -> np.ndarray | "cp.ndarray":
    """
    Compute primitive Cartesian overlap integrals for one shell pair.
    Follows libcint CINTg1e_ovlp.
    Returns array of shape (ncart(li), ncart(lj)).
    """
    p = ai + aj
    rij = tuple((ai * ri[k] + aj * rj[k]) / p for k in range(3))
    rirj = tuple(ri[k] - rj[k] for k in range(3))
    r2 = sum(rirj[k] ** 2 for k in range(3))
    Kab = np.exp(-ai * aj / p * r2)
    fac = Kab * ci * cj * (np.pi / p) ** 1.5
    aij2 = 0.5 / p

    nfi = _ncart(li)
    nfj = _ncart(lj)
    i_orbs = _cart_comp(li)
    j_orbs = _cart_comp(lj)

    # Choose base for vertical recurrence (libcint: li_ceil vs lj_ceil)
    if li >= lj:
        lbase, lother = li, lj
        rx = ri
        rirj_vert = rirj
    else:
        lbase, lother = lj, li
        rx = rj
        rirj_vert = tuple(-rirj[k] for k in range(3))

    rijrx = tuple(rij[k] - rx[k] for k in range(3))
    nmax = lbase + lother

    # Build 1D E arrays for x, y, z (like libcint gx, gy, gz)
    dli = nmax + 1
    dlj = lbase + 1
    g_size = dli * dlj

    # libcint: g_stride_i=1, g_stride_j=dli for overlap (nrys_roots=1)
    def _vert_recurrence(axis: int) -> np.ndarray | "cp.ndarray":
        g = xp.zeros(g_size)
        g[0] = 1.0
        if nmax == 0:
            return g
        rrx = rijrx[axis]
        di = 1
        dj = dli
        g[di] = rrx * g[0]
        for i in range(1, nmax):
            idx = i + 1
            idx_m1 = i - 1
            idx_0 = i
            g[idx] = i * aij2 * g[idx_m1] + rrx * g[idx_0]
        rv = rirj_vert[axis]
        for j in range(1, lbase + 1):
            for i in range(nmax - j + 1):
                n = j * dj + i * di
                g[n] = g[n + di - dj] + rv * g[n - dj]
        return g

    gx = _vert_recurrence(0)
    gy = _vert_recurrence(1)
    gz = _vert_recurrence(2)

    # Map Cartesian (i,j) to flat index in g
    # libcint: idx[n,0] = ofjx + di*i_nx, etc. Product gx*gy*gz gives overlap
    # For CINTg1e_index_xyz: idx[cart_ij, 0..2] indexes into gx, gy, gz
    # Simpler: for each (ia, ib) Cartesian pair, we need the compound index
    # In libcint, for bra cart (nx,ny,nz) and ket (nx',ny',nz'):
    # they use a single flat index n = j*nfi + i, and idx[n,0] = offset in gx
    # The g arrays are 1D; the x,y,z are separate. gout = gx[idx_x] * gy[idx_y] * gz[idx_z]
    # The index: for bra (nx,ny,nz) ket (nxj,nyj,nzj), the libcint index into g:
    # ofjx = dj * j_nx[j], ofjy = g_size + dj*j_ny[j], ofjz = 2*g_size + dj*j_nz[j]
    # So gx index = dj*nxj + di*nxi (where di=1, dj=dli in the "ibase" case)
    # Actually di = g_stride_i = nrys = 1, dj = g_stride_j = dli
    # So gx[idx] = gx[ nxi + nxj*dli ] for the bra x component nxi and ket x component nxj
    # So the (i,j) in the 2D recurrence is (nxi+nxj, nyi+nyj, nzi+nzj) for the total
    # and we need E2(nxi, nxj) * E2(nyi, nyj) * E2(nzi, nzj)
    # The libcint layout: for compound index n = (j_f,j) * nfi + (i_f,i), they store
    # gx at g_stride_i * i_nx[i] + g_stride_j * j_nx[j] = nxi + dli * nxj
    # So gx[nxi + dli*nxj] is the E2_x(nxi, nxj) value... but the recurrence builds
    # E2(i+j) compound. Let me check: in CINTg1e_ovlp, they have one index i from 0 to nmax
    # and j from 0 to lj. So g[i*di + j*dj] = E2(i, j) for the "vertical" index i and
    # "horizontal" index j. Actually in the Obara-Saika notation, we have E(0,0) and
    # build E(i,j) for i+j <= nmax. The libcint stores it as g[ptr] where ptr = i*di + j*dj.
    # For the final output, we need E(nxi,nxj)*E(nyi,nyj)*E(nzi,nzj). So the index into g
    # for (nxi,nxj) is gx[nxi + dli*nxj] when we use the libcint layout with j being
    # the "other" index. Actually in libcint when li>=lj, di=1, dj=dli, rx=ri.

    # Re-check: envs->nf = nfi * nfj. The idx has 3 entries per n. idx[n*3+0] is the
    # index into gx. So for cartesian bra i (with nx,ny,nz) and ket j (with nx',ny',nz'):
    # idx points to gx at offset that corresponds to (nx, nx') for the x component.
    # The recurrence produces g such that g[di*ix + dj*jx] = E2_x(ix, jx).
    # With di=1, dj=dli: g[ix + dli*jx]. So E2_x(bra_x, ket_x) = gx[bra_x + dli*ket_x].

    S = xp.zeros((nfi, nfj))
    for ia, (nxi, nyi, nzi) in enumerate(i_orbs):
        for ib, (nxj, nyj, nzj) in enumerate(j_orbs):
            # Libcint layout when li>=lj: bra is "i", ket is "j"
            # g_stride_i=1, g_stride_j=dli. Index for (ix, jx) = ix*1 + jx*dli
            # But we have li, lj - when li>=lj the "i" in the recursion is the bra (li)
            # and "j" is the ket (lj). So E2(nxi, nxj) for x.
            if li >= lj:
                ix_bra, jx_ket = nxi, nxj
                iy_bra, jy_ket = nyi, nyj
                iz_bra, jz_ket = nzi, nzj
            else:
                ix_bra, jx_ket = nxj, nxi
                iy_bra, jy_ket = nyj, nyi
                iz_bra, jz_ket = nzj, nzi
            val = (
                gx[ix_bra + dli * jx_ket]
                * gy[iy_bra + dli * jy_ket]
                * gz[iz_bra + dli * jz_ket]
            )
            S[ia, ib] = fac * val
    return S


def _cart_to_idx(l: int) -> dict[tuple[int, int, int], int]:
    """Map (nx, ny, nz) to Cartesian index for angular momentum l."""
    return {coord: i for i, coord in enumerate(_cart_comp(l))}


def _apply_x1i(
    xp,
    g: np.ndarray | "cp.ndarray",
    g_ext: np.ndarray | "cp.ndarray",
    ri: tuple[float, float, float],
    r_origin: tuple[float, float, float],
    li: int,
    lj: int,
    axis: int,
) -> np.ndarray | "cp.ndarray":
    """
    Compute dipole: f = g_ext[(a+1)|b] + (ri - r_origin)[axis] * g[(a|b)].
    g_ext is overlap for bra (li+1), g is overlap for bra (li).
    """
    ri_origin = ri[axis] - r_origin[axis]
    i_orbs = _cart_comp(li)
    # Index in extended basis (li+1) for (nxi+1, nyi, nzi) etc
    cart_to_idx_ext = _cart_to_idx(li + 1)

    f = xp.zeros_like(g)
    for ia, (nxi, nyi, nzi) in enumerate(i_orbs):
        if axis == 0:
            coord_plus = (nxi + 1, nyi, nzi)
        elif axis == 1:
            coord_plus = (nxi, nyi + 1, nzi)
        else:
            coord_plus = (nxi, nyi, nzi + 1)

        ia_plus = cart_to_idx_ext.get(coord_plus)
        for ib in range(g.shape[1]):
            if ia_plus is not None:
                f[ia, ib] = g_ext[ia_plus, ib] + ri_origin * g[ia, ib]
            else:
                f[ia, ib] = ri_origin * g[ia, ib]
    return f


def _shell_pair_dipole(
    xp,
    mol_bra,
    mol_ket,
    ish: int,
    jsh: int,
    r_origin: tuple[float, float, float],
) -> tuple[np.ndarray | "cp.ndarray", np.ndarray | "cp.ndarray", np.ndarray | "cp.ndarray"]:
    """Compute dipole integrals for one shell pair. Returns (Dx, Dy, Dz) in Cartesian."""
    bas_bra = mol_bra._bas
    bas_ket = mol_ket._bas
    env_bra = mol_bra._env
    env_ket = mol_ket._env
    atm_bra = mol_bra._atm
    atm_ket = mol_ket._atm

    li = int(bas_bra[ish, ANG_OF])
    lj = int(bas_ket[jsh, ANG_OF])
    nprim_i = int(bas_bra[ish, NPRIM_OF])
    nprim_j = int(bas_ket[jsh, NPRIM_OF])
    nctr_i = int(bas_bra[ish, NCTR_OF])
    nctr_j = int(bas_ket[jsh, NCTR_OF])

    atom_i = int(bas_bra[ish, ATOM_OF])
    atom_j = int(bas_ket[jsh, ATOM_OF])
    ptr_ri = int(atm_bra[atom_i, PTR_COORD])
    ptr_rj = int(atm_ket[atom_j, PTR_COORD])
    ri = tuple(float(env_bra[ptr_ri + k]) for k in range(3))
    rj = tuple(float(env_ket[ptr_rj + k]) for k in range(3))

    ptr_exp_i = bas_bra[ish, PTR_EXP]
    ptr_exp_j = bas_ket[jsh, PTR_EXP]
    ptr_coeff_i = bas_bra[ish, PTR_COEFF]
    ptr_coeff_j = bas_ket[jsh, PTR_COEFF]

    nfi = _ncart(li)
    nfj = _ncart(lj)
    Dx = xp.zeros((nfi, nfj))
    Dy = xp.zeros((nfi, nfj))
    Dz = xp.zeros((nfi, nfj))

    # For dipole (a|r|b), we need (a+1_axis|b) + (R_i - R_orig) * (a|b).
    # So we need overlap for extended bra (li+1, lj) to get (a+1|b) values.
    for ip in range(nprim_i):
        for jp in range(nprim_j):
            ai_val = env_bra[ptr_exp_i + ip]
            aj_val = env_ket[ptr_exp_j + jp]
            # PySCF coeff layout: [nctr, nprim] -> env[ptr + ic*nprim + ip]
            ci = env_bra[ptr_coeff_i + ip] if nctr_i > 0 else 1.0
            cj = env_ket[ptr_coeff_j + jp] if nctr_j > 0 else 1.0
            # Overlap with extended bra to get (a+1|b) for x1i recurrence
            S_ext = _overlap_primitive(
                xp, ai_val, aj_val, ri, rj, ci, cj, li + 1, lj
            )
            S = _overlap_primitive(
                xp, ai_val, aj_val, ri, rj, ci, cj, li, lj
            )
            # D_axis = (a+1_axis|b) + (R_i - R_orig)_axis * (a|b)
            Dx += _apply_x1i(xp, S, S_ext, ri, r_origin, li, lj, 0)
            Dy += _apply_x1i(xp, S, S_ext, ri, r_origin, li, lj, 1)
            Dz += _apply_x1i(xp, S, S_ext, ri, r_origin, li, lj, 2)

    return Dx, Dy, Dz


def _cart2sph_transform(xp, l: int) -> np.ndarray | "cp.ndarray":
    """Get Cartesian-to-spherical transform matrix for angular momentum l."""
    c2s = gto.cart2sph(l)
    return xp.asarray(c2s)


def int1e_r_gpu(
    mol_bra,
    mol_ket,
    xp=None,
    r_origin: tuple[float, float, float] | None = None,
) -> np.ndarray | "cp.ndarray":
    """
    Compute dipole moment integrals ⟨i|r - r_origin|j⟩ between two molecules.

    Port of PySCF libcint intor_cross('int1e_r'). Returns (3, n_bra, n_ket) in Bohr.

    Args:
        mol_bra: PySCF Mole object (bra basis, e.g. helper_mol)
        mol_ket: PySCF Mole object (ket basis, e.g. auxmol_ml)
        xp: Array module (numpy or cupy). If None, uses numpy.
        r_origin: Origin for dipole (r - r_origin). Default (0,0,0).

    Returns:
        Array of shape (3, n_ao_bra, n_ao_ket) for x, y, z components.
    """
    if xp is None:
        xp = _cp if _CUPY_AVAILABLE and _cp is not None else np
    if r_origin is None:
        r_origin = (0.0, 0.0, 0.0)

    ao_loc_bra = mol_bra.ao_loc
    ao_loc_ket = mol_ket.ao_loc
    nbra = ao_loc_bra[-1]
    nket = ao_loc_ket[-1]
    out = xp.zeros((3, nbra, nket))

    for ish in range(mol_bra.nbas):
        for jsh in range(mol_ket.nbas):
            Dx_cart, Dy_cart, Dz_cart = _shell_pair_dipole(
                xp, mol_bra, mol_ket, ish, jsh, r_origin
            )
            li = mol_bra.bas_angular(ish)
            lj = mol_ket.bas_angular(jsh)

            c2s_i = _cart2sph_transform(xp, li)
            c2s_j = _cart2sph_transform(xp, lj)

            # Transform to spherical: D_sph = c2s_i @ D_cart @ c2s_j.T
            # Skip transform when molecule uses Cartesian basis (mol.cart=True)
            start_i = ao_loc_bra[ish]
            start_j = ao_loc_ket[jsh]
            ni = ao_loc_bra[ish + 1] - start_i
            nj = ao_loc_ket[jsh + 1] - start_j
            use_c2s_bra = ni == (2 * li + 1)  # spherical bra
            use_c2s_ket = nj == (2 * lj + 1)  # spherical ket
            # PySCF cart2sph(l) returns (n_cart, n_sph); D_sph = c.T @ D_cart @ c
            if use_c2s_bra and use_c2s_ket:
                Dx_sph = c2s_i.T @ Dx_cart @ c2s_j
                Dy_sph = c2s_i.T @ Dy_cart @ c2s_j
                Dz_sph = c2s_i.T @ Dz_cart @ c2s_j
            elif use_c2s_bra:
                Dx_sph = c2s_i.T @ Dx_cart
                Dy_sph = c2s_i.T @ Dy_cart
                Dz_sph = c2s_i.T @ Dz_cart
            elif use_c2s_ket:
                Dx_sph = Dx_cart @ c2s_j
                Dy_sph = Dy_cart @ c2s_j
                Dz_sph = Dz_cart @ c2s_j
            else:
                Dx_sph = Dx_cart
                Dy_sph = Dy_cart
                Dz_sph = Dz_cart

            out[0, start_i : start_i + ni, start_j : start_j + nj] = Dx_sph
            out[1, start_i : start_i + ni, start_j : start_j + nj] = Dy_sph
            out[2, start_i : start_i + ni, start_j : start_j + nj] = Dz_sph

    return out
