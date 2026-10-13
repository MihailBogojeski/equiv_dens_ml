"""gpu4pyscf versions of the PySCF routines on the data-loading hot path.

Only import this module when utils.base.USE_GPU4PYSCF is True.
"""
import numpy as np
import torch
import equiv_dens.compat  # noqa: F401  (before cupy/gpu4pyscf)
import cupy as cp
from cupyx.scipy.interpolate import BSpline as CPBSpline
from pyscf.lib import param
from gpu4pyscf.dft import numint as gnumint
from gpu4pyscf.dft import gen_grid as ggen_grid
from equiv_dens.utils.hirshfeld_analysis import _ensure_bspline_compatible


def get_opt(mol):
    """GDFTOpt (sorted mol + GPU env) cached on the mol, rebuilt if the mol was rebuilt."""
    cache = getattr(mol, "_gdftopt_cache", None)
    if cache is None or cache[0] is not mol._bas:
        cache = (mol._bas, gnumint._GDFTOpt.from_mol(mol))
        mol._gdftopt_cache = cache
    return cache[1]


def eval_ao(mol, coords, deriv=0):
    return gnumint.eval_ao(mol, cp.asarray(coords, dtype=cp.float64), deriv=deriv, gdftopt=get_opt(mol))


def _rho_mo(ao, mo_coeff, mo_occ, density_grad):
    c = cp.asarray(mo_coeff[:, mo_occ > 0])
    occ = cp.asarray(mo_occ[mo_occ > 0])
    if density_grad:
        c0 = ao[0] @ c
        rho = (c0 * c0 * occ).sum(axis=1)
        grad = [2 * (c0 * (ao[i] @ c) * occ).sum(axis=1) for i in (1, 2, 3)]
        return cp.stack([rho] + grad, axis=0)
    c0 = ao @ c
    return (c0 * c0 * occ).sum(axis=1)


def sample_density(mol, coords, coeffs, projected, density_grad):
    """Density (and gradient) of one molecule on coords [n, 3] (Bohr). Returns cupy
    array [n] or [4, n] (CPU pyscf eval_rho2 layout)."""
    ao = eval_ao(mol, coords, deriv=int(density_grad))
    if projected:
        coeffs = np.atleast_2d(coeffs).sum(axis=0)
        c = cp.asarray(coeffs)
        if density_grad:
            return cp.stack([ao[i] @ c for i in range(4)], axis=0)
        return ao @ c
    if coeffs["mo_occ"].ndim > 1:
        return sum(_rho_mo(ao, coeffs["mo_coeff"][j], coeffs["mo_occ"][j], density_grad)
                   for j in range(coeffs["mo_occ"].shape[0]))
    return _rho_mo(ao, coeffs["mo_coeff"], coeffs["mo_occ"], density_grad)


def grids_to_cupy(grid_spec):
    return {k: (cp.asarray(v[0].numpy()), cp.asarray(v[1].numpy())) for k, v in grid_spec.items()}


def get_partition(mol, grid_spec_cp):
    """Becke partitioned molecular grid on the GPU. Returns cupy (coords [n, 3], weights [n])."""
    return ggen_grid.get_partition(mol, grid_spec_cp)


def cp_to_torch(x):
    return torch.from_dlpack(x)


_spline_cache = {}


def _cp_spline(spl, deriv=False):
    key = (id(spl), deriv)
    if key not in _spline_cache:
        s = _ensure_bspline_compatible(spl)
        if deriv:
            s = _ensure_bspline_compatible(s.derivative())
        _spline_cache[key] = (spl, CPBSpline(cp.asarray(s.t), cp.asarray(s.c), s.k))
    return _spline_cache[key][1]


def eval_spline_density(spl, coords, density_grad=False):
    """GPU version of hirshfeld_analysis.eval_spline_density (coords and result on the GPU)."""
    x_in = torch.norm(coords, dim=-1) / param.BOHR
    x_log = torch.log(x_in).double()
    sp = _cp_spline(spl)
    y_out = torch.from_dlpack(sp(cp.from_dlpack(x_log.contiguous().view(-1))).reshape(x_log.shape))
    y_out = torch.where((y_out < 0) | torch.isnan(y_out), torch.zeros_like(y_out), y_out)
    if density_grad:
        deriv = _cp_spline(spl, deriv=True)
        spl_deriv = torch.from_dlpack(deriv(cp.from_dlpack(x_log.contiguous().view(-1))).reshape(x_log.shape))
        y_deriv = spl_deriv.unsqueeze(-1) * (1 / x_in).unsqueeze(-1) * (coords / x_in.unsqueeze(-1)) / param.BOHR
        y_out = torch.cat([y_out.unsqueeze(-1), y_deriv], dim=-1)
    return y_out
