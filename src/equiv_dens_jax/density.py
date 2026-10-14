"""Portable JAX implementation of DenSnet's density coefficients and expansion.

Species selection and all ragged orbital packing are prepared on the host by
``prepare_layout``. The resulting dictionaries contain only static metadata and
are intended to be closed over by ``jax.jit`` functions.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.nn as jnn

from .common import attr, child, buffer
from .layers import apply as apply_layer
from .harmonics import spherical_harmonics, spherical_harmonics_deriv
from .gaussians import gaussian_rbf, gaussian_rbf_deriv, gto_norm, BOHR_ANGSTROM


def _basis(attrs):
    basis = attrs.get("orbital_basis")
    if basis is None:
        raise ValueError("density node has no orbital_basis metadata")
    return {int(z): [tuple(map(int, orb)) for orb in orbs] for z, orbs in basis.items()}


def prepare_layout(node, atoms):
    """Prepare static per-slot species and orbital packing for a fixed atom layout."""
    basis = _basis(node["attrs"])
    nums = atoms.get("batch_atom_numbers", atoms.get("atom_numbers"))
    if nums is None:
        raise KeyError("atoms requires batch_atom_numbers or atom_numbers")
    nums = jax.device_get(nums)
    # Match the source's max over batch for each atom slot, not per-example species.
    species = tuple(int(x) for x in nums.max(axis=0).tolist())
    # The converted module's orbital_basis already reflects constructor-time
    # core_basis_ratio filtering; applying that ratio again would filter twice.
    max_order = max((orb[2] for orbs in basis.values() for orb in orbs), default=0)
    shell_counts, radial_counts, r_max = {}, {}, {}
    sph_index, rad_index = {}, {}
    next_sph, next_rad = [0] * (max_order + 1), [0] * (max_order + 1)
    for z, orbs in basis.items():
        grouped = {}
        for _, n, L in orbs:
            grouped.setdefault(L, []).append(n)
        for L, counts in grouped.items():
            key = (z, L)
            shell_counts[key] = tuple(counts)
            sph_index[key] = tuple(range(next_sph[L], next_sph[L] + len(counts)))
            next_sph[L] += len(counts)
            radial_counts[key] = tuple(counts)
            rad_index[key] = tuple(range(next_rad[L], next_rad[L] + sum(counts)))
            next_rad[L] += sum(counts)
            r_max[key] = max(counts, default=0)
    per_species = {z: {"orbs": tuple(orbs), "keys": tuple(dict.fromkeys((z, o[2]) for o in orbs)),
                       "max_order": max((o[2] for o in orbs), default=0)} for z, orbs in basis.items()}
    # Slot may be padding (species zero), and source skips it entirely.
    idx = atoms.get("batch_idx_pos")
    idx = None if idx is None else tuple(int(x) for x in jax.device_get(idx).tolist())
    return {"species": species, "first_row_species": tuple(int(x) for x in nums[0].tolist()),
            "batch_idx_pos": idx, "basis": basis, "by_species": per_species,
            "max_order": max_order, "shell_counts": shell_counts,
            "radial_counts": radial_counts, "sph_index": sph_index,
            "rad_index": rad_index, "r_max": r_max,
            "atom_count": len(species), "batch_size": int(nums.shape[0])}


def _array_dict_for_atoms(atoms, key, layout):
    values = []
    for i, z in enumerate(layout["species"]):
        values.append({} if z == 0 else atoms[key][i])
    return values


def coefficients_dict_to_vector(atoms, layout, *, weights=False):
    """Pack atom dictionaries in legacy PySCF shell order, preserving odd-L sign."""
    name = "coeff_weights" if weights else "spherical_coeffs"
    pieces = []
    for i, z in enumerate(layout["species"]):
        if z == 0:
            continue
        per_key_count = {}
        for _, _, L in layout["basis"][z]:
            key = (z, L)
            n = per_key_count.get(key, 0)
            x = atoms[name][i][key][..., n:n+1]
            # [B,1,2L+1,1] to [B,2L+1]; PySCF p order and phase.
            x = x[:, 0, :, 0]
            if L == 1:
                x = x[:, [2, 0, 1]]
            if L % 2:
                x = -x
            pieces.append(x)
            per_key_count[key] = n + 1
    return jnp.concatenate(pieces, axis=-1) if pieces else jnp.zeros((layout["batch_size"], 0))


def coefficients_to_dict(vector, layout, radial_basis, *, dtype=None):
    """Unpack PySCF-order coefficients plus fixed radial basis data to atom dicts."""
    vector = jnp.asarray(vector, dtype=dtype)
    out = {"spherical_coeffs": [], "radial_width": [], "radial_scale": []}
    offset = 0
    for z in layout["species"]:
        sph, widths, scales = {}, {}, {}
        if z:
            seen = {}
            for orbital_slot, (_, n, L) in enumerate(layout["basis"][z]):
                key = (z, L); slot = seen.get(key, 0); seen[key] = slot + 1
                count = 2 * L + 1
                x = vector[:, offset:offset+count]
                offset += count
                if L % 2: x = -x
                if L == 1: x = x[:, [1, 2, 0]]
                sph[key] = jnp.concatenate((sph[key], x[:, None, :, None]), axis=-1) if key in sph else x[:, None, :, None]
                rb = radial_basis[z][orbital_slot]
                w, s = jnp.asarray(rb[0], dtype=vector.dtype), jnp.asarray(rb[1], dtype=vector.dtype)
                widths[key] = jnp.concatenate((widths[key], w.reshape((1,1,-1,1))), axis=-1) if key in widths else w.reshape((1,1,-1,1))
                scales[key] = jnp.concatenate((scales[key], s.reshape((1,1,-1,1))), axis=-1) if key in scales else s.reshape((1,1,-1,1))
            # Historical vector reconstruction gives batch-specific coeffs but fixed radial values.
            B = vector.shape[0]
            widths = {k: jnp.broadcast_to(v, (B,*v.shape[1:])) for k,v in widths.items()}
            scales = {k: jnp.broadcast_to(v, (B,*v.shape[1:])) for k,v in scales.items()}
        out["spherical_coeffs"].append(sph); out["radial_width"].append(widths); out["radial_scale"].append(scales)
    if offset != vector.shape[-1]:
        raise ValueError(f"coefficient vector has {vector.shape[-1]-offset} trailing values")
    return out


def _get_numbers(atoms):
    return atoms.get("batch_atom_numbers", atoms.get("atom_numbers"))


def _get_positions(atoms):
    return atoms.get("batch_positions", atoms.get("positions"))


def _masked(value, atoms, i):
    mask = atoms.get("batch_atom_mask")
    if mask is None: return value
    m = jnp.asarray(mask[:, i], dtype=value.dtype).reshape((value.shape[0],) + (1,) * (value.ndim-1))
    return value * m


def _integral_constraint(node, state, atoms, layout):
    integ = node["children"].get("integral_constraint")
    if not integ: return atoms
    coeffs, widths, scales = [], [], []
    refs = []
    for i,z in enumerate(layout["species"]):
        if z == 0: continue
        for key in atoms["spherical_coeffs"][i]:
            if key[1] == 0:
                coeffs.append((atoms["spherical_coeffs"][i][key] * atoms["radial_scale"][i][key]).reshape((layout["batch_size"],-1)))
                widths.append(atoms["radial_width"][i][key].reshape((layout["batch_size"],-1)))
                scales.append(atoms["radial_scale"][i][key].reshape((layout["batch_size"],-1)))
                refs.append((i,key,atoms["spherical_coeffs"][i][key].shape[-1]))
    if not coeffs: return atoms
    c,w,s = jnp.concatenate(coeffs,-1), jnp.concatenate(widths,-1), jnp.concatenate(scales,-1)
    norms = 1 / gto_norm(0,w) / (2*jnp.sqrt(jnp.pi))
    if attr(integ,"remove_atom_density",False):
        v = jnp.where(s != 0,c/jnp.where(s != 0,s,1),0)
        q = s*norms
        v = v - (jnp.sum(v*q,-1,keepdims=True)/jnp.sum(q*q,-1,keepdims=True))*q
        ptr = 0
        # Deliberately preserve source's pointer advance by shell count, including its quirk.
        for i,key,n in refs:
            sl = v[:,ptr:ptr+n]
            atoms["spherical_coeffs"][i][key] = sl[:,None,None,:]
            ptr += n
    else:
        factor = jnp.sum(_get_numbers(atoms),axis=-1,keepdims=True) / jnp.sum(c*norms,axis=-1,keepdims=True)
        factor = factor[:,:,None,None]
        # Source clamps integral_scale but does not apply that clamped value.
        for i,key,_ in refs:
            atoms["spherical_coeffs"][i][key] = atoms["spherical_coeffs"][i][key]*factor
    return atoms


def _extract(node,state,atoms,layout,fs):
    out_sph = apply_layer(child(node,"spherical_output"),state,fs)
    if "linear_output" in node["children"]:
        out_sph = apply_layer(child(node,"linear_output"),state,out_sph)
    if attr(node,"scale_sph_order",True): out_sph = [x*(10.**-L) for L,x in enumerate(out_sph)]
    if attr(node,"positive_coeffs",False): out_sph[0] = jnn.softplus(out_sph[0])
    pred = attr(node,"pred_radial_coeffs",True)
    rwidth, rscale = [], []
    if pred:
        widths = child(node,"radial_width")["children"].values()
        scales = child(node,"radial_scale")["children"].values()
        for w,s in zip(widths,scales):
            rwidth.append(jnp.tanh(apply_layer(w,state,fs[0])))
            raw = apply_layer(s,state,fs[0])
            rscale.append(jnp.tanh(raw) if attr(node,"positive_coeffs",False) else raw)
    result = {k: [] for k in ("spherical_coeffs","radial_width","radial_scale","coeff_weights")}
    for i,z in enumerate(layout["species"]):
        sc,rw,rs,cw = {},{},{},{}
        if z:
            mask = atoms.get("batch_atom_mask")
            for key in layout["by_species"][z]["keys"]:
                L=key[1]; inds=layout["sph_index"][key]; counts=layout["shell_counts"][key]
                sc[key] = out_sph[L][:,i:i+1,:,jnp.array(inds)]
                rcount=layout["r_max"][key]
                raw_w=jnp.zeros((fs[0].shape[0],1,rcount,len(counts)),dtype=fs[0].dtype)
                raw_s=jnp.zeros_like(raw_w)
                if pred:
                    rinds=layout["rad_index"][key]
                    # source destination order: for each shell, its radial prefix is flattened row-major
                    destinations=[r*len(counts)+sh for sh,n in enumerate(counts) for r in range(n)]
                    vals_w=rwidth[L][:,i,0,jnp.array(rinds)]
                    vals_s=rscale[L][:,i,0,jnp.array(rinds)]
                    raw_w=raw_w.reshape((fs[0].shape[0],1,-1)).at[:,:,jnp.array(destinations)].set(vals_w[:,None,:])
                    raw_s=raw_s.reshape((fs[0].shape[0],1,-1)).at[:,:,jnp.array(destinations)].set(vals_s[:,None,:])
                    raw_w=raw_w.reshape((fs[0].shape[0],1,rcount,len(counts)))
                    raw_s=raw_s.reshape((fs[0].shape[0],1,rcount,len(counts)))
                if attr(node,"coeff_weights",None) is not None:
                    cw[key] = jnp.sum(buffer(node,state,f"coeff_weights_{z}_{L}"),axis=-2,keepdims=True)
                    cw[key] = jnp.broadcast_to(cw[key],(cw[key].shape[0],1,2*L+1,len(counts)))
                init_s=buffer(node,state,f"init_sph_{z}_{L}") if f"init_sph_{z}_{L}" in node["buffers"] else None
                if init_s is not None and L==0:
                    sc[key]=sc[key]+init_s
                    if pred:
                        raw_w=jnp.clip(raw_w+buffer(node,state,f"init_width_{z}_{L}"),-.999999,.99999)
                        raw_s=raw_s+buffer(node,state,f"init_scale_{z}_{L}")
                if attr(node,"radial_coeffs",None) is not None:
                    iw=buffer(node,state,f"init_radial_width_{z}_{L}"); ins=buffer(node,state,f"init_radial_scale_{z}_{L}")
                    offset=attr(node,"ml_width_offset",0); width_scale=attr(node,"ml_width_scale",1)
                    raw_w=(raw_w+1)*width_scale+offset; raw_w=raw_w*iw
                    if attr(node,"positive_coeffs",False): raw_s=((raw_s+1)*width_scale+offset)*ins
                    else: raw_s=raw_s+ins
                    if mask is not None: raw_s=raw_s*jnp.asarray(mask[:,i],fs[0].dtype)[:,None,None,None]
                rw[key],rs[key]=raw_w,raw_s
        result["spherical_coeffs"].append(sc); result["radial_width"].append(rw); result["radial_scale"].append(rs); result["coeff_weights"].append(cw)
    return result


def apply_density_head(node,state,atoms,layout=None):
    """Compute physical coefficients, radial functions, constraints and DF vector."""
    atoms=dict(atoms); layout=layout or prepare_layout(node,atoms)
    # Copy nested coefficient maps before any updates; shallow dict copies leak
    # JAX tracers when a delta/constraint head runs under grad or jit.
    for name in ("spherical_coeffs","radial_width","radial_scale","coeff_weights"):
        if name in atoms:
            atoms[name]=[dict(item) for item in atoms[name]]
    if node["type"] in ("DFDensityCoeffs","FreeAtomDensityCoeffs"):
        radial_basis=attr(node,"radial_coeffs",None)
        if node["type"]=="FreeAtomDensityCoeffs":
            atom_dens=attr(node,"atom_dens",{})
            numbers=_get_numbers(atoms); zs=layout["first_row_species"]
            vec=jnp.concatenate([jnp.asarray(atom_dens[int(z)]["df_coeffs"]).reshape(-1) for z in zs])[None,:]
            vec=jnp.broadcast_to(vec,(numbers.shape[0],vec.shape[-1]))
            atoms["df_coeffs"]=vec
        coeffs=coefficients_to_dict(atoms["df_coeffs"],layout,radial_basis)
        atoms.update(coeffs)
        atoms["sph_dict"]=layout["sph_index"]
        return _set_geometry(atoms,layout)
    if node["type"]!="DensityCoeffsNetwork": raise NotImplementedError(node["type"])
    fs=atoms.get("sph_repr_batch",atoms["sph_repr"])
    # Legacy representations return packed active atoms [1,Nactive,m,F]; the
    # Torch head scatters them back to [B,N,m,F] before coefficient extraction.
    if layout["batch_idx_pos"] is not None and (
            fs[0].shape[0] != layout["batch_size"] or fs[0].shape[1] != layout["atom_count"]):
        packed_index=jnp.asarray(layout["batch_idx_pos"],dtype=jnp.int32)
        scattered=[]
        for x in fs:
            flat=jnp.zeros((layout["batch_size"]*layout["atom_count"],*x.shape[2:]),dtype=x.dtype)
            flat=flat.at[packed_index].set(x.reshape((-1,*x.shape[2:])))
            scattered.append(flat.reshape((layout["batch_size"],layout["atom_count"],*x.shape[2:])))
        fs=scattered
    coeffs=_extract(node,state,atoms,layout,fs)
    if "spherical_coeffs" in atoms:
        for name in ("spherical_coeffs","radial_width","radial_scale","coeff_weights"):
            if name in atoms:
                for i in range(layout["atom_count"]):
                    for k,v in coeffs[name][i].items():
                        atoms[name][i][k]=jnp.concatenate((v,atoms[name][i][k]),axis=-1)
    else:
        atoms.update(coeffs)
        if node["children"].get("integral_constraint"):
            atoms=_integral_constraint(node,state,atoms,layout)
        atoms["df_coeffs"]=coefficients_dict_to_vector(atoms,layout)
        if attr(node,"coeff_weights",None) is not None:
            atoms["df_weights"]=coefficients_dict_to_vector(atoms,layout,weights=True)
    atoms["sph_dict"]=layout["sph_index"]; atoms["rad_dict"]=layout["rad_index"]
    return atoms


def _set_geometry(atoms,layout):
    R=atoms.get("positions"); idx_i=atoms.get("idx_i"); idx_j=atoms.get("idx_j")
    if R is not None and idx_i is not None and idx_j is not None:
        disp=jnp.take(R,idx_j,axis=-2)-jnp.take(R,idx_i,axis=-2)
        d=jnp.linalg.norm(disp,axis=-1,keepdims=True)
        atoms.update(distances=d,directions=disp/d)
        atoms["sph"]=[y[...,None] for y in spherical_harmonics(layout["max_order"],disp/d)]
    atoms.pop("density",None)
    return atoms


def _eval_key(coords,pos,coeff,width,scale,L,need_grad):
    rel=coords-pos; d=jnp.linalg.norm(rel,axis=-1,keepdims=True)
    u=jnp.where(d!=0,-rel/jnp.where(d!=0,d,1),0)
    sphs=spherical_harmonics(L,u); sph=sphs[L]
    coeff=coeff[:,0,:,:] # [B,m,n]
    width=width[:,0,:,:]; scale=scale[:,0,:,:]
    # B,G,1,1,1 against B,1,1,r,n; sum radial basis axis.
    rb=gaussian_rbf(d[...,None],width[:,None,:,:],scale[:,None,:,:],L)
    part=jnp.sum(rb*sph[...,None]*coeff[:,None,:,:],axis=(-2,-1))
    if not need_grad: return part,None
    deriv=spherical_harmonics_deriv(L,u)[L]
    eye=jnp.eye(3,dtype=coords.dtype)
    grad_u=-(eye/d[...,None] - rel[..., :,None]*rel[...,None,:]/d[...,None]**3)
    grad_u=jnp.where(jnp.isfinite(grad_u),grad_u,0)
    sph=sph[...,None]*coeff[:,None,:,:]
    sph_grad=deriv[...,None]*coeff[:,None,:,None,:]
    # Keep the legacy tensor contractions and reduction order exactly.
    sph_grad_c=jnp.sum(sph_grad[:,:,None,:,:,:]*grad_u[:,:,:,None,:,None],axis=-2)
    dr=gaussian_rbf_deriv(d[...,None],width[:,None,:,:],scale[:,None,:,:],L)
    grad_r=dr * (rel/jnp.where(d!=0,d,1))[..., :, None]
    radial=jnp.sum(grad_r[:,:,:,None,:]*sph[:,:,None,:,:],axis=-2)
    angular=jnp.sum(rb[:,:,None,:,:]*sph_grad_c,axis=-2)
    return part,jnp.sum(radial+angular,axis=-1)*BOHR_ANGSTROM


def apply_density_expansion(node,state,atoms,layout=None,*,eval_atoms=None,eval_L=None,density_grad=None):
    """Evaluate Gaussian density and the historical explicit spatial gradient."""
    atoms=dict(atoms); layout=layout or prepare_layout(node,atoms)
    density_grad=attr(node,"density_grad",False) if density_grad is None else density_grad
    coords=atoms["coords"]; numbers=_get_numbers(atoms); positions=_get_positions(atoms)
    if density_grad and layout["max_order"] > 5:
        raise NotImplementedError("legacy explicit density gradients support angular order L<=5")
    rho=jnp.zeros(coords.shape[:-1],dtype=coords.dtype)
    grad=jnp.zeros_like(coords) if density_grad else None
    eval_atoms=set(range(layout["atom_count"])) if eval_atoms is None else set(eval_atoms)
    # Every slot's species is fixed by layout exactly as in legacy extraction.
    for i,z in enumerate(layout["species"]):
        if z==0 or i not in eval_atoms: continue
        pos=positions[:,i:i+1,:]
        for key in layout["by_species"][z]["keys"]:
            L=key[1]
            if eval_L is not None and L not in eval_L: continue
            rho_i,grad_i=_eval_key(coords,pos,atoms["spherical_coeffs"][i][key],atoms["radial_width"][i][key],atoms["radial_scale"][i][key],L,density_grad)
            rho=rho+rho_i
            if density_grad: grad=grad+grad_i
    constraint=attr(node,"expansion_constraint",None)
    if constraint=="sq": rho=rho**2
    elif constraint=="abs": rho=jnp.sqrt(rho**2)
    elif constraint=="sp": rho=jnn.softplus(1e8*rho)/1e8+1e-30
    elif constraint not in (None,"none"): raise ValueError(f"unsupported density constraint {constraint!r}")
    if attr(node,"integral_constraint",None)=="grid":
        ne=jnp.sum(numbers,axis=-1,keepdims=True)
        weights=atoms["coord_weights"]
        scaling=ne/jnp.sum(rho*weights,axis=1,keepdims=True)
        # The source trainer anneals this buffer after each training step.  A
        # per-batch input keeps that update dynamic under a compiled JAX step;
        # absent an override, retain the serialized buffer/attribute default.
        factor=atoms.get("density_grid_scaling_factor")
        if factor is None:
            factor=buffer(node,state,"grid_scaling_factor") if "grid_scaling_factor" in node["buffers"] else attr(node,"grid_scaling_factor",0)
        factor=jnp.asarray(factor,dtype=rho.dtype)
        rho=rho*(scaling*(1-factor)+factor)
    atoms["density"]=rho
    if density_grad: atoms["density_grad"]=grad
    return atoms
