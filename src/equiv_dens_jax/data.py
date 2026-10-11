"""Torch-free preparation and loading of current DenSnet array batches."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .serialization import load_tree


def prepare_atoms(atoms, *, cutoff=None):
    """Create the existing compressed batch and graph layout from NumPy arrays.

    Already-prepared current batches pass through unchanged. Raw positions are
    batch-first [B,N,3], with zero atomic numbers denoting padding. Molecules in
    a batch must use compatible species columns, as required by the current head.
    ``cutoff`` is optional: without it the exact complete directed graph is used.
    Edges never connect different molecules and include no self edges.
    """
    atoms = dict(atoms)
    if "batch_atom_numbers" in atoms and "batch_idx_pos" in atoms:
        return atoms
    numbers = np.asarray(atoms["atom_numbers"],dtype=np.int64)
    positions = np.asarray(atoms["positions"])
    if numbers.ndim == 1:
        numbers = numbers[None]
    if positions.ndim == 2:
        positions = positions[None]
    if numbers.shape != positions.shape[:-1] or positions.shape[-1] != 3:
        raise ValueError("atom_numbers and positions shapes do not match")
    if not np.isfinite(positions).all() or np.any(numbers<0):
        raise ValueError("Invalid atomic numbers or nonfinite positions")
    if np.any(np.sum(numbers>0,axis=1)==0):
        raise ValueError("Every molecule needs at least one atom")
    for column in numbers.T:
        if len(set(column[column>0]))>1:
            raise ValueError("Mixed species in one batch column: align species slots before batching")
    mask = numbers>0
    active = np.flatnonzero(mask.ravel())
    packed = positions.reshape(-1,3)[active]
    packed_numbers = numbers.ravel()[active]
    batch_indices = np.repeat(np.arange(len(numbers)),numbers.shape[1])[active]
    receiver,sender=[],[]
    for batch in range(len(numbers)):
        indices = np.flatnonzero(batch_indices==batch)
        for i in indices:
            for j in indices:
                if i != j and (cutoff is None or np.linalg.norm(packed[j]-packed[i])<cutoff):
                    receiver.append(i)
                    sender.append(j)
    first = {}
    for index,z in enumerate(numbers.max(axis=0)):
        if z:
            first.setdefault(int(z),index)
    atoms.update(positions=packed[None],atom_numbers=packed_numbers[None],
                 atom_mask=mask.ravel(),batch_atom_mask=mask,batch_atom_numbers=numbers,
                 batch_positions=positions,batch_idx_pos=active,atom_batch_idx=batch_indices[None],
                 atom_numbers_first_positions=first,
                 idx_i=np.asarray(receiver,dtype=np.int64),idx_j=np.asarray(sender,dtype=np.int64))
    if "forces" in atoms:
        forces = np.asarray(atoms["forces"])
        if forces.ndim==2:
            forces=forces[None]
        if forces.shape!=positions.shape:
            raise ValueError("Raw force labels must match raw positions")
        atoms["forces"] = forces.reshape(-1,3)[active][None]
    for field in ("coords","density","coord_weights","density_grad"):
        if field in atoms:
            value = np.asarray(atoms[field])
            expected = 3 if field in {"coords","density_grad"} else 2
            if value.ndim==expected-1:
                value=value[None]
            if value.shape[0]!=len(numbers):
                raise ValueError(f"{field} batch count does not match molecules")
            atoms[field]=value
    return atoms


def unpack_atoms(value, atoms):
    """Restore [B,N,...] arrays without assigning one molecule's atoms to another."""
    numbers=np.asarray(atoms["batch_atom_numbers"])
    indices=np.asarray(atoms["batch_idx_pos"],dtype=np.int64)
    value=np.asarray(value)
    if value.shape[:2]!=(1,len(indices)):
        raise ValueError("Expected a compressed [1,total_active,...] array")
    out=np.zeros((numbers.size,*value.shape[2:]),dtype=value.dtype)
    out[indices]=value[0]
    return out.reshape((*numbers.shape,*value.shape[2:]))


def load_batches(path):
    """Load prepared JSON/NPZ bundles or one raw array NPZ batch.

    A bundle may contain ``batches`` (list), ``inputs`` (one batch), or an atom
    dictionary itself. NPZ loading never enables pickle/object arrays.
    """
    path=Path(path)
    if path.is_dir():
        tree=load_tree(path)
        batches=tree.get("batches",[tree.get("inputs",tree)])
    else:
        with np.load(path,allow_pickle=False) as archive:
            batches=[{key:archive[key].copy() for key in archive.files}]
    if not batches:
        raise ValueError("Dataset contains no batches")
    return [prepare_atoms(batch) for batch in batches]


def sample_grid(atoms, count, *, seed=0):
    """Explicit deterministic subsampling of provided grids and aligned labels."""
    if count<=0:
        raise ValueError("Grid sample count must be positive")
    result=dict(atoms)
    size=np.asarray(atoms["coords"]).shape[1]
    if count>size:
        raise ValueError("Sample count exceeds the supplied grid")
    indices=np.random.default_rng(seed).choice(size,count,replace=False)
    for field in ("coords","coord_weights","density","density_grad","atom_density","atom_density_grad"):
        if field in result:
            result[field]=np.asarray(result[field])[:,indices]
    return result
