"""Current density-conditioned DenSnet energy equations, in native JAX.

Prepare packing metadata from a host density-output dictionary once per fixed
species/mask layout. Close over that layout for jit and position derivatives.
Forces are composed outside these heads; density position dependence is retained.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .common import attr, child, parameter
from . import layers


def _torch_abs(value):
    # Torch uses subgradient 0 at exactly zero; JAX abs otherwise chooses +1.
    # Preserve the source's higher-AD convention at structural zero channels.
    return jnp.where(value > 0, value, jnp.where(value < 0, -value, 0))


def prepare_layout(node, atoms_numpy):
    """Freeze the existing coefficient dictionary packing, without reading values.

    ``atoms_numpy`` must include the density head's legacy coefficient/radial
    dictionaries. Its array shapes, dictionary keys and atom mask are metadata.
    """
    mask = np.asarray(atoms_numpy["atom_mask"]).reshape(-1)
    layout = {"active": np.flatnonzero(mask)}
    batches = np.asarray(atoms_numpy["atom_batch_idx"])
    batch_size = atoms_numpy["batch_atom_numbers"].shape[0]
    if np.any(batches < 0) or np.any(batches >= batch_size):
        raise ValueError("Energy atom_batch_idx is outside the fixed batch layout")
    if node["type"] == "RepresentationEnergyNetwork":
        return layout
    coeffs = atoms_numpy["spherical_coeffs"]
    if not coeffs or any(not slot for slot in coeffs):
        raise ValueError("Energy packing requires nonempty coefficient dictionaries per slot")
    keys = tuple(tuple(slot) for slot in coeffs)
    max_order = max(key[1] for slot in keys for key in slot)
    num_coeffs, num_radial = [0] * (max_order + 1), [0] * (max_order + 1)
    atom_df = attr(node, "atom_df_coeffs")
    # Preserve the original helper's species counting and SAD index shifts.
    seen_species = set()
    for slot, slot_keys in zip(coeffs, keys):
        current = [0] * (max_order + 1)
        for key in slot_keys:
            z, L = key
            current[L] += slot[key].shape[-1]
            if L == 0 and atom_df is not None:
                current[0] += np.asarray(atom_df[z]).shape[1]
        if slot_keys[-1][0] not in seen_species:
            seen_species.add(slot_keys[-1][0])
            num_coeffs = [a + b for a, b in zip(num_coeffs, current)]
    for slot in atoms_numpy["radial_scale"]:
        for key, value in slot.items():
            num_radial[key[1]] = max(num_radial[key[1]], value.shape[-2])
    indices = atoms_numpy.get("sph_dict", atoms_numpy.get("L_dict"))
    if indices is None:
        raise ValueError("Energy packing requires sph_dict or L_dict")
    indices = {key: np.asarray(value, dtype=np.int32) for key, value in indices.items()}
    for slot, slot_keys in zip(coeffs, keys):
        for key in slot_keys:
            inds = indices[key]
            if inds.ndim != 1 or len(inds) != slot[key].shape[-1]:
                raise ValueError("Energy feature indices and coefficient channels differ")
            if key[1] == 0 and atom_df is not None:
                if np.asarray(atom_df[key[0]]).shape[-1] != len(inds):
                    raise ValueError("Current SAD append requires equal fixed/learned channel counts")
                shifted = inds + inds[0]
                destinations = np.concatenate((shifted, shifted + len(inds)))
            else:
                destinations = inds
            if (np.any(destinations < 0) or np.any(destinations >= num_coeffs[key[1]])
                    or len(np.unique(destinations)) != len(destinations)):
                raise ValueError("Energy feature indices are outside/duplicate in the current packing")
    if node["type"] != "SphericalLinearEnergyNetwork" and len(mask) != batch_size * len(keys):
        raise ValueError("Energy atom mask and coefficient slot layout differ")
    layout.update(keys=keys, max_order=max_order, num_coeffs=tuple(num_coeffs),
                  num_radial=tuple(num_radial), indices=indices)
    return layout


def coefficients_to_tensors(node, atoms, layout, *, radial_coeffs=True):
    """Exact legacy species-channel packing, including the optional SAD append."""
    first = next(iter(atoms["spherical_coeffs"][0].values()))
    batch_size = first.shape[0]
    spherical, scales, widths = [], [], []
    atom_df = attr(node, "atom_df_coeffs")
    atom_width = attr(node, "atom_df_widths")
    atom_scale = attr(node, "atom_df_scales")
    if atom_width is not None and not radial_coeffs:
        raise NotImplementedError("Current SAD packing requires predicted radial coefficients")
    for L in range(layout["max_order"] + 1):
        sphere_slots, scale_slots, width_slots = [], [], []
        for i, keys in enumerate(layout["keys"]):
            sphere = jnp.zeros((batch_size, 1, 2 * L + 1, layout["num_coeffs"][L]),
                               dtype=first.dtype)
            shape = (batch_size, 1, layout["num_radial"][L], layout["num_coeffs"][L])
            scale = jnp.zeros(shape, dtype=first.dtype)
            width = jnp.zeros(shape, dtype=first.dtype)
            for key in keys:
                if key[1] != L:
                    continue
                inds = layout["indices"][key]
                if L == 0 and atom_df is not None:
                    sad_inds = inds + inds[0]
                    inds = sad_inds + len(inds)
                    sphere = sphere.at[..., sad_inds].set(jnp.asarray(atom_df[key[0]], dtype=first.dtype))
                    if atom_width is not None:
                        width = width.at[..., sad_inds].set(jnp.asarray(atom_width[key[0]], dtype=first.dtype))
                        scale = scale.at[..., sad_inds].set(jnp.asarray(atom_scale[key[0]], dtype=first.dtype))
                sphere = sphere.at[..., inds].set(atoms["spherical_coeffs"][i][key])
                if radial_coeffs:
                    width = width.at[..., inds].set(atoms["radial_width"][i][key])
                    scale = scale.at[..., inds].set(atoms["radial_scale"][i][key])
            sphere_slots.append(sphere)
            scale_slots.append(scale)
            width_slots.append(width)
        spherical.append(jnp.concatenate(sphere_slots, axis=1))
        if radial_coeffs:
            scales.append(jnp.concatenate(scale_slots, axis=1))
            widths.append(jnp.concatenate(width_slots, axis=1))
    return spherical, scales, widths


def _apply_list(node, state, values):
    return [layers.apply(sub, state, value)
            for sub, value in zip(node["children"].values(), values)]


def _scatter_energy(atoms, atom_energy):
    batch_size = atoms["batch_atom_numbers"].shape[0]
    energy = jnp.zeros((batch_size,), dtype=atoms["positions"].dtype)
    return energy.at[atoms["atom_batch_idx"].reshape(-1)].add(atom_energy.reshape(-1))[:, None]


def apply_energy(node, state, atoms, *, layout=None):
    """Return a copied atom dictionary with energy, without an internal force call."""
    kind = node["type"]
    supported = ("SphericalHarmonicsEnergyNetwork", "SphericalHarmonicsEmbeddingEnergyNetwork",
                 "SphericalLinearEnergyNetwork", "RepresentationEnergyNetwork")
    if kind not in supported:
        raise NotImplementedError(f"Unsupported energy head: {kind}")
    result = dict(atoms)
    if kind == "RepresentationEnergyNetwork":
        features = list(atoms["sph_repr"])
        features[0] = layers.apply(child(node, "out_activation"), state, features[0])
        atom_energy = layers.apply(child(node, "energy_output"), state, features)[0][..., 0, 0]
        result["energy"] = _scatter_energy(atoms, atom_energy)
        return result
    if layout is None:
        layout = prepare_layout(node, atoms)
    predicted = attr(node, "pred_radial_coeffs", True)
    spatial = kind != "SphericalLinearEnergyNetwork"
    if spatial and not predicted:
        raise NotImplementedError(
            "Current Torch spherical energy heads index empty radial lists when "
            "pred_radial_coeffs=False; this historical configuration is undefined"
        )
    spherical, scales, widths = coefficients_to_tensors(node, atoms, layout,
                                                       radial_coeffs=predicted)
    if spatial:
        def flatten(value):
            return jnp.take(value.reshape((1, -1, *value.shape[2:])), layout["active"], axis=1)
        spherical, scales, widths = [list(map(flatten, values))
                                     for values in (spherical, scales, widths)]
    L0_start = kind == "SphericalHarmonicsEnergyNetwork" and attr(node, "L0_start", True)
    features, radial_all = [], 0
    if predicted:
        for L, (scale, width) in enumerate(zip(scales, widths)):
            radial = layers.apply(child(child(node, "coeff_activation"), L), state,
                                  scale * parameter(node, state, f"radial_scale_filters_{L}")
                                  * width * parameter(node, state, f"radial_width_filters_{L}"))
            if kind == "SphericalHarmonicsEnergyNetwork":
                radial = radial / jnp.sqrt(_torch_abs(radial) + 1e-8)
            radial = jnp.sum(radial, axis=-2, keepdims=True)
            if L0_start:
                radial_all = radial_all + layers.apply(child(child(node, "radial_L0_map"), L), state, radial)
                features.append(spherical[L])
            else:
                features.append(spherical[L] * radial)
        if L0_start:
            features[0] = features[0] * radial_all
            features[0] = features[0] / jnp.sqrt(_torch_abs(features[0]) + 1e-8)
    else:
        features = spherical
    features = _apply_list(child(node, "input_layer"), state, features)
    normalize = attr(node, "normalize", 0)
    if L0_start:
        features = layers.apply(child(node, "L0_change"), state, features)
    if kind == "SphericalHarmonicsEmbeddingEnergyNetwork":
        if normalize > 1:
            features = [layers.layer_norm(value) for value in features]
        embedded = layers.apply(child(node, "embedding"), state, atoms["atom_numbers"])
        features = [a + b for a, b in zip(embedded, features)]
    elif spatial and (normalize > 1 or L0_start):
        features = [layers.layer_norm(value) for value in features]
    orders = attr(node, "_order") if kind == "SphericalHarmonicsEnergyNetwork" else attr(node, "order")
    accumulated = [jnp.asarray(0, dtype=features[0].dtype)
                   for _ in range(max(attr(node, "order_max"), attr(node, "orbitals_max_order")) + 1)]
    if kind != "SphericalHarmonicsEnergyNetwork":
        for L, value in enumerate(features):
            accumulated[L] = accumulated[L] + value
    if spatial:
        radial_basis = layers.apply(child(node, "radial_basis_functions"), state, atoms["distances"])[..., None, :]
    for i, module in enumerate(child(node, "module")["children"].values()):
        features = layers.apply(child(child(node, "order_change"), i), state, features)
        if spatial:
            features, update = layers.apply(module, state, features, radial_basis, atoms["sph"],
                                            atoms["idx_i"], atoms["idx_j"], neighbor_mask=1)
        else:
            features = layers.apply(module, state, features)
            update = features
        for L in range(orders[i] + 1):
            scale = layers.residual_scale(accumulated[L], update[L], normalize)
            accumulated[L] = update[L] * scale + accumulated[L] * scale
    accumulated[0] = layers.apply(child(node, "out_activation"), state, accumulated[0])
    # Preserve the linear head's historical use of xs rather than its accumulated fs.
    output_features = features if not spatial else accumulated
    atom_energy = layers.apply(child(node, "energy_output"), state, output_features)[0][..., 0, 0]
    result["energy"] = (_scatter_energy(atoms, atom_energy) if spatial
                        else jnp.sum(atom_energy, axis=1, keepdims=True))
    return result
