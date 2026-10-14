"""Native counterparts of the current ErrorDict and density loss formulas."""
from __future__ import annotations

import math
import jax
import jax.numpy as jnp


def _torch_abs(value):
    """Match Torch's zero subgradient, including exact-match MAE losses."""
    return jnp.where(value == 0,jnp.zeros_like(value),jnp.abs(value))


def _torch_vector_norm(value,axis=-1):
    """Match Torch vector_norm's zero-vector first derivative."""
    squared = jnp.sum(value*value,axis=axis)
    zero = squared == 0
    return jnp.where(zero,0,jnp.sqrt(jnp.where(zero,1,squared)))


def _pairs(count, key, pair_indices=None):
    if pair_indices is not None:
        return pair_indices
    if count < 2:
        raise ValueError("Coulomb/Hartree sampled losses require at least two grid points")
    if key is None:
        raise ValueError("Supply an explicit PRNG key or shared pair_indices for sampled losses")
    first = jax.random.permutation(key, count)
    return first, jnp.roll(first, 1)


def pair_integral(values, coords, weights, key=None, pair_indices=None):
    i, j = _pairs(values.shape[1], key, pair_indices)
    distance = jnp.linalg.norm(coords[:, i] - coords[:, j], axis=-1)
    return jnp.sum(weights[:, i] * weights[:, j] * values[:, i] * values[:, j] / distance, axis=1)


def density_LDA_loss(prediction, target, weights):
    return jnp.sum(jnp.maximum(prediction, 0)**(4/3)*weights, axis=1) - jnp.sum(target**(4/3)*weights, axis=1)


def density_KL_loss(prediction, target, weights):
    target = jnp.maximum(target, 1e-8)
    prediction = jnp.maximum(prediction, 1e-8)
    target = target / jnp.sum(target*weights, axis=1, keepdims=True)
    prediction = prediction / jnp.sum(prediction*weights, axis=1, keepdims=True)
    return jnp.sum(target*(jnp.log(target)-jnp.log(prediction))*weights, axis=1)


def mixed_distance_loss(abs_diff, sq_diff, coords, positions, numbers, width=0.2):
    positions = jax.lax.stop_gradient(positions)
    offset = jnp.where(numbers == 0, jnp.finfo(jnp.float32).max, 0)
    positions = positions + offset[..., None]
    distance = jnp.min(jnp.linalg.norm(coords[:, None] - positions[:, :, None], axis=-1), axis=1)
    factor = jnp.exp(-distance**2*width)
    return jnp.sqrt(jnp.sum(factor*sq_diff, axis=1)) + jnp.sum((1-factor)*abs_diff, axis=1)


class ErrorDict:
    """Preserves the source loss reductions and configured component weights.

    Stochastic pair losses take an explicit key; shared pair indices allow exact
    Torch/JAX equation comparisons independently of differing PRNG algorithms.
    """

    def __init__(self, loss_weights, weights_balance=1, percentage_error=True,
                 max_errors=None, weights_decay=None, weights_min=None,
                 loss_comp=None, relative_en=False, df_loss_weights=False,
                 loss_comp_weights=None):
        if weights_balance == 0:
            raise ValueError("weights_balance must be nonzero")
        self.loss_weights = dict(loss_weights)
        self.weights_balance = weights_balance
        self.percentage_error = percentage_error
        self.max_errors = max_errors or {key: math.inf for key in loss_weights}
        self.weights_decay = weights_decay or {key: 1.0 for key in loss_weights}
        self.weights_min = weights_min or dict(loss_weights)
        self.loss_comp = loss_comp or {key: ["mae"] for key, weight in loss_weights.items() if weight > 0}
        self.relative_en = relative_en
        self.df_loss_weights = df_loss_weights
        self.loss_comp_weights = loss_comp_weights or {
            key: {component: 1.0 for component in self.loss_comp[key]}
            for key, weight in loss_weights.items() if weight > 0}

    def compute(self, predictions, data, exclude_energy_min=False, *, key=None, pair_indices=None,
                loss_weights_override=None):
        result = {"loss": jnp.asarray(0., dtype=data["positions"].dtype)}
        weights = data.get("coord_weights")
        for field, field_weight in self.loss_weights.items():
            if field_weight <= 0:
                continue
            if field == "energy_min":
                if exclude_energy_min:
                    continue
                loss = jnp.mean(predictions[field])
                result.update({field+"_mae": loss, field+"_rmse": loss})
            else:
                components = self.loss_comp[field]
                diff = predictions[field] - data[field]
                if field == "df_coeffs" and self.df_loss_weights:
                    diff = diff * predictions["df_weights"]
                if field == "energy" and self.relative_en:
                    diff = diff - (jnp.mean(predictions[field])-jnp.mean(data[field]))
                balanced = 1
                if "density" in field and weights is not None:
                    balanced = jnp.sign(weights)*jnp.abs(weights)**(1/self.weights_balance)
                    balanced = balanced*jnp.sum(weights)/jnp.sum(balanced)
                padded = balanced
                if hasattr(balanced, "ndim") and diff.ndim > balanced.ndim:
                    padded = balanced.reshape(balanced.shape+(1,)*(diff.ndim-balanced.ndim))
                abs_diff, sq_diff = _torch_abs(diff)*padded, diff**2*padded
                if field == "density":
                    mse_per_batch = jnp.sum(sq_diff, axis=1)
                    mae = jnp.mean(jnp.sum(abs_diff, axis=1))
                    mse, rmse = jnp.mean(mse_per_batch), jnp.mean(jnp.sqrt(mse_per_batch))
                else:
                    mse, mae = jnp.mean(sq_diff), jnp.mean(abs_diff)
                    rmse = jnp.sqrt(mse)
                losses = {"mae": mae, "mse": mse, "rmse": rmse}
                if field == "density":
                    if "coulomb" in components:
                        losses["coulomb"] = jnp.mean(pair_integral(_torch_abs(diff),data["coords"],weights,key,pair_indices))
                    if "mixed_dist_err" in components or "perc_mixed_dist_err" in components:
                        mixed = jnp.mean(mixed_distance_loss(abs_diff,sq_diff,data["coords"],data["batch_positions"],data["batch_atom_numbers"]))
                        losses["mixed_dist_err"] = mixed
                        if "perc_mixed_dist_err" in components:
                            denominator = jnp.mean(mixed_distance_loss(data[field]*balanced,data[field]**2*balanced,data["coords"],data["batch_positions"],data["batch_atom_numbers"]))
                            losses["perc_mixed_dist_err"] = mixed/denominator
                    if "perc_mae" in components:
                        losses["perc_mae"] = mae/jnp.mean(jnp.sum(data[field]*balanced,axis=1))
                    if "perc_rmse" in components:
                        losses["perc_rmse"] = rmse/jnp.mean(jnp.sqrt(jnp.sum(data[field]**2*balanced,axis=1)))
                    for kind in ("lda", "hartree"):
                        if any(kind+"_"+metric in components for metric in ("mae", "rmse")):
                            difference = density_LDA_loss(predictions[field],data[field],weights) if kind=="lda" else (
                                pair_integral(predictions[field],data["coords"],weights,key,pair_indices)-pair_integral(data[field],data["coords"],weights,key,pair_indices))
                            losses[kind+"_mae"] = jnp.mean(_torch_abs(difference))
                            losses[kind+"_rmse"] = jnp.sqrt(jnp.mean(difference**2))
                    if "kl_loss" in components:
                        losses["kl_loss"] = jnp.mean(density_KL_loss(predictions[field],data[field],weights))
                    centered = data["coords"]-jnp.mean(data["coords"],axis=1,keepdims=True)
                    if "dpm_loss" in components:
                        losses["dpm_loss"] = jnp.mean(_torch_vector_norm(jnp.sum(-diff[...,None]*weights[...,None]*centered,axis=1)))
                    if "dpm_abs_loss" in components:
                        losses["dpm_abs_loss"] = jnp.mean(jnp.sum(_torch_abs(diff)*weights*jnp.linalg.norm(centered,axis=-1),axis=1))
                if field == "density_grad":
                    if any(name in components for name in ("norm_int","perc_norm_int","mean_perc_norm_int")):
                        # Source intentionally sums over the batch as well.
                        norm_int = jnp.sum(_torch_vector_norm(diff)*weights)
                        losses["norm_int"] = norm_int
                        if "perc_norm_int" in components:
                            losses["perc_norm_int"] = norm_int/jnp.mean(jnp.sum(data["density"]*balanced,axis=1))
                        if "mean_perc_norm_int" in components:
                            # Opt-in batch-independent pilot objective; retain
                            # the legacy batch-summed component above verbatim.
                            losses["mean_perc_norm_int"] = norm_int/diff.shape[0]/jnp.mean(
                                jnp.sum(data["density"]*balanced,axis=1))
                    if "kinetic_vw" in components:
                        vw_pred = jnp.nansum(_torch_vector_norm(predictions[field])**2/predictions["density"]*weights)/8
                        vw_target = jnp.nansum(_torch_vector_norm(data[field])**2/data["density"]*weights)/8
                        losses["kinetic_vw"] = _torch_abs(vw_pred-vw_target)
                maximum = self.max_errors[field]
                if math.isfinite(maximum):
                    # Preserve source clamp(min=...), even though the name may
                    # suggest an upper bound. Do not silently repair its policy.
                    for metric, bound in (("mae",maximum),("rmse",math.sqrt(2)*maximum),("mse",maximum**2)):
                        losses[metric] = jnp.where(mae > maximum,jnp.maximum(losses[metric],bound),losses[metric])
                unknown = set(components)-set(losses)
                if unknown:
                    raise ValueError(f"Unsupported {field} loss components: {sorted(unknown)}")
                result.update({field+"_"+metric:value for metric,value in losses.items()})
                loss = sum(self.loss_comp_weights[field][metric]*losses[metric] for metric in components)
            result[field+"_loss"] = loss
            effective_weight=field_weight if loss_weights_override is None else loss_weights_override[field]
            result["loss"] = result["loss"]+effective_weight*loss
        return result

    def configuration(self):
        return {name:getattr(self,name) for name in ("loss_weights","weights_balance","percentage_error",
                "max_errors","weights_decay","weights_min","loss_comp","relative_en","df_loss_weights","loss_comp_weights")}
