import string
import random
import torch
import numpy as np

_sqrt2 = np.sqrt(2)


# used for creating a "unique" id for a run (almost impossible to generate the same twice)
def generate_id(
    size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits
):
    return "".join(random.SystemRandom().choice(chars) for _ in range(size))


def compute_error_dict(
    predictions,
    data,
    loss_weights,
    max_errors,
    coord_weights=None,
    weights_balance=1,
    minimize_en=False,
):
    error_dict = {}
    error_dict["loss"] = 0.0
    for key in loss_weights.keys():
        if loss_weights[key] > 0:
            diff = predictions[key] - (data[key])
            if key == "density" and coord_weights is not None:
                balanced_weights = torch.sign(coord_weights) * torch.abs(
                    coord_weights
                ) ** (1 / weights_balance)
                balanced_weights *= torch.sum(coord_weights) / torch.sum(
                    balanced_weights
                )
                diff *= balanced_weights
            abs_diff = torch.abs(diff)
            sq_diff = diff ** 2
            mse = torch.mean(sq_diff)
            rmse = torch.sqrt(mse)
            mae = torch.mean(abs_diff)
            if mae > max_errors[key]:
                error_dict[key + "_mae"] = torch.tensor(max_errors[key])
                error_dict[key + "_rmse"] = torch.tensor(_sqrt2 * max_errors[key])
            else:
                error_dict[key + "_mae"] = mae
                error_dict[key + "_rmse"] = rmse
            loss = mae + rmse
            if key == "energy" and minimize_en:
                loss = torch.mean(predictions["energy"])
                error_dict[key + "_mae"] = loss
                error_dict[key + "_rmse"] = loss
            error_dict["loss"] = error_dict["loss"] + loss_weights[key] * loss
    return error_dict


# returns an error dictionary filled with zeros
def empty_error_dict(loss_weights, fill_value=0.0):
    error_dict = {}
    error_dict["loss"] = fill_value
    for key in loss_weights.keys():
        if loss_weights[key] > 0:
            error_dict[key + "_mae"] = fill_value
            error_dict[key + "_rmse"] = fill_value
    return error_dict
