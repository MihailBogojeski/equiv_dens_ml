import string
import random
import torch
import numpy as np

_sqrt2 = np.sqrt(2)

#used for creating a "unique" id for a run (almost impossible to generate the same twice)
def generate_id(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))


def compute_error_dict(predictions, data, loss_weights, max_errors, coord_weights=None):
    error_dict = {}
    error_dict['loss'] = 0.0
    for key in loss_weights.keys():
        if loss_weights[key] > 0:
            diff = predictions[key] - (data[key])
            abs_diff = torch.abs(diff)
            sq_diff = diff**2
            if key == 'density' and coord_weights is not None:
                w_abs_diff = abs_diff * coord_weights
                w_sq_diff = sq_diff * coord_weights
            else:
                w_abs_diff = 0
                w_sq_diff = 0
            mse  = torch.mean(sq_diff)
            rmse = torch.sqrt(mse)
            mae  = torch.mean(abs_diff)
            if key == 'density' and coord_weights is not None:
                w_mse  = torch.mean(w_sq_diff)
                w_rmse = torch.sqrt(w_mse)
                w_mae  = torch.mean(w_abs_diff)
            else:
                w_mse  = 0
                w_rmse = 0
                w_mae  = 0
            if mae > max_errors[key]:
                error_dict[key + '_mae']  = torch.tensor(max_errors[key])
                error_dict[key + '_rmse'] = torch.tensor(_sqrt2 * max_errors[key])
            else:
                error_dict[key + '_mae']  = mae
                error_dict[key + '_rmse'] = rmse
            loss = mae + rmse
            if key == 'density' and coord_weights is not None:
                # print('weighting coords and scaling')
                w_loss = w_mae + w_rmse
                # print('orig loss', loss)
                # print('w_loss', w_loss)
                # print('coord weights mean', torch.mean(coord_weights))
                # print('coord weights median', torch.median(coord_weights))
                loss = w_loss + loss / 1000
                # print('scaled loss + w_loss', loss)
            # loss = mae
            error_dict['loss'] = error_dict['loss'] + loss_weights[key]*loss
    return error_dict

#returns an error dictionary filled with zeros
def empty_error_dict(loss_weights, fill_value=0.0):
    error_dict = {}
    error_dict['loss'] = fill_value
    for key in loss_weights.keys():
        if loss_weights[key] > 0:
            error_dict[key+'_mae']  = fill_value
            error_dict[key+'_rmse'] = fill_value
    return error_dict
