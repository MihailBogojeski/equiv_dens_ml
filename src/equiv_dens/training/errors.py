import torch
import numpy as np
import equiv_dens.scripts.transform_df_coeffs as transform_df_coeffs


_sqrt2 = np.sqrt(2)


class ErrorDict:
    """
    Used to calcuate the loss for the various properties
    """
    def __init__(self, loss_weights, weights_balance=1,
                 percentage_error=True,
                 max_errors=None,
                 weights_decay=None,
                 weights_min=None,
                 loss_comp=None,
                 relative_en=False,
                 df_loss_weights=False,
                 ):
        self.loss_weights = loss_weights
        self.weights_balance = weights_balance
        self.percentage_error = percentage_error
        self.weights_decay = weights_decay
        self.weights_min = weights_min
        self.loss_comp = loss_comp
        self.relative_en = relative_en
        self.df_loss_weights = df_loss_weights
        if self.weights_decay is None:
            self.weights_decay = {}
            for key in self.loss_weights.keys():
                self.weights_decay[key] = 1.0
        if self.weights_min is None:
            self.weights_min = {}
            for key in self.loss_weights.keys():
                self.weights_min[key] = self.loss_weights[key]
        if max_errors is None:
            self.max_errors = {key: np.inf for key in self.loss_weights.keys()}
        else:
            self.max_errors = max_errors
        if self.loss_comp is None:
            self.loss_comp = {key: 'mae+rmse' for key in self.loss_weights.keys()}

    def compute(self, predictions, data, exclude_energy_min=False):
        error_dict = {}
        error_dict["loss"] = torch.tensor(0.0).to(data['positions'])
        if 'coord_weights' in data.keys():
            coord_weights = data['coord_weights']
        else:
            coord_weights = None
        for key in self.loss_weights.keys():
            if self.loss_weights[key] > 0:
                if key == "energy_min":
                    if exclude_energy_min:
                        continue
                    loss = torch.mean(predictions[key])
                    error_dict[key + "_mae"] = loss
                    error_dict[key + "_rmse"] = loss
                else:
                    diff = predictions[key] - (data[key])
                    if key == 'df_coeffs':
                        diff = predictions[key] - transform_df_coeffs.transform(data[key], data['atom_numbers']) 
                        if self.df_loss_weights:
                            diff = diff * predictions['df_weights']

                    if key == "energy" and self.relative_en:
                        en_offset = torch.mean(predictions[key]) - torch.mean(data[key])
                        print('en_offset', en_offset)
                        diff = diff - en_offset
                    # print('error key', key)
                    # print('pred.shape', predictions[key].shape)
                    # print('data.shape', data[key].shape)
                    # print('diff.shape', diff.shape)
                    if key == "density" and coord_weights is not None:
                        balanced_weights = torch.sign(coord_weights)\
                            * torch.abs(coord_weights) ** (1 / self.weights_balance)
                        balanced_weights *= torch.sum(coord_weights) / torch.sum(balanced_weights)
                    else:
                        balanced_weights = 1
                    abs_diff = torch.abs(diff) * balanced_weights
                    sq_diff = (diff ** 2) * balanced_weights
                    #     print('sq diff no weights negative', torch.sum((diff ** 2) < 0))
                    #     print('sq diff negative', torch.sum(sq_diff < 0))
                    mse = torch.mean(sq_diff, dim=1)
                    # use abs to bypass rare cases of negative mse, while still giving some loss
                    mse = torch.mean(torch.abs(mse))
                    rmse = torch.sqrt(mse)
                    mae = torch.mean(abs_diff)
                    # print('RMSE:', rmse)
                    # print('MAE:', mae)
                    if key == "density" and self.percentage_error and coord_weights is not None:
                        rmse = rmse / torch.sqrt(torch.mean((data[key] ** 2) * balanced_weights))
                        mae = mae / torch.mean(data[key] * balanced_weights)
                        # print('pct RMSE:', rmse)
                        # print('pct MAE:', mae)
                    if mae > self.max_errors[key]:
                        error_dict[key + "_mae"] = torch.tensor(self.max_errors[key])
                        error_dict[key + "_rmse"] = torch.tensor(_sqrt2 * self.max_errors[key])
                    else:
                        error_dict[key + "_mae"] = mae
                        error_dict[key + "_rmse"] = rmse
                    if self.loss_comp[key] == 'mae+rmse':
                        loss = mae + rmse
                    elif self.loss_comp[key] == 'mae':
                        loss = mae
                    elif self.loss_comp[key] == 'rmse':
                        loss = rmse
                    else:
                        raise Exception("Unsupported loss composition.")
                error_dict[key + '_loss'] = loss
                error_dict["loss"] = error_dict["loss"] + self.loss_weights[key] * loss
        return error_dict

    # returns an error dictionary filled with zeros
    def empty(self, fill_value=0.0):
        error_dict = {}
        error_dict["loss"] = fill_value
        for key in self.loss_weights.keys():
            if self.loss_weights[key] > 0:
                error_dict[key + "_mae"] = fill_value
                error_dict[key + "_rmse"] = fill_value
                error_dict[key + "_loss"] = fill_value
        return error_dict
