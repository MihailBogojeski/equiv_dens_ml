import torch
import numpy as np
import equiv_dens.scripts.transform_df_coeffs as transform_df_coeffs
from equiv_dens.utils import orbitals
from equiv_dens.training import density_errors


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
                 loss_comp_weights=None,
                 ):
        self.loss_weights = loss_weights
        self.weights_balance = weights_balance
        self.percentage_error = percentage_error
        self.weights_decay = weights_decay
        self.weights_min = weights_min
        self.loss_comp = loss_comp
        self.relative_en = relative_en
        self.df_loss_weights = df_loss_weights
        self.loss_comp_weights = loss_comp_weights
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
            self.loss_comp = {key: ['mae'] for key in self.loss_weights.keys() if self.loss_weights[key] > 0}
        if self.loss_comp_weights is None:
            self.loss_comp_weights = {key: {loss_key: 1.0 for loss_key in self.loss_comp[key]}
                                      for key in self.loss_weights.keys() if self.loss_weights[key] > 0}
        # if 'density' in self.loss_comp.keys() and self.percentage_error:
        #     if 'mae' in self.loss_comp['density']:
        #         self.loss_comp['density'].append('scaled_mae')
        #         self.loss_comp['density'].remove('mae')
        #         self.loss_comp_weights['density']['scaled_mae'] = self.loss_comp_weights['density']['mae']
        #         self.loss_comp_weights['density'].pop('mae')
        #     if 'rmse' in self.loss_comp['density']:
        #         self.loss_comp['density'].append('scaled_rmse')
        #         self.loss_comp['density'].remove('rmse')
        #         self.loss_comp_weights['density']['scaled_rmse'] = self.loss_comp_weights['density']['rmse']
        #         self.loss_comp_weights['density'].pop('rmse')

    def compute(self, predictions, data, exclude_energy_min=False):
        error_dict = {}
        error_dict["loss"] = torch.tensor(0.0).to(data['positions'])
        if 'coord_weights' in data.keys():
            coord_weights = data['coord_weights']
        else:
            coord_weights = None
        for key in self.loss_weights.keys():
            loss = 0
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
                        diff = predictions[key] - transform_df_coeffs.transform(data[key], data['batch_atom_numbers'])
                        if self.df_loss_weights:
                            diff = diff * predictions['df_weights']

                    if key == "energy" and self.relative_en:
                        en_offset = torch.mean(predictions[key]) - torch.mean(data[key])
                        diff = diff - en_offset
                    if key == "density" and coord_weights is not None:
                        balanced_weights = torch.sign(coord_weights)\
                            * torch.abs(coord_weights) ** (1 / self.weights_balance)
                        balanced_weights *= torch.sum(coord_weights) / torch.sum(balanced_weights)
                    else:
                        balanced_weights = 1
                    abs_diff = torch.abs(diff) * balanced_weights
                    sq_diff = (diff ** 2) * balanced_weights
                    if key == 'density':
                        mse = torch.sum(sq_diff, dim=1)
                        rmse = torch.sqrt(mse)
                        rmse = torch.mean(rmse)
                        mae = torch.mean(torch.sum(abs_diff, dim=1))
                        mse = torch.mean(mse)
                    else:
                        mse = torch.mean(sq_diff)
                        rmse = torch.sqrt(mse)
                        mae = torch.mean(abs_diff)
                    losses = {'mae': mae, 'rmse': rmse, 'mse': mse}
                    if key == "density":
                        if 'coulomb' in self.loss_comp[key]:
                            losses['coulomb'] = torch.mean(density_errors._density_coulomb_loss(diff, data['coords'], coord_weights))
                        if 'mixed_dist_err' in self.loss_comp[key] or 'perc_mixed_dist_err' in self.loss_comp[key]:
                            losses['mixed_dist_err'] = torch.mean(density_errors._density_mixed_distance_loss(abs_diff, sq_diff, data['coords'],
                                                                                                              data['batch_positions'],
                                                                                                              data['batch_atom_numbers']))
                            if 'perc_mixed_dist_err' in self.loss_comp[key]:
                                losses['perc_mixed_dist_err'] = losses['mixed_dist_err'] /\
                                    torch.mean(density_errors._density_mixed_distance_loss(data['density'] * balanced_weights, data['density']**2 * balanced_weights, data['coords'],
                                                                                          data['batch_positions'],
                                                                                          data['batch_atom_numbers']))
                        if 'perc_mae' in self.loss_comp[key]:
                            losses['perc_mae'] = mae / torch.mean(torch.sum(data[key] * balanced_weights, dim=1))
                        if 'perc_rmse' in self.loss_comp[key]:
                            losses['perc_rmse'] = rmse / torch.mean(torch.sqrt(torch.sum((data[key] ** 2) * balanced_weights, dim=1)))
                        if 'lda_mae' in self.loss_comp[key] or 'lda_rmse' in self.loss_comp[key]:
                            lda_diff = density_errors.density_LDA_loss(predictions['density'], data['density'], coord_weights)
                            losses['lda_mae'] = torch.mean(torch.abs(lda_diff))
                            losses['lda_rmse'] = torch.sqrt(torch.mean(lda_diff ** 2))
                        if 'hartree_mae' in self.loss_comp[key] or 'hartree_rmse' in self.loss_comp[key]:
                            hartree_diff = density_errors.density_hartree_loss(predictions['density'], data['density'], data['coords'], coord_weights)
                            losses['hartree_mae'] = torch.mean(torch.abs(hartree_diff))
                            losses['hartree_rmse'] = torch.sqrt(torch.mean(hartree_diff ** 2))
                        if 'kl_loss' in self.loss_comp[key]:
                            losses['kl_loss'] = torch.mean(density_errors.density_KL_loss(
                                predictions['density'], data['density'],
                                data['batch_atom_numbers'], coord_weights))
                        if 'dpm_loss' in self.loss_comp[key]:
                            losses['dpm_loss'] = torch.mean(density_errors.dipole_pointwise_int_loss(
                                predictions['density'], data['density'],
                                data['coords'], coord_weights))
                    if mae > self.max_errors[key]:
                        losses['mae'] = torch.clamp(losses['mae'], self.max_errors[key])
                        losses['rmse'] = torch.clamp(losses['rmse'], torch.sqrt(2) * self.max_errors[key])
                        losses['mse'] = torch.clamp(losses['mse'], self.max_errors[key]**2)
                    # print('losses', losses)
                    for loss_key in losses.keys():
                        # print('key', key, 'loss_key', loss_key)
                        error_dict[key + '_' + loss_key] = losses[loss_key]
                        if loss_key in self.loss_comp[key]:
                            # print('key', key, 'loss_key', loss_key)
                            # print('losses keys', losses.keys())
                            # print('loss comp weights', self.loss_comp_weights)
                            loss += self.loss_comp_weights[key][loss_key] * losses[loss_key]
                    error_dict[key + '_loss'] = loss
                error_dict['loss'] = error_dict['loss'] + self.loss_weights[key] * loss
                # print('error dict', error_dict)
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
