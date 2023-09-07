import torch


def density_LDA_loss(pred_dens, target_dens, grid_weights):
    """
    Compute the differences in the LDA exchange-correlation energies between the predicted and target density.

    Args:
        pred_dens (torch.Tensor): Predicted density.
        target_dens (torch.Tensor): Target/ground truth density.
        grid_weights (torch.Tensor): Weights of the integration grid.
    """
    pred_dens = pred_dens.clamp(min=0)
    lda_pred = torch.sum(pred_dens ** (4 / 3) * grid_weights, dim=1)
    lda_target = torch.sum(target_dens ** (4 / 3) * grid_weights, dim=1)

    return lda_pred - lda_target


def _density_coulomb_loss(dens_diff, grid_coords, grid_weights, subsampling=0):
    """
    Compute the dfferences in the coulomb energy integral between the predicted and target density.

    Args:
        dens_diff (torch.Tensor): Pre-computed difference between the densities.
        grid_coords (torch.Tensor): Coordinates of the integration grid.
        grid_weights (torch.Tensor): Weights of the integration grid.
    """
    dens_diff = torch.abs(dens_diff)
    if subsampling == 0:
        idx1 = torch.randperm(dens_diff.shape[1])
        idx2 = torch.roll(idx1, 1)
    elif subsampling > 1:
        idx1 = torch.randint(0, dens_diff.shape[1])
        idx2 = torch.randint(0, dens_diff.shape[1])
        while torch.min(torch.abs(idx1 - idx2)) == 0:
            idx2 = torch.randint(0, dens_diff.shape[1])
    else:
        idx = torch.arange(dens_diff.shape[1])
        idx1 = idx.repeat(dens_diff.shape[1], 1).t().reshape(-1)
        idx2 = idx.repeat(dens_diff.shape[1]).view(-1)
    diff1 = dens_diff[:, idx1]
    diff2 = dens_diff[:, idx2]
    weights1 = grid_weights[:, idx1]
    weights2 = grid_weights[:, idx2]
    coords_dist = torch.norm(grid_coords[:, idx1] - grid_coords[:, idx2], dim=2)

    coulomb = torch.sum(weights1 * weights2 * diff1 * diff2 / coords_dist, dim=1)
    return coulomb


def density_coulomb_loss(pred_dens, target_dens, grid_coords, grid_weights, subsampling=0):
    """
    Compute the dfferences in the coulomb energy integral between the predicted and target density.

    Args:
        dens_diff (torch.Tensor): Pre-computed difference between the densities.
        grid_coords (torch.Tensor): Coordinates of the integration grid.
        grid_weights (torch.Tensor): Weights of the integration grid.
    """
    return _density_coulomb_loss(pred_dens - target_dens, grid_coords, grid_weights, subsampling)


def _density_mixed_distance_loss(dens_abs_diff, dens_sq_diff, grid_coords, atom_pos, atom_numbers, width=0.2):
    atom_pos = atom_pos.detach()
    atom_num = atom_numbers.detach().to(atom_pos)
    atom_pos.requires_grad = False
    offset = torch.zeros_like(atom_numbers).to(atom_pos)
    offset[atom_numbers == 0] = torch.finfo(torch.float32).max
    atom_pos = atom_pos + offset.unsqueeze(-1)
    distance_from_atoms = torch.norm(grid_coords.unsqueeze(1) - atom_pos.unsqueeze(2), dim=-1)
    min_distance_from_atoms = torch.min(distance_from_atoms, dim=1)[0]

    rmse_factor = torch.exp(-(min_distance_from_atoms**2) * width)
    mae_factor = 1 - rmse_factor
    dist_rmse = torch.sqrt(torch.sum(rmse_factor * dens_sq_diff, dim=1))
    dist_mae = torch.sum(mae_factor * dens_abs_diff, dim=1)
    mixed_dist_loss = dist_rmse + dist_mae
    return mixed_dist_loss


def density_mixed_distance_loss(pred_dens, target_dens, grid_weights, grid_coords, atom_pos, atom_numbers, width=0.2):
    dens_diff = pred_dens - target_dens
    dens_abs_diff = torch.abs(dens_diff) * grid_weights
    dens_sq_diff = dens_diff**2 * grid_weights

    return _density_mixed_distance_loss(dens_abs_diff, dens_sq_diff, grid_coords, atom_pos, atom_numbers, width)
