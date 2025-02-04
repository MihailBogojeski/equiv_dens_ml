import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftedSoftplus(nn.Module):
    """
    Learnable shifted softplus activation
    """

    def __init__(self, num_features, initial_alpha=1.0, initial_beta=1.0):
        super(ShiftedSoftplus, self).__init__()
        self._log2 = math.log(2)
        self.num_features = num_features
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.register_parameter("alpha", nn.Parameter(torch.Tensor(self.num_features)))
        self.register_parameter("beta", nn.Parameter(torch.Tensor(self.num_features)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.alpha, self.initial_alpha)
        nn.init.constant_(self.beta, self.initial_beta)

    def forward(self, x):
        return self.alpha * torch.where(
            self.beta != 0,
            (F.softplus(self.beta * x) - self._log2) / self.beta,
            0.5 * x,
        )


class Swish(nn.Module):
    """
    Learnable swish activation
    """

    def __init__(self, num_features, initial_alpha=1.0, initial_beta=1.702):
        super(Swish, self).__init__()
        self.num_features = num_features
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.register_parameter("alpha", nn.Parameter(torch.Tensor(self.num_features)))
        self.register_parameter("beta", nn.Parameter(torch.Tensor(self.num_features)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.alpha, self.initial_alpha)
        nn.init.constant_(self.beta, self.initial_beta)

    def forward(self, x):
        return self.alpha * x * torch.sigmoid(self.beta * x)


class NormGate(nn.Module):
    """
    QHNet's NormGate (https://arxiv.org/abs/2306.04922)
    """

    def __init__(self,
                 num_features,
                 order,
                 mlp_activation="swish",
                 mlp_hidden_size=128):
        super(NormGate, self).__init__()

        self.num_features = num_features
        self.order = order
        self.mlp_hidden_size = mlp_hidden_size if mlp_hidden_size > 0 else (self.order + 1) * self.num_features
        
        if mlp_activation == "swish":
            self.mlp_activation = Swish(self.mlp_hidden_size)
        elif mlp_activation == "ssp":
            self.mlp_activation = ShiftedSoftplus(self.mlp_hidden_size)
        else:
            raise ValueError(f"Unsupported activation function: {mlp_activation}")
        # print(f"normgate max order: {self.max_order}")
        self.mlp = nn.Sequential(
            nn.Linear((self.order + 1) * self.num_features, self.mlp_hidden_size),
            self.mlp_activation,
            nn.Linear(self.mlp_hidden_size, (self.order + 1) * self.num_features))
        
    def forward(self, x):

        # dimensions for reshaping mlp output
        bs, n_atoms, _, num_features = x[0].shape
        # n_orders = len(x)

        norms = [torch.norm(x[l], dim=-2, keepdim=True) for l in range(self.order + 1)]
        norms = torch.cat(norms, dim=-2)
        # print("norms", norms.shape)

        norms = norms.view(bs, n_atoms, -1)
        # print("norms (flattened)", norms.shape)

        # print("normgate")
        # print("norms", norms.shape)
        mlp_norms = self.mlp(norms)
        # print("mlp_norms", mlp_norms.shape)

        mlp_norms = mlp_norms.view(bs, n_atoms, self.order + 1, num_features)
        # print("mlp_norms (reshaped)", mlp_norms.shape)

        # print(f"x[0]: {x[0].shape}")
        # print(f"x[1]: {x[1].shape}")
        # print(f"x[2]: {x[2].shape}")
        # print(f"mlp_norms: {mlp_norms.shape}")

        # l=0 mlp output is not multiplied with l=0 input
        x_norm = [mlp_norms[: , :, [0]]]
        for L in range(1, self.order + 1):
            x_norm.append(x[L] * mlp_norms[:, :, [L]])
            
        if len(x_norm) < len(x):
            for L in range(len(x_norm), len(x)):
                x_norm.append(torch.zeros_like(x[L]))

        # x_norm = [mlp_norms[:, :, [0]]] + [x[i] * mlp_norms[:, :, [i]] for i in range(1, self.order + 1)]

        # print("x_norm", len(x_norm), x_norm[0].shape, x_norm[-1].shape)

        return x_norm