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
                 max_order,
                 mlp_activation="swish",
                 mlp_hidden_size=128):
        super(NormGate, self).__init__()

        self.num_features = num_features
        self.max_order = max_order
        
        if mlp_activation == "swish":
            self.mlp_activation = Swish(self.num_features)
        elif mlp_activation == "ssp":
            self.mlp_activation = ShiftedSoftplus(self.num_features)
        else:
            raise ValueError(f"Unsupported activation function: {mlp_activation}")
        
        self.mlp_hidden_size = mlp_hidden_size

        self.mlp = nn.Sequential(
            nn.Linear((self.max_order + 1) * self.num_features, self.mlp_hidden_size),
            self.mlp_activation,
            nn.Linear(self.mlp_hidden_size, (self.max_order + 1) * self.num_features))
        
    def forward(self, x):

        # dimensions for reshaping mlp output
        bs, n_atoms, _, num_features = x[0].shape
        degrees = len(x)

        norms = [torch.norm(x[i], dim=-2, keepdim=True) for i in range(1, len(x))]
        norms = [x[0]] + norms
        norms = torch.cat(norms, dim=-2)
        # print("norms", norms.shape)

        norms = norms.view(bs, n_atoms, -1)
        # print("norms (flattened)", norms.shape)

        mlp_norms = self.mlp(norms)
        # print("mlp_norms", mlp_norms.shape)

        mlp_norms = mlp_norms.view(bs, n_atoms, degrees, num_features)
        # print("mlp_norms (reshaped)", mlp_norms.shape)

        # print(f"x[0]: {x[0].shape}")
        # print(f"x[1]: {x[1].shape}")
        # print(f"x[2]: {x[2].shape}")
        # print(f"mlp_norms: {mlp_norms.shape}")

        # l=0 mlp output is not multiplied with l=0 input
        x_norm = [mlp_norms[:, :, [0]]] + [x[i] * mlp_norms[:, :, [i]] for i in range(1, len(x))]

        # print("x_norm", len(x_norm), x_norm[0].shape, x_norm[-1].shape)

        return x_norm