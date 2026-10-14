import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from equiv_dens.utils.electron_configurations import electron_configurations


class Embedding(nn.Module):
    """
    Embedding layer which takes scalar nuclear charges Z and transforms
    them to vectors of size num_features
    """

    def __init__(self, num_features, Zmax=87):
        super(Embedding, self).__init__()
        self.num_features = num_features
        self.Zmax = Zmax
        self.register_buffer("electron_config", torch.tensor(electron_configurations))
        self.register_parameter(
            "element_embedding",
            nn.Parameter(torch.Tensor(self.Zmax, self.num_features)),
        )
        self.config_linear = nn.Linear(
            self.electron_config.size(1), self.num_features, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.element_embedding, -np.sqrt(3), np.sqrt(3))
        nn.init.orthogonal_(self.config_linear.weight)

    def forward(self, Z):
        embedding = self.element_embedding + self.config_linear(self.electron_config)
        return embedding[Z]


class SphericalEmbedding(nn.Module):
    """
    Wraps the ordinary embedding layer and returns features for all orders
    """

    def __init__(self, order, num_features, Zmax=87):
        super(SphericalEmbedding, self).__init__()
        self.order = order
        self.num_features = num_features
        self.Zmax = Zmax
        self.embedding = Embedding(self.num_features, self.Zmax)

    def forward(self, Z):
        """Packed features of shape [*Z.shape, (order + 1)**2, num_features] with only L=0 nonzero."""
        x = self.embedding(Z).unsqueeze(-2)
        return F.pad(x, (0, 0, 0, (self.order + 1) ** 2 - 1))
