"""
Implements point encoder for the implicit model
(n_batch, n_points, 3) -> (n_batch * n_points, 512)
"""

import torch
import torch.nn as nn


class PointEncoder(nn.Module):
    """
    Implements point encoder for the implicit models
    (n_batch, n_points, 3) -> (n_batch * n_points, 512)
    """

    def __init__(self, in_features=3, features_d=64):
        """Default constructor for point encoder

        :param in_features: input feature dimensions
        :param features_d: hidden dimensions


        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=features_d),
            nn.ReLU(),
            nn.Linear(in_features=features_d, out_features=features_d * 2),
            nn.ReLU(),
            nn.Linear(in_features=features_d * 2, out_features=features_d * 4),
            nn.ReLU(),
            nn.Linear(in_features=features_d * 4, out_features=features_d * 8),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Implements the forward pass through the PointEncoder

        :param x: (n_batch, n_points, 3) 3D point locations
        :return: (n_batch*n_points, hidden_dim) point encoding
        """

        x = x.reshape((-1, 3))
        return self.encoder(x)


def test():
    n_points = 100
    n_batch = 64
    hidden_dim = 64
    input = torch.rand(size=(n_batch, n_points, 3)) * 100
    encoder = PointEncoder(3, hidden_dim)
    print(encoder(input).shape)


if __name__ == "__main__":
    test()
