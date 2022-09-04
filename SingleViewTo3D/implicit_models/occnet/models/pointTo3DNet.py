"""
Implements a simple decoder for point to 3D representation
"""

import torch
import torch.nn as nn


class OccNet(nn.Module):
    """
    Implements occupancy network decoder
    """

    def __init__(self, in_size=3, features_d=64):
        """
        Default constructor to initialize the network

        :param in_size(int): size of the input coordinates -- 3 by default for (x, y, z)
        :param features_d(int): hidden layer dimension
        """
        super(OccNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_size, out_features=features_d),
            nn.ReLU(),
            nn.Linear(in_features=features_d, out_features=features_d),
            nn.ReLU(),
            nn.Linear(in_features=features_d, out_features=features_d),
            nn.ReLU(),
            nn.Linear(in_features=features_d, out_features=1),
            # no sigmoid layer as we use BCEWithLogitsLoss
        )

    def forward(self, x):
        """
        Implements a forward pass through the occupancy network

        :param x: (torch.tensor) (N, 3) tensor of position coordinates of N points
        :return: (N, 1) occupancy values
        """
        return self.net(x).squeeze()


def initialize_weights(model):
    """
    Implements normal weight initialization for the network

    :param model: model
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0.0, std=1.0)


def test():
    input = torch.rand(size=(10, 3)) * 100
    occnet = OccNet(3, 64)
    print(occnet(input).shape)


if __name__ == "__main__":
    test()
