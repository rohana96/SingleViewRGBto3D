"""Implements decoder class for 3D point cloud prediction"""
import torch
import torch.nn as nn


class DecoderPointCloud(nn.Module):
    """Implements decoder class for 3D point cloud prediction"""

    def __init__(self, features_d=512, n_points=1000):
        """Default constructor for point cloud decoder

        Args:
            features_d(int): input dimension

        Returns:
            None
        """

        super(DecoderPointCloud, self).__init__()
        self.n_points = n_points
        self.decoder = nn.Sequential(
            nn.Linear(in_features=features_d, out_features=1024),
            nn.PReLU(),
            nn.Linear(in_features=1024, out_features=2048),
            nn.PReLU(),
            nn.Linear(in_features=2048, out_features=3*self.n_points),
            # nn.Tanh()
        )

    def forward(self, features):
        """Run forward pass through the decoder

        Args:
            features(Tensor): input tensor of features of shape (512, )

        Returns:
            Tensor: point cloud
        """
        return self.decoder(features).reshape((-1, self.n_points, 3))


def test():
    input = torch.randn(20, 512)
    model = DecoderPointCloud()
    print(model(input).shape)


if __name__ == "__main__":
    test()