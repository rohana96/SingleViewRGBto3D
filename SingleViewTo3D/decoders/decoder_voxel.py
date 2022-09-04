"""Implements decoder class for voxel grid prediction"""
import torch
import torch.nn as nn


class ConvDecoderVoxel(nn.Module):
    """Implements 3d conv-based decoder class for voxel grid prediction"""

    def __init__(self, in_channels=1, features_d=4):
        """Default constructor for voxel grid prediction

        Args:
            features_d(int): input dimension

        Returns:
            None
        """

        super(ConvDecoderVoxel, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=4 * features_d, kernel_size=3, stride=1, padding=1),
            self._block(in_channels=4 * features_d, out_channels=2 * features_d),
            self._block(in_channels=2 * features_d, out_channels=features_d),
            nn.ConvTranspose3d(in_channels=features_d, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def _block(self, in_channels, out_channels, kernel_size=2, stride=2):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2),
            nn.PReLU(),
        )

    def forward(self, features):
        """Run forward pass through the decoder

        Args:
            features(Tensor): input tensor of features of shape (512, )

        Returns:
            Tensor: 3D voxel grid
        """
        features = torch.reshape(features, (-1, 1, 8, 8, 8))
        return self.decoder(features)


class LinearDecoderVoxel(nn.Module):
    """Implements linear layers based ecoder class for voxel grid prediction"""

    def __init__(self, in_features=512, out_size=32):
        """Default constructor for voxel grid prediction"""

        super(LinearDecoderVoxel, self).__init__()
        self.out_size = out_size
        self.decoder = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.PReLU(),
                nn.Linear(1024, 2048),
                nn.PReLU(),
                nn.Linear(2048, out_size ** 3)
        )

    def forward(self, features):
        """Run forward pass through the decoder

        Args:
            features(Tensor): input tensor of features of shape (512, )

        Returns:
            Tensor: 3D voxel grid
        """
        out = self.decoder(features)
        out = torch.reshape(out, (-1, 1, self.out_size, self.out_size, self.out_size))
        return out


def test():
    input = torch.randn(20, 512)
    # input = input.reshape((-1, 1, 8, 8, 8))
    model = LinearDecoderVoxel()
    print(model(input).shape)


if __name__ == "__main__":
    test()
