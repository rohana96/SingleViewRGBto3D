"""
Implements the conv encoder for the implicit model
(n_batch, C, H, W) -> (n_batch, 512)
"""

import torch
import torch.nn as nn
from torchvision import models as torchvision_models
import torchvision.transforms as T


class ConvEncoder(nn.Module):
    """
    Implements the encoder for the implicit models
    (n_batch, C, H, W) -> (n_batch, 512)
    """

    def __init__(self, in_channels=3, hidden_dim=64):
        """
        Default constructor for class ConvEncoder

        :param in_channels: number of input image channels
        :param hidden_dim: number of hidden channels after first conv layer
        :return: (n_batch, out_dim) feature vector
        """
        super().__init__()
        self.encoder = nn.Sequential(
            self._convBlock(in_channels, hidden_dim, 4, 2, 1),
            self._convBlock(hidden_dim, hidden_dim * 2, 4, 2, 1),
            self._convBlock(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            self._convBlock(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _convBlock(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        """
        Implements basic block for conv encoder

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param padding: padding
        :return:
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        """
        Implements forward pass through the encoder
        :param x: input image
        :return: (n_batch, out_dim) feature encoding
        """
        images_normalize = self.normalize(x)
        out = self.encoder(images_normalize)
        out = out.reshape((out.shape[0], -1))
        return out


class ResnetEncoder(nn.Module):
    """
    Implements the resnet18 encoder for the implicit models
    (n_batch, C, H, W) -> (n_batch, 512)
    """

    def __init__(self):
        """
        Default constructor for class ResnetEncoder

        :return: (n_batch, out_dim) feature vector
        """
        super().__init__()
        vision_model = torchvision_models.resnet18(pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        """
        Implements forward pass through the encoder

        :param x: input image
        :return: (n_batch, out_dim) feature encoding
        """
        images_normalize = self.normalize(x)
        out = self.encoder(images_normalize)
        out = out.reshape((out.shape[0], -1))
        return out


def test():
    x = torch.randn(size=(64, 3, 100, 100))
    encoder = ConvEncoder()  # ResnetEncoder()
    print(encoder(x).shape)


if __name__ == "__main__":
    test()
