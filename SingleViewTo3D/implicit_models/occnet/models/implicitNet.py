"""
Implements the encoder + decoder for the implicit models. The architecture is heavily inspired by Occupancy network.

Encoder:
    1. image encoder: A resnet18 is used to encode the image to a 512-d feature vector
    2. point encoder: A linear layers-based encoder to output 512-d feature vector

Decoder:
    A resnet based decoder of point features with batch normalization conditioned on image latent
"""

import torch
import torch.nn as nn
from torchvision import transforms

from SingleViewTo3D.implicit_models.occnet.models.convEncoder import ConvEncoder, ResnetEncoder
from SingleViewTo3D.implicit_models.occnet.models.pointEncoder import PointEncoder
from SingleViewTo3D.implicit_models.occnet.models.cbnDecoder import CBNDecoder


class ImplicitModel(nn.Module):
    """
    Implements the encoder-decoder architecture for the implicit model. The architecture is heavily inspired by Occupancy network.

    Encoder:
        1. image encoder: A resnet18 is used to encode the image to a 512-d feature vector
        2. point encoder: A linear layers-based encoder to output 512-d feature vector

    Decoder:
        A resnet based decoder of point features with batch normalization conditioned on image latent
    """

    def __init__(self, n_points=5000, encoder_type="resnet18", decoder_type="simple"):
        """
        Default constructor for class implicit_models

        :param encoder_type: type of image encoder -- "resnet18", "conv"
        """
        super().__init__()
        self.n_points = n_points
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.img_encoder = None
        self.decoder = None
        self.decoder_type = decoder_type

        # initialize image encoder
        if encoder_type == "resnet18":
            self.img_encoder = ResnetEncoder()
        elif encoder_type == "conv":
            self.img_encoder = ConvEncoder()

        # initialize point encoder
        self.point_encoder = PointEncoder(in_features=3)

        # initialize decoder
        if decoder_type == "cbn_decoder":
            self.decoder = CBNDecoder(n_points=self.n_points)
        elif decoder_type == "simple":
            self.decoder = nn.Sequential(
                nn.Linear(in_features=512, out_features=256),
                nn.PReLU(),
                nn.Linear(in_features=256, out_features=128),
                nn.PReLU(),
                nn.Linear(in_features=128, out_features=1),
            )

    def forward(self, pts, images):
        """
        Implements a forward pass through the implicit model

        :param pts: (n_batch, n_points, 3) dimensional input 3D points
        :param images: (n_batch, H, W, C) dimensional input images
        :return: (n_batch, n_points, 1) occupancy value
        """
        images = images.permute(0, 3, 1, 2)
        img_encoding = self.img_encoder(images)  # n_batch x 512
        point_encoding = self.point_encoder(pts)  # (n_batch * n_points) x 512

        n_batch = img_encoding.shape[0]
        n_points = point_encoding.shape[0] // n_batch
        img_encoding = img_encoding.repeat_interleave(n_points, dim=0)  # (n_batch * n_points) x 512

        if self.decoder_type == "simple":
            comb_encoding = img_encoding + point_encoding
            return self.decoder(comb_encoding).reshape((n_batch, n_points, 1))  # (n_batch, n_points, 3)

        elif self.decoder_type == "cbn_decoder":
            return self.decoder(point_encoding, img_encoding).reshape(n_batch, n_points, 1)
        return None


def test():
    points = torch.rand(size=(64, 100, 3))
    images = torch.randn(size=(64, 137, 137, 3))

    implicit_model = ImplicitModel(encoder_type="resnet18", decoder_type="simple")
    print(implicit_model(points, images).shape)


if __name__ == "__main__":
    test()
