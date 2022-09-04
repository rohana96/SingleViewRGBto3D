"""Implements common encoder class"""

from torchvision import models as torchvision_models
from torchvision.transforms import transforms
import torch.nn as nn
import torch


class Encoder(nn.Module):
    """Implements common encoder class"""

    def __init__(self, args):
        """Default constructor for encoder class"""
        super(Encoder, self).__init__()
        vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, images):
        """Run forward pass through the decoder

        Args:
            images(np.nddarray): (n, h, w, c) tensor

        Returns:
            Tensor: encoded features
        """

        images_normalize = self.normalize(images.permute(0, 3, 1, 2))
        encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1)
        return encoded_feat


def test():
    # x = torch.randn(size=(64, 100, 100, 3))
    # encoder = Encoder()
    # print(encoder(x).shape)
    pass


if __name__ == "__main__":
    test()
