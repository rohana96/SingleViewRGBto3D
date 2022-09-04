"""
Implements conditional Batch Norm based decoder in Occupancy networks
"""
import torch.nn as nn
import torch


class CBNDecoder(nn.Module):
    """
    Implements conditional Batch Norm based decoder in Occupancy networks
    """

    def __init__(self, in_features=512, hidden_dim=512, img_encoding_dim=512, n_points=100):
        """
        Default constructor for class CBNDecoder

        :param in_features: number of input dimension
        :param hidden_dim: number of hidden dimension
        :param img_encoding_dim: image encoding dimension
        """
        super().__init__()
        self.resnetCBNBlock = ResnetCBNBlock(in_features, hidden_dim, img_encoding_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, point_encoding, img_encoding, num_resnet_blocks=3):
        """
        Implements the forward pass through the decoder
        :param num_resnet_blocks: number of resnet blocks in the network
        :param img_encoding:  (n_batch, 512) dimensional image feature vector
        :param point_encoding: (n_batch * n_points, 512) dimensional feature encoding of query 3D point(x, y, z)
        :return: (n_batch * n_points, 1) occupancy value
        """

        n_batch = img_encoding.shape[0]
        n_points = point_encoding.shape[0]//n_batch
        img_encoding = img_encoding.repeat_interleave(n_points, dim=0)
        out = point_encoding
        for i in range(num_resnet_blocks):
            out = self.resnetCBNBlock(out, img_encoding)
        return self.linear(out)


class ResnetCBNBlock(nn.Module):
    """Implements the resnet style block with conditional batch normalization"""

    def __init__(self, in_features=512, out_features=512, img_encoding_dim=512):
        """
        Default constructor for class ResnetCBNBlock

        :param img_encoding_dim: (n_batch * n_points, 512) dimensional image encoding
        :param in_features: number of input features
        :param out_features: number of output features
        :return: (n_batch * n_points, out_features) dimensional feature vector
        """
        super().__init__()
        self.skip = nn.Identity()
        if in_features != out_features:
            self.skip = nn.Conv1d(in_features, out_features, kernel_size=1, bias=False)
        self.cond_batch_norm = CondBatchNorm(out_features, img_encoding_dim)
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=in_features, out_features=out_features)
        )

    def forward(self, point_encoding, img_encoding):
        """
        Implements a forward through conditional batch normalization

        :param point_encoding: (n_batch * n_points, 512) point feature
        :param img_encoding: (n_batch * n_points, 512) image feature
        :return: (n_batch * n_points, 512) output feature
        """

        residual = self.linear(self.cond_batch_norm(point_encoding, img_encoding))
        residual = self.linear(self.cond_batch_norm(residual, img_encoding))
        return self.skip(point_encoding.unsqueeze(-1)).squeeze() + residual


class CondBatchNorm(nn.Module):
    """
    Implements conditional batch norm on point features
    """

    def __init__(self, img_encoding_dim=512, out_features=512):
        """
        Default constructor for class CondBatchNorm

        :param img_encoding_dim: (n_batch * n_points, 512) dimensional image encoding
        :param out_features: (n_batch * n_points, 512) dimensional point encoding
        :return:
        """
        super().__init__()
        self.conv_gamma = nn.Conv1d(img_encoding_dim, out_features, kernel_size=1)
        self.conv_beta = nn.Conv1d(img_encoding_dim, out_features, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_features, affine=False)

    def forward(self, point_encoding, img_encoding):
        """
        Implements a forward through conditional batch normalization block

        :param point_encoding: (n_batch * n_points, 512) point feature
        :param img_encoding: (n_batch * n_points, 512) image feature
        :return: (n_batch * n_points, 512) output feature
        """

        img_encoding = img_encoding.unsqueeze(-1)
        gamma = self.conv_gamma(img_encoding).squeeze(-1)
        beta = self.conv_beta(img_encoding).squeeze(-1)
        net = self.bn(point_encoding)
        return gamma * net + beta


def test():
    # test CondBatchNorm
    point_encoding = torch.rand(size=(10, 512))
    image_encoding = torch.rand(size=(10, 512))
    cbn_block = CondBatchNorm(512, 512)
    print(cbn_block(point_encoding, image_encoding).shape)

    # test ResnetCBNBlock
    point_encoding = torch.rand(size=(10, 512))
    image_encoding = torch.rand(size=(10, 512))
    resnet_cbn_block = ResnetCBNBlock(512, 512, 512)
    print(resnet_cbn_block(point_encoding, image_encoding).shape)

    # test CBNDecoder
    point_encoding = torch.rand(size=(100, 512))
    image_encoding = torch.rand(size=(10, 512))
    cbn_decoder = CBNDecoder(512, 512, 512)
    print(cbn_decoder(point_encoding, image_encoding).shape)


if __name__ == "__main__":
    test()
