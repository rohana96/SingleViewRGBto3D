"""Implements loss function for mesh representation"""

import torch
from pytorch3d.loss import mesh_laplacian_smoothing


class MeshLoss:
    """Implements loss function for mesh representation"""

    def __init__(self, reg_loss="laplacian"):
        """Default constructor

        Args:
            reg_loss: type of regularization loss
        """
        self.reg_loss = reg_loss

    def run(self, *args, **kwargs):
        """Computes loss

        Returns:
            float: loss value
        """
        if self.reg_loss == "laplacian":
            return self.laplacian(*args, **kwargs)

    def laplacian(self, meshes, method="uniform"):
        """Implements laplacian smoothening losses

        Args:
            meshes: input batch of meshes
            method: weighting scheme-- "uniform", "cot", "cotcurv"

        Returns:
            None
        """
        loss = mesh_laplacian_smoothing(meshes, method=method)
        return loss

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


def test():
    pass


if __name__ == "__main__":
    test()