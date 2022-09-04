"""Implements voxel loss"""

from torch.nn import BCEWithLogitsLoss


class VoxelLoss:
    """Implements voxel loss"""

    def __init__(self, method="BCEWithLogits"):
        """Default constructor

        Args:
            method: type of loss
        """
        self.method = method

    def run(self, voxel_src, voxel_tgt, *args, **kwargs):
        """Compute loss

        Args:
            voxel_src(Tensor):
            voxel_tgt(Tensor):
        """
        if self.method == "BCEWithLogits":
            criterion = BCEWithLogitsLoss()
            return criterion(voxel_src, voxel_tgt)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
