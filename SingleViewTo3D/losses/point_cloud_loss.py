"""Implements point cloud loss"""

import torch
from pytorch3d.ops import knn_points


class PointCloudLoss:
    """Implements point cloud loss"""

    def __init__(self, method="chamfer"):
        """Default constructor"""
        self.method = method

    def run(self, *args, **kwargs):
        if self.method == "chamfer":
            return self.chamfer_loss(*args, **kwargs)

    def chamfer_loss(self, point_cloud_src, point_cloud_tgt, K=1):
        """Computes chamfer loss between two point clouds

        Args:
            point_cloud_src(Tensor): (n, p, 3) dimensional predicted cloud
            point_cloud_tgt(Tensor): (n, p, 3) dimensional target cloud
            K(int): number of nearest neighbors to return

        Returns:
            Tensor[float]: chamfer loss between point clouds
        """
        knn_pred = knn_points(point_cloud_src, point_cloud_tgt, K=K)
        pred_to_gt_dists = knn_pred.dists[..., 0]  # (n, p)

        knn_gt = knn_points(point_cloud_tgt, point_cloud_src, K=K)
        gt_to_pred_dists = knn_gt.dists[..., 0]  # (n, p)

        loss_chamfer = torch.mean(pred_to_gt_dists) + torch.mean(gt_to_pred_dists)
        return loss_chamfer

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


def test():
    a = torch.tensor([[[1.0, 2, 0], [2, 3, 0], [1, 1, 0]]])
    b = torch.tensor([[[1.0, 1, 0], [2, 3, 0], [1, 1, 0]]])
    criterion = PointCloudLoss()
    # print(criterion(a, b))


if __name__ == "__main__":
    test()
