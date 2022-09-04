import sys
import pytorch3d
from pytorch3d.structures import Pointclouds

sys.path.append("..")
import imageio
import LearningPytorch3D.src.utils as utils


class VisualizePointCloud:
    def __init__(self):
        pass

    def visualize(self, point_cloud, id, save_dir='', *args, **kwargs):
        """
        Renders a set 2D projections of a point cloud

        :param save_dir:
        :param id: (int) output id
        :param point_cloud: torch tensor of dimension (1, n_points, 3)
        """
        points = point_cloud.squeeze(0)
        color = (points - points.min()) / (points.max() - points.min())
        point_cloud = pytorch3d.structures.Pointclouds(
            points=[points], features=[color],
        )
        frames = utils.get_point_cloud_images(point_cloud, *args, **kwargs)
        filepath = save_dir + f'point_cloud_{id}.gif'
        imageio.mimsave(filepath, frames, fps=5)

    def __call__(self, *args, **kwargs):
        self.visualize(*args, **kwargs)
