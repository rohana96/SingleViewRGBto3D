import sys

import pytorch3d
import torch

sys.path.append("..")
import imageio
import LearningPytorch3D.src.utils as utils


class VisualizeMesh:

    def __init__(self):
        pass

    def visualize(self, mesh, id, save_dir='', *args, **kwargs):
        """
        Renders a set 2D projections of a mesh

        :param mesh:
        :param id:
        :param save_dir:
        """

        frames = utils.get_mesh_images(mesh)
        filepath = save_dir + f'mesh_{id}.gif'
        imageio.mimsave(filepath, frames, fps=5)

    def __call__(self, *args, **kwargs):
        self.visualize(*args, **kwargs)
