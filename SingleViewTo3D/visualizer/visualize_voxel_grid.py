import sys

sys.path.append("..")
import imageio
import torch
import LearningPytorch3D.src.utils as utils
import mcubes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex


class VisualizeVoxel:

    def __init__(self):
        pass

    def visualize(self, voxel_grid, id, save_dir='', *args, **kwargs):
        """
        Renders a set 2D projections of a voxel grid

        :param save_dir:
        :param voxel_grid:
        :param id:
        """
        mesh = self._get_mesh_from_voxel(voxel_grid)
        frames = utils.get_mesh_images(mesh)
        filepath = save_dir + f'voxel_{id}.gif'
        imageio.mimsave(filepath, frames, fps=5)

    def _get_mesh_from_voxel(self, voxel_grid, voxel_size=32, min_value=-1.1, max_value=1.1):
        """
        Extracts mesh from voxel

        :param voxel_grid:
        :return: pytorch3d mesh
        """
        voxel_grid = voxel_grid > 0.1
        vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxel_grid.detach().cpu().squeeze().numpy()), isovalue=0.0)  # vertices=[V,3],
        # faces=[F,3]
        vertices = torch.tensor(vertices).float()
        faces = torch.tensor(faces.astype(int))  # make sure that you typecast faces to int
        color = [0.8, 0.8, 1]
        vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
        textures = torch.ones(size=(1, vertices.shape[0], 3))  # (1, N_v, 3)
        textures = textures * torch.tensor(color)  # (1, N_v, 3)
        mesh = Meshes([vertices], [faces], textures=TexturesVertex(textures))
        return mesh

    def __call__(self, *args, **kwargs):
        self.visualize(*args, **kwargs)
