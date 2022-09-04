"""
Usage:
    python -m src.q5_3 --render implicit --num_samples 100 --image_size 256 --output_path images/q5/torus_implicit.gif
"""
import sys
sys.path.append("..")
import argparse

import imageio
import mcubes
import pytorch3d
import torch

from LearningPytorch3D.src.utils import get_mesh_images
from LearningPytorch3D.starter.utils import get_device


def render_torus(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    R = 0.75
    r = 0.25
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)

    voxels = (torch.sqrt(X ** 2 + Y ** 2) - R)**2 + (Z**2) - (r**2)
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    #vertices = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )

    return get_mesh_images(mesh=mesh, light_location=[[0.0, 0.0, -4.0]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--render", type=str, default="implicit",)
    parser.add_argument("--output_path", type=str, default="images/q5/torus_implicit.gif")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    images = render_torus()
    imageio.mimsave(args.output_path, images, fps=5)