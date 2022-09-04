"""
Sample code to render a gif

Usage:
    python -m src.q2_1 --image_size 256 --output_path images/decoders/tetrahedron_render.gif
"""

import sys
sys.path.append("..")
import imageio
import argparse

import pytorch3d
import torch

from LearningPytorch3D.src.utils import get_mesh_images
from LearningPytorch3D.starter.utils import get_device, get_mesh_renderer


def render_gif(
     image_size=256, color=[0.7, 0.7, 1], device=None,
):
    my_images = []
    if device is None:
        device = get_device()


    vertices = torch.tensor([
        (2,  2,  2),
        (1, -1, -1),
        (-1, 1, -1),
        (-2, -1, 1)
    ]).type(torch.FloatTensor)

    faces = torch.tensor([
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]).type(torch.FloatTensor)

    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    return get_mesh_images(mesh=mesh, light_location=[[0, 0, 3]], T_relative=torch.tensor([0, 0, 4]), device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="images/decoders/tetrahedron_render.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    images = render_gif(image_size=args.image_size)
    imageio.mimsave(args.output_path, images, fps=5)
