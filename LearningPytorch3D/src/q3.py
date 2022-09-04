"""
Sample code to render a gif

Usage:
    python -m src.q3 --image_size 256 --output_path images/implicit_models/cow_texture.gif
"""
import sys
sys.path.append("..")
import imageio
import argparse

import pytorch3d
import torch

from LearningPytorch3D.src.utils import get_mesh_images
from LearningPytorch3D.starter.utils import get_device, get_mesh_renderer, load_mesh


def render_gif(
        cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None,
):
    my_images = []
    if device is None:
        device = get_device()
    renderer = get_mesh_renderer(image_size=image_size)

    vertices, faces = load_mesh(cow_path)

    color1 = torch.tensor([0, 0, 1])
    color2 = torch.tensor([1, 0, 0])
    z_min = vertices[vertices[:, 2:].argmin()][2]
    z_max = vertices[vertices[:, 2:].argmax()][2]

    color = torch.empty(size=(vertices.shape[0], 3))
    for i,(x, y, z) in enumerate(vertices):
        alpha = (z - z_min) / (z_max - z_min)
        color_curr = alpha * color2 + (1 - alpha) * color1
        color[i, :] = color_curr

    texture = color
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    # textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = texture.unsqueeze(0) #textures * color  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    return get_mesh_images(mesh=mesh, light_location=[[0, 0, -4]], device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/implicit_models/cow_texture.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    images = render_gif(cow_path=args.cow_path, image_size=args.image_size)
    imageio.mimsave(args.output_path, images, fps=5)
